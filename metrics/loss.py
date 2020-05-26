# This file includes modifications to softmax_loss.py” 
# distributed through the GitHub library https://github.com/allenai/allennlp 
# under this license https://github.com/allenai/allennlp/blob/master/LICENSE.
# Copyright with respect to the modifications: Copyright 2020 Naver Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from typing import Dict, Optional, Tuple, Union
from overrides import overrides

import torch
from allennlp.training.metrics.metric import Metric
from torch.nn import SmoothL1Loss, HingeEmbeddingLoss, CrossEntropyLoss
"""


@Metric.register("loss")
class Loss(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self, loss) -> None:
        """

        :param loss: loss to be calculated for eg.   HingeEmbeddingLoss()
        """
        self._total_value = 0.0
        self._count = 0
        self.loss = loss

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """

        if type(self.loss) == CrossEntropyLoss:

            loss = self.loss(predictions, gold_labels)

        else:

            label_onehot = torch.ones_like(predictions) * -1
            label_onehot.scatter_(1, gold_labels.unsqueeze(1), 1)

            prob = torch.nn.functional.softmax(predictions, dim=1)
            loss = self.loss(prob, label_onehot)

        self._total_value += list(self.unwrap_to_tensors(loss))[0]
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return float(average_value)

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0


@Metric.register("sequenceloss")
class SequenceLoss(Metric):
    """
    Computes the loss metric across a sequence
    """
    def __init__(self, loss) -> None:
        """

        :param loss: loss to be calculated for eg.   HingeEmbeddingLoss()
        """
        self._total_value = 0.0
        self._count = 0
        self.loss = loss

    @overrides
    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """

        if type(self.loss) == CrossEntropyLoss:

            loss = self.loss(predictions, gold_labels)

        else:

            label_onehot = torch.ones_like(predictions) * -1
            label_onehot.scatter_(1, gold_labels.unsqueeze(1), 1)

            prob = torch.nn.functional.softmax(predictions, dim=-1)
            loss = self.loss(prob, label_onehot)

        self._total_value += list(self.unwrap_to_tensors(loss))[0]
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The average of all values that were passed to ``__call__``.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return float(average_value)

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0
