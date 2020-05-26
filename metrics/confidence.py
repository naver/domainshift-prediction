from typing import Dict, Optional, Tuple, Union
from overrides import overrides

import torch
from torch.nn import SmoothL1Loss
from allennlp.training.metrics.metric import Metric


@Metric.register("confidence")
class Confidence(Metric):
    """
    This :class:`Metric` breaks with the typical ``Metric`` API and just stores values that were
    computed in some fashion outside of a ``Metric``.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    ``Metric`` API.
    """
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

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

        # perplexity is the max probability of the output classes
        confidence = torch.max(torch.nn.functional.softmax(predictions, dim=1), dim=1)[0].mean()
        self._total_value += list(self.unwrap_to_tensors(confidence))[0]
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
