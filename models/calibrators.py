"""
Model for calibrating NN models for classification
Code based on the paper: https://arxiv.org/pdf/1706.04599.pdf
and the github repo https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
"""


from typing import Dict

import torch
from torch import nn

from allennlp.models import Model

from allennlp.training.metrics import CategoricalAccuracy
from metrics.loss import Loss
from torch.nn import SmoothL1Loss, HingeEmbeddingLoss, CrossEntropyLoss
from metrics.confidence import Confidence
from models.base.basetagger import BaseSequenceTagger
from allennlp.nn.util import sequence_cross_entropy_with_logits

class Calibrator(Model):

    def __init__(self, model, vocab) -> None:
        super(Calibrator, self).__init__(vocab)

        self.model = model
        self.vocab = vocab
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        if type(model) is BaseSequenceTagger:

            self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "confidence": Confidence(),
            }

            self.criterion = sequence_cross_entropy_with_logits

        else:
            self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "hinge-loss": Loss(HingeEmbeddingLoss()),
                "huber-loss": Loss(SmoothL1Loss()),
                "cross-entropy-loss": Loss(CrossEntropyLoss()),
                "confidence": Confidence()
            }

            self.criterion = CrossEntropyLoss()

        # freeze original model param and keep only the temperature to be trainable
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                label: torch.Tensor = None, **kwargs):

        output = self.model(sentence)
        logits = output["logits"]

        calibrated_logits = self.temperature_scale(logits)

        output = {
            "logits": logits,
            "calibrated_logits": calibrated_logits
        }

        if label is not None:

            if self.criterion is sequence_cross_entropy_with_logits:
                output["loss"] = self.criterion(calibrated_logits, label, weights=torch.ones_like(label))
            else:
                output["loss"] = self.criterion(calibrated_logits, label)

            for metric_name, metric in self.metrics.items():
                metric(calibrated_logits, label)

        return output

    def temperature_scale(self, logits) -> Dict[str, torch.Tensor]:

        temperature = self.temperature.unsqueeze(1).expand(*logits.shape)
        return (logits / temperature)


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {}

        for metric_name, metric in self.metrics.items():

                metrics[metric_name] = metric.get_metric(reset)

        return metrics