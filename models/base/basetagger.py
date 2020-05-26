
from typing import Dict

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import Sequential
from torch import nn


from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from metrics.loss import SequenceLoss
from torch.nn import SmoothL1Loss, HingeEmbeddingLoss, CrossEntropyLoss
from metrics.confidence import Confidence
from allennlp.nn.util import sequence_cross_entropy_with_logits


@Model.register("tagger")
class BaseSequenceTagger(Model):
    """
    a model for sequence tagging
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 dropout: float = 0.5
                 ) -> None:
        """

        :param word_embeddings: the embeddings to start with
        :param encoder:  the seq2seq transformer of embeddings can be LSTM for example
        :param vocab: dataset input and output vocabulary
        """

        super(BaseSequenceTagger, self).__init__(vocab)

        self.word_embeddings = word_embeddings

        self.encoder = encoder

        # Representations this is the layer that is just above the last layer and the non linearity (hidden[-1])
        # is is used to calculate FID score, and similar metrics that's why we expose it into self.representations
        # class attribute
        self.representations = self.encoder

        self.hidden2tags = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                           out_features=vocab.get_vocab_size('pos'))

        # self.accuracy = CategoricalAccuracy()
        self.criterion = sequence_cross_entropy_with_logits

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            # "huber-loss": Loss(SmoothL1Loss()),
            "confidence": Confidence()
        }

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                label: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:

        words = sentence
        mask = get_text_field_mask(words)

        embeddings = self.word_embeddings(words)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2tags(self.dropout(encoder_out))

        output = {"logits": logits,
                  "probs": torch.nn.functional.softmax(logits, dim=-1),
                  "class": torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
                  }

        if label is not None:

            output["loss"] = self.criterion(logits, label,
                                            weights=torch.ones_like(label))

            for metric_name, metric in self.metrics.items():

                    metric(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {}

        for metric_name, metric in self.metrics.items():

                metrics[metric_name] = metric.get_metric(reset)

        return metrics
