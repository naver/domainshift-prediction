

from typing import Dict

import torch
from torch.nn import CrossEntropyLoss
from torch import nn


from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from metrics.loss import Loss
from torch.nn import SmoothL1Loss, HingeEmbeddingLoss, CrossEntropyLoss
from metrics.confidence import Confidence


@Model.register("text-classifier")
class BaseTextClassifier(Model):
    """
    a model that takes a sequence of word embeddings, transforms input words
    and give an output class for the whole input sequence
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 dropout: float = 0.5,
                 n_linear_layers=1,
                 ) -> None:
        """

        :param word_embeddings: the embeddings to start with
        :param encoder:  the seq2seq transformer of embeddings can be LSTM for example
        :param vocab: dataset input and output vocabulary
        """

        super(BaseTextClassifier, self).__init__(vocab)

        self.word_embeddings = word_embeddings

        self.encoder = encoder

        # Representations this is the layer that is just above the last layer and the non linearity (hidden[-1])
        # is is used to calculate FID score, and similar metrics that's why we expose it into self.representations
        # class attribute
        self.representations = self.encoder

        if n_linear_layers > 0:
            extra_hiddens = []
            for k in range(n_linear_layers):
                extra_hiddens += [nn.Linear(self.encoder.get_output_dim(), self.encoder.get_output_dim()), nn.ReLU(True)]
            self.extra_hiddens = nn.Sequential(*extra_hiddens)
        else:
            self.extra_hiddens = None

        self.hidden2label = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                            out_features=vocab.get_vocab_size('labels'))

        # self.accuracy = CategoricalAccuracy()
        self.criterion = CrossEntropyLoss()

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "hinge-loss": Loss(HingeEmbeddingLoss()),
            "huber-loss": Loss(SmoothL1Loss()),
            "cross-entropy-loss": Loss(CrossEntropyLoss()),
            "confidence": Confidence()
        }

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                label: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)

        if self.extra_hiddens is not None:
            encoder_out = self.extra_hiddens(encoder_out)

        logits = self.hidden2label(self.dropout(encoder_out))

        output = {"logits": logits,
                  "probs": torch.nn.functional.softmax(logits, dim=1),
                  "class": torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1)
                  }

        if label is not None:

            output["loss"] = self.criterion(logits, label)

            for metric_name, metric in self.metrics.items():

                    metric(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {}

        for metric_name, metric in self.metrics.items():

                metrics[metric_name] = metric.get_metric(reset)

        return metrics