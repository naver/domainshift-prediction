

from typing import Dict, Union, Any

import torch
from torch.nn import CrossEntropyLoss
from torch import nn

from allennlp.common import Params
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from metrics.loss import Loss
from torch.nn import SmoothL1Loss, HingeEmbeddingLoss, CrossEntropyLoss
from metrics.confidence import Confidence


@Model.register("domain-classifier")
class DomainClassifier(Model):
    """
    a model that takes a sequence of word embeddings, transforms input words
    and give an output class for the whole input sequence
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 dropout: float = 0.5,
                 num_extra_layers: int = 0,
                 dd_hidden_dim=None
                 ) -> None:
        """

        :param word_embeddings: the embeddings to start with
        :param encoder:  the seq2seq transformer of embeddings can be LSTM for example
        :param vocab: dataset input and output vocabulary
        """

        super(DomainClassifier, self).__init__(vocab)

        self.word_embeddings = word_embeddings

        self.encoder = encoder

        if dd_hidden_dim is None:
            self.h_size = encoder.get_output_dim()
        else:
            self.h_size = dd_hidden_dim

        # Add extra layer of hidden linear layers with relu for the encoder output.
        if num_extra_layers > 0:
            extra_hiddens = []

            extra_hiddens += [nn.Linear(encoder.get_output_dim(), self.h_size), nn.ReLU(True)]
            for k in range(num_extra_layers-1):
                extra_hiddens += [nn.Linear(self.h_size, self.h_size), nn.ReLU(True)]

            self.extra_hiddens = nn.Sequential(*extra_hiddens)
        else:
            self.extra_hiddens = None

        # Linear layer to calculate domain class
        self.hidden2label = torch.nn.Linear(in_features=self.h_size,
                                            out_features=vocab.get_vocab_size('labels'))

        self.representations = self.extra_hiddens[-2] if self.extra_hiddens is not None else self.encoder

        self.dropout = nn.Dropout(dropout)

        self.criterion = CrossEntropyLoss()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "hinge-loss": Loss(HingeEmbeddingLoss()),
            "huber-loss": Loss(SmoothL1Loss()),
            "cross-entropy-loss": Loss(CrossEntropyLoss()),
            "perplexity": Confidence()
        }

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


@Model.register("sequence-domain-classifier")
class SequenceDomainClassifier(DomainClassifier):
    """
    a model that takes a sequence of word embeddings, transforms input words
    and give an output class for the whole input sequence
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Any,
                 vocab: Vocabulary,
                 dropout: float = 0.5,
                 num_extra_layers: int = 0,
                 num_extra_rnn_layers: int = 0,
                 extra_rnn_hidden_size: int = 150
                 ) -> None:
        """

        :param word_embeddings: the embeddings to start with
        :param encoder:  the seq2seq transformer of embeddings can be LSTM for example
        :param vocab: dataset input and output vocabulary
        """

        super(SequenceDomainClassifier, self).__init__(word_embeddings, encoder, vocab, dropout, num_extra_layers)

        if num_extra_rnn_layers > 0:
            # run a bidirectional lstm over the outputs of the task encoder pos tagger
            rnn_params = Params({"type": "lstm",
                                 "input_size": self.encoder.get_output_dim(),
                                 "hidden_size": extra_rnn_hidden_size,
                                 "num_layers": num_extra_rnn_layers,
                                 "bidirectional": True})

            self.extra_rnn = Seq2VecEncoder.from_params(rnn_params)
            self.extra_rnn_reshape = nn.Linear(self.extra_rnn.get_output_dim(), self.h_size)

        else:
            self.extra_rnn = None
            self.extra_rnn_reshape = None

        # linear linear without non-linearity to make sure shapes match


    def forward(self,
                sentence: Dict[str, torch.Tensor],
                label: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(sentence)

        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)

        if self.extra_rnn is not None:

            encoder_out = self.extra_rnn(encoder_out, mask)
            encoder_out = self.extra_rnn_reshape(encoder_out)

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