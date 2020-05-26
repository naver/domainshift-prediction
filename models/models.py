
import os
import torch
from allennlp.common import Params
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, ElmoTokenEmbedder

from models.base.basetextclassifier import BaseTextClassifier
from models.base.basetagger import BaseSequenceTagger
from models.base.domainclassifier import *


def create_model(vocab: Vocabulary, embedding_dim: int,
                 hidden_dim: int, TaskModel: Model = BaseTextClassifier,
                 wemb: str = None, encoder_type: str = "lstm",
                 pretrained_model: BaseTextClassifier = None,
                 fix_pretrained_weights: bool = False, **kwargs) -> Model:
    """
    :param vocab: input / output vocabulary of the dataset
    :param embedding_dim:
    :param hidden_dim:
    :param TaskModel:  the model to apply
    :param encoder_type: GRU, LSTM
    :param wemb: type of word embeddings being used None, ELMO, Glove
    :param dropout:
    :param n_layers:
    :param pretrained_model: use a pretrained model as an input to copy the encoder layers from
    e.g. for building a domain classifier
    :param fix_pretrained_weights: whether to fix embeddings of the encoding layer or not
    (only if a pretrained model is provided)
    :return:
    """

    if wemb is None: wemb = "random"

    if wemb.lower() == "elmo":

        word_embeddings_params = Params({
                "embedding_dim": 100,
                "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                "trainable": False
              })

        elmo_params = Params({
                "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                "do_layer_norm": False,
                "dropout": 0.5,
                "requires_grad": False
              })

        token_embeddings = Embedding.from_params(vocab, word_embeddings_params)
        elmo_embeddings = ElmoTokenEmbedder.from_params(vocab, elmo_params)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings, "elmo": elmo_embeddings})

    elif wemb.lower() == "glove" or "http" in wemb or os.path.exists(wemb):

        if wemb.lower() == "glove":
            pretrained_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"
            embedding_dim = 100
        else:
            pretrained_file = wemb

        word_embeddings_params = Params({
                "embedding_dim": embedding_dim,
                "pretrained_file": pretrained_file,
                "trainable": False
        })

        token_embeddings = Embedding.from_params(vocab=vocab, params=word_embeddings_params)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings})

    else:
        token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size("tokens"),
                                     embedding_dim=embedding_dim)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings})

    embedding_dim = word_embeddings.get_output_dim()
    rnn_params = Params({"type": encoder_type,
                         "input_size": embedding_dim,
                         "hidden_size": hidden_dim,
                         "num_layers": kwargs["num_layers"],
                         "bidirectional": kwargs["bidirectional"]})

    del kwargs["num_layers"]
    del kwargs["bidirectional"]

    if TaskModel is BaseSequenceTagger or (TaskModel is SequenceDomainClassifier and kwargs["num_extra_rnn_layers"] > 0):
        rnn = Seq2SeqEncoder.from_params(rnn_params)
    else:
        rnn = Seq2VecEncoder.from_params(rnn_params)

    model = TaskModel(word_embeddings, rnn, vocab, **kwargs)

    # if a Pretrained model is provided
    # in the case of copying encoding representations from the task classifier to the domain classifier
    if pretrained_model is not None:

        # freezing embeddings of the encoder and the word embeddings
        model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
        model.word_embeddings.load_state_dict(pretrained_model.word_embeddings.state_dict())

        if fix_pretrained_weights:

            for p in model.encoder.parameters():
                p.requires_grad = False

            for p in model.word_embeddings.parameters():
                p.requires_grad = False

    return model







