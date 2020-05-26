"""
code to measure domain shift using a set of metrics and run
over Sentiment Analysis Datasets 
"""

import random
import numpy as np
import os
import argparse

import torch
import torch.optim as optim

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.file_utils import cached_path
from allennlp.data.iterators import BucketIterator

from allennlp.training import Trainer
from metrics.utils import evaluate

from models.models import create_model
from models.base.basetextclassifier import BaseTextClassifier
from models.base.domainclassifier import DomainClassifier

from readers.reviews_readers import *
from predictors.classification import DocumentClassificationPredictor
from models.calibrators import Calibrator


OPTIM_config = {
    "sgd": optim.SGD,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop
}


def reader_params(reader_name: str) -> Params:
    """
    helper function to prepare params for dataset_reader
    """
    if args.W_EMB is None:
        dataset_params = Params({
            "type": reader_name,
            "binary_output": True
        })

    elif args.W_EMB.lower() == "elmo":
        dataset_params = Params({
            "type": reader_name,
            "binary_output": True,
            "token_indexers": {
                "tokens": {
                    "type": "single_id",
                    "lowercase_tokens": True
                },
                "elmo": {
                    "type": "elmo_characters"
                }
            }
        })
    else:
        dataset_params = Params({
            "type": reader_name,
            "binary_output": True
        })

    return dataset_params

def add_arguments(parser):
    """
    helper function to add arguments
    :param parser: an argparse object
    :return: parser
    """
    config = [
        # TASK Model Params
        ["EMBEDDING_DIM", 300, int, False, "dimension of word embeddings"],
        ["HIDDEN_DIM", 150, int, False, "dimension of hidden embeddings of RNN"],
        ["ENCODER_RNN", "lstm", str, False, "Encoder RNN either LSTM or GRU"],
        ["W_EMB", None, str, False, "word embeddings type glove / elmo / URL / PATH leave empty = random"],
        ["DATASET_READER", None, str, True, "Registered Allenlp dataset reader for this dataset"],
        ["N_LAYERS", 1, int, False, "number of hidden layers for the RNN of the task model"],
        ["N_LINEAR_LAYERS", 2, int, False, "number of hidden linear layers for the task model"],
        ["DROPOUT", 0.5, float, False, "value of dropout"],

        # Domain Discriminator Model Params  all start by DD
        ["DD_HIDDEN_DIM", 250, int, False, "dimension of hidden layer"],
        ["DD_DATASET_READER", None, str, True, "Allennlp Reader of for domain discrimination"],
        ["DD_DROPOUT", 0.5, float, False, "value of dropout"],
        ["DD_EXTRA_N_LAYERS", 10, int, False, "number of hidden layers for both the task model and the domain classifier"],

        # Training Params
        ["BATCH_SIZE", 64, int, False, "Batch size"],
        ["PATIENCE", 3, int, False, "patience"],
        ["EPOCHS", 30, int, False, "epochs"],
        ["LR", 0.01, float, False, "learning rate"],
        ["OPTIM", "adam", str, False, "optimizer type"],
        ["GRAD_CLIP", 3, float, False, "value of gradient clipping threshold"],

        # Domain Discriminator Training Params
        ["DD_BATCH_SIZE", 64, int, False, "Batch size"],
        ["DD_PATIENCE", 3, int, False, "patience"],
        ["DD_EPOCHS", 30, int, False, "epochs"],
        ["DD_LR", 0.01, float, False, "learning rate"],
        ["DD_OPTIM", "adam", str, False, "optimizer type"],
        ["DD_GRAD_CLIP", 10, float, False, "value of gradient clipping threshold"],

        # Server Params
        ["GPU", 0, int, False, "device id"],
        ["SEED", 999, int, False, "random seed"],
        ["DATASET_CACHED_DIR", "./dataset/cached/", str, False,
         "Directory where dataset will be downloaded and chached"],
        ["SERIALIZATION_DIR", None, str, False, "Directory where model will be serialized and results will be logged"],
        ["DD_DATASET_CACHED_DIR", "./dataset/dd_cached/", str, False,
         "Directory where domain classification will be downloaded and chached"],
        ["DD_SERIALIZATION_DIR", None, str, False,
         "Directory where model will be serialized and results will be logged"]
    ]

    for v in config:
        parser.add_argument("--" + v[0], default=v[1], type=v[2], required=v[3], help=v[4])

    return parser


def reset_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a classifier over classification dataset')
    parser.add_argument('--d1train', help="path to training dataset", required=True)
    parser.add_argument('--d1train2', help="path to a 2nd auxilary training dataset", required=True)
    parser.add_argument('--d1valid', help="path to validation dataset", required=True)
    parser.add_argument('--d1test', help="path to test dataset", required=True)

    parser.add_argument('--d2train', help="path to training dataset or multiple ones separated by comma", required=True)
    parser.add_argument('--d2valid', help="path to validation dataset or multiple ones separated by comma",
                        required=True)
    parser.add_argument('--d2test', help="path to test dataset or multiple ones separated by comma", required=True)
    parser.add_argument('--out', help="path to output file to log experiments results in",
                        default="all_test_metrics.txt", required=False)

    parser = add_arguments(parser)

    args = parser.parse_args()

    reset_seed(args.SEED)
    GPU = args.GPU


    ##############################
    #  Preparing Dataset Readers #
    ##############################

    # Preparing datasets for original task "d1"
    d1_train_dataset_path = args.d1train
    d1_train2_dataset_path = args.d1train2
    d1_valid_dataset_path = args.d1valid
    d1_test_dataset_path = args.d1test

    reader = DatasetReader.from_params(reader_params(args.DATASET_READER))
    d1_train_dataset = reader.read(d1_train_dataset_path)
    d1_train2_dataset = reader.read(d1_train2_dataset_path)
    d1_valid_dataset = reader.read(d1_valid_dataset_path)
    d1_test_dataset = reader.read(d1_test_dataset_path)

    # Preparing datasets for original task "d2"
    d2_train_dataset_path = args.d2train
    d2_valid_dataset_path = args.d2valid
    d2_test_dataset_path = args.d2test

    reader = DatasetReader.from_params(reader_params(args.DATASET_READER))
    d2_train_dataset = reader.read(d2_train_dataset_path)
    d2_valid_dataset = reader.read(d2_valid_dataset_path)
    d2_test_dataset = reader.read(d2_test_dataset_path)

    # Domain discrimination
    dd_reader = DatasetReader.from_params(reader_params(args.DD_DATASET_READER))

    dd_train_dataset = dd_reader.read("%s,%s" % (d1_train_dataset_path, d2_train_dataset_path))
    dd_valid_dataset = dd_reader.read("%s,%s" % (d1_valid_dataset_path, d2_valid_dataset_path))
    dd_test_dataset = dd_reader.read("%s,%s" % (d1_test_dataset_path, d2_test_dataset_path))

    vocab = Vocabulary.from_instances(d1_train_dataset + d1_train2_dataset + d1_valid_dataset, max_vocab_size=30000)

    print("""
    #######################################################
    # Starting with optimizing a model on the task itself #
    #######################################################
    """)

    # MODEL
    reset_seed(args.SEED)
    model = create_model(vocab, args.EMBEDDING_DIM, args.HIDDEN_DIM, num_layers=args.N_LAYERS,
                         TaskModel=BaseTextClassifier,
                         n_linear_layers=args.N_LINEAR_LAYERS,
                         encoder_type=args.ENCODER_RNN, bidirectional=True, wemb=args.W_EMB, dropout=args.DROPOUT)

    # Training
    iterator = BucketIterator(batch_size=args.BATCH_SIZE, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    optimizer = OPTIM_config[args.OPTIM.lower()](model.parameters(), lr=args.LR)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=d1_train_dataset,
                      validation_dataset=d1_valid_dataset,
                      patience=args.PATIENCE,
                      num_epochs=args.EPOCHS,
                      cuda_device=args.GPU,
                      serialization_dir=None,
                      validation_metric="+accuracy",
                      grad_clipping=args.GRAD_CLIP,
                      )

    d1_train_metrics = trainer.train()

    print("""
    ########################################
    # calibrate confidence scores of model #
    ########################################
    """)

    model_calibrator = Calibrator(model, vocab)

    # Training
    iterator = BucketIterator(batch_size=args.BATCH_SIZE, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    optimizer = OPTIM_config[args.OPTIM.lower()](model_calibrator.parameters(), lr=args.LR)

    calib_trainer = Trainer(model=model_calibrator,
                            optimizer=optimizer,
                            iterator=iterator,
                            train_dataset=d1_valid_dataset,
                            validation_dataset=d1_train2_dataset,
                            patience=args.PATIENCE,
                            num_epochs=args.EPOCHS,
                            cuda_device=args.GPU,
                            serialization_dir=None,
                            validation_metric="-loss",
                            grad_clipping=args.GRAD_CLIP,
                        )

    # d1_calib_metrics = calib_trainer.train()

    print("""
    ######################################
    # Optimizing The domain discriminator#
    ######################################
    """)

    # Domain discriminator Model
    reset_seed(args.SEED)
    dd_model = create_model(vocab, args.EMBEDDING_DIM, args.HIDDEN_DIM, num_layers=args.N_LAYERS,
                            num_extra_layers=args.DD_EXTRA_N_LAYERS,
                            TaskModel=DomainClassifier,
                            encoder_type=args.ENCODER_RNN, bidirectional=True,
                            wemb=args.W_EMB, dropout=args.DD_DROPOUT,
                            pretrained_model=model,
                            dd_hidden_dim=args.DD_HIDDEN_DIM,
                            fix_pretrained_weights=True)

    # Training the Domain Discriminator
    dd_iterator = BucketIterator(batch_size=args.DD_BATCH_SIZE, sorting_keys=[("sentence", "num_tokens")])
    dd_iterator.index_with(vocab)

    dd_optimizer = OPTIM_config[args.DD_OPTIM.lower()](dd_model.parameters(), lr=args.DD_LR)

    dd_trainer = Trainer(model=dd_model,
                         optimizer=dd_optimizer,
                         iterator=dd_iterator,
                         train_dataset=dd_train_dataset,
                         validation_dataset=dd_valid_dataset,
                         patience=args.DD_PATIENCE,
                         num_epochs=args.DD_EPOCHS,
                         cuda_device=args.GPU,
                         serialization_dir=None,
                         validation_metric="+accuracy",
                         grad_clipping=args.DD_GRAD_CLIP,
                         )

    dd_metrics = dd_trainer.train()

    print("""
    ###############################################
    # Optimizing The generic domain discriminator #
    ###############################################
    """)

    # Domain discriminator Model
    reset_seed(args.SEED)
    g_dd_model = create_model(vocab, args.EMBEDDING_DIM, args.HIDDEN_DIM, num_layers=args.N_LAYERS,
                              TaskModel=DomainClassifier,
                              dd_hidden_dim=args.DD_HIDDEN_DIM,
                              num_extra_layers=args.DD_EXTRA_N_LAYERS,
                              encoder_type=args.ENCODER_RNN, bidirectional=True,
                              wemb=args.W_EMB, dropout=args.DD_DROPOUT)

    # Training the Domain Discriminator
    g_dd_iterator = BucketIterator(batch_size=args.DD_BATCH_SIZE, sorting_keys=[("sentence", "num_tokens")])
    g_dd_iterator.index_with(vocab)

    g_dd_optimizer = OPTIM_config[args.DD_OPTIM.lower()](g_dd_model.parameters(), lr=args.DD_LR)

    g_dd_trainer = Trainer(model=g_dd_model,
                         optimizer=g_dd_optimizer,
                         iterator=g_dd_iterator,
                         train_dataset=dd_train_dataset,
                         validation_dataset=dd_valid_dataset,
                         patience=args.DD_PATIENCE,
                         num_epochs=args.DD_EPOCHS,
                         cuda_device=args.GPU,
                         serialization_dir=None,
                         validation_metric="+accuracy",
                         grad_clipping=args.DD_GRAD_CLIP,
                         )

    g_dd_metrics = g_dd_trainer.train()

    print("""
    ##############################
    # d1d2d1: Reverse classifier #
    ##############################
    """)

    reset_seed(args.SEED)

    r_reader = DatasetReader.from_params(reader_params(args.DATASET_READER))
    predictor = DocumentClassificationPredictor(model=model, dataset_reader=r_reader)

    r_d2_train_dataset = reader.read(d2_train_dataset_path, annotator=predictor)
    r_d2_valid_dataset = reader.read(d2_valid_dataset_path, annotator=predictor)
    r_d2_test_dataset = reader.read(d2_test_dataset_path, annotator=predictor)

    r_model = create_model(vocab, args.EMBEDDING_DIM, args.HIDDEN_DIM, num_layers=args.N_LAYERS,
                                  encoder_type=args.ENCODER_RNN,
                           bidirectional=True,
                           wemb=args.W_EMB, dropout=args.DROPOUT)

    # Training
    d2iterator = BucketIterator(batch_size=args.BATCH_SIZE, sorting_keys=[("sentence", "num_tokens")])
    d2iterator.index_with(vocab)

    r_optimizer = OPTIM_config[args.OPTIM.lower()](r_model.parameters(), lr=args.LR)

    trainer = Trainer(model=r_model,
                      optimizer=r_optimizer,
                      iterator=d2iterator,
                      train_dataset=r_d2_train_dataset,
                      validation_dataset=r_d2_valid_dataset,
                      patience=args.PATIENCE,
                      num_epochs=args.EPOCHS,
                      cuda_device=args.GPU,
                      serialization_dir=None,
                      validation_metric="+accuracy",
                      grad_clipping=args.GRAD_CLIP,
                      )

    r_train_metrics = trainer.train()

    print("""
    ####################################
    # d1d1d1: Reverse classifier on d1 #
    ####################################
    """)

    reset_seed(args.SEED)

    r_opt_reader = DatasetReader.from_params(reader_params(args.DATASET_READER))

    r_opt_d1_train_dataset = reader.read(d1_train2_dataset_path, annotator=predictor)
    r_opt_d1_valid_dataset = reader.read(d1_valid_dataset_path, annotator=predictor)
    r_opt_d1_test_dataset = reader.read(d1_test_dataset_path)

    r_opt_model = create_model(vocab, args.EMBEDDING_DIM, args.HIDDEN_DIM, num_layers=args.N_LAYERS,
                               TaskModel=BaseTextClassifier,
                               bidirectional=True,
                               encoder_type=args.ENCODER_RNN, wemb=args.W_EMB, dropout=args.DROPOUT)

    r_opt_optimizer = OPTIM_config[args.OPTIM.lower()](r_opt_model.parameters(), lr=args.LR)

    trainer = Trainer(model=r_opt_model,
                      optimizer=r_opt_optimizer,
                      iterator=iterator,
                      train_dataset=r_opt_d1_train_dataset,
                      validation_dataset=r_opt_d1_valid_dataset,
                      patience=args.PATIENCE,
                      num_epochs=args.EPOCHS,
                      cuda_device=args.GPU,
                      serialization_dir=None,
                      validation_metric="+accuracy",
                      grad_clipping=args.GRAD_CLIP,
                      )

    r_opt_train_metrics = trainer.train()

    ###########
    # TESTING #
    ###########

    # Domain Discriminator start model  dds
    # a model that learns proper representations to discriminate between D1 and D2
    # this can serve like the upper bound of DD results
    reset_seed(args.SEED)

    d1_test_dataset = reader.read(d1_test_dataset_path)
    d2_test_dataset = reader.read(d2_test_dataset_path)

    # classifier suffering domain shift
    reset_seed(args.SEED)
    d1_on_d1_test_metrics = evaluate(model, d1_test_dataset, iterator, GPU)
    d1_on_d2_test_metrics = evaluate(model, d2_test_dataset, iterator, GPU)  # classifier suffering domain shift

    # calibrated classifier suffering domain shift
    reset_seed(args.SEED)
    calib_d1_on_d1 = evaluate(model_calibrator, d1_test_dataset, iterator, GPU)
    calib_d1_on_d2 = evaluate(model_calibrator, d2_test_dataset, iterator, GPU)

    reset_seed(args.SEED)
    dd_test_metrics = evaluate(dd_model, dd_test_dataset, dd_iterator, GPU)  # classifier between two domains
    g_dd_test_metrics = evaluate(g_dd_model, dd_test_dataset, g_dd_iterator, GPU)  # classifier between two domains

    reset_seed(args.SEED)
    # r_on_train_test_metrics = evaluate(r_model, d1_train_dataset, d2iterator, GPU)  # RCA on d1_Train dataset
    r_test_metrics = evaluate(r_model, d1_test_dataset, d2iterator, GPU)  # RCA on d1_test

    reset_seed(args.SEED)
    # r_opt_on_train_test_metrics = evaluate(r_opt_model, d1_train_dataset, iterator, GPU)  # optimal RCA on d1_train
    r_opt_test_metrics = evaluate(r_opt_model, d1_test_dataset, iterator, GPU)  # optimal RCA on d1_test

    all_test_metrics = [[d1_train_dataset_path, d2_train_dataset_path,
                         d1_on_d1_test_metrics, d1_on_d2_test_metrics,
                         calib_d1_on_d1, calib_d1_on_d2,
                         dd_test_metrics, g_dd_test_metrics,
                         # r_on_train_test_metrics,
                         r_test_metrics,
                         # r_opt_on_train_test_metrics,
                         r_opt_test_metrics
                         ]
                        ]

    s = ""
    for d1_name, d2_name, d1d1, d1d2, cald1d1, cald1d2, dd, g_dd , r, r_opt in all_test_metrics:

        a_measure = (1 - dd["huber-loss"]) * 100
        a_measure_loss = (dd["hinge-loss"]) * 100
        g_a_measure = (1 - g_dd["huber-loss"]) * 100
        g_a_measure_loss = (g_dd["hinge-loss"]) * 100

        s += "d1: %s\n" % d1_train_dataset_path \
             + "d2: %s\n" % d2_name \
             + "clf_d1_on_d1: %s\n" % d1d1["accuracy"] \
             + "clf_d1_on_d2: %s\n" % d1d2["accuracy"] \
             + "clf drop:%f2\n" % (d1d1["accuracy"] - d1d2["accuracy"]) \
             + "a-measure-huber:%s\n" % a_measure \
             + "a-measure-hinge:%s\n" % a_measure_loss \
             + "dd-accuracy:%s\n" % dd["accuracy"] \
             + "g-a-measure-huber:%s\n" % g_a_measure \
             + "g-a-measure-hinge:%s\n" % g_a_measure_loss \
             + "g-dd-accuracy:%s\n" % g_dd["accuracy"] \
             + "d1_on_d1_confidence:%s\n" % d1d1["confidence"] \
             + "d1_on_d2_confidence:%s\n" % d1d2["confidence"] \
             + "cal_on_d1:%s\n" % cald1d1["confidence"] \
             + "cal_on_d2:%s\n" % cald1d2["confidence"] \
             + "PARAM: %s\n" % str([args.W_EMB, args.SEED, args.HIDDEN_DIM, args.N_LAYERS, args.N_LINEAR_LAYERS, args.DROPOUT, args.LR]) \
             + "DD_PARAM: %s \n" % str([args.DD_HIDDEN_DIM, args.DD_DROPOUT, args.DD_EXTRA_N_LAYERS, args.DD_LR]) \
            + "RCA: %s \n" % r["accuracy"] \
            + "OPTIMAL RCA: %s \n" % r_opt["accuracy"] \
            + "d1d1 RCA drop: %s \n" % (d1d1["accuracy"] - r["accuracy"]) \
            + "d1d1 OPTIMAL_RCA drop: %s \n" % (d1d1["accuracy"] - r_opt["accuracy"]) \

    print(s)
    with open(args.out, "a") as f:
        f.write(s)
