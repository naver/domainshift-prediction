
from typing import Iterator, List, Dict

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

# from allennlp.data.dataset_readers import DatasetReader
from readers.base_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.predictors import Predictor
from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader

@DatasetReader.register('reviews')
class SentimentDatasetReader(DatasetReader):
    """
    Dataset reader of amazon sentiment analysis dataset
    on the form of tab separated
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 binary_output: bool = False) -> None:

        super().__init__(lazy=False)
        self.binary_output = binary_output
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tokenize = SpacyWordSplitter().split_words

    def _read(self, file_path: str, annotator: Predictor=None) -> Iterator[Instance]:
        """
        take a file name of an amazon dataset .tsv file
        review is on 13 column and the label is in
        and process the file and stream Instances
        """

        with open(file_path) as f:

            for l in f:
                l = l.strip()
                l = l.split("\t")
                try:
                    if len(l) != 2:  # check only two columns per line to avoid reading malformed lines
                        continue

                    # remove empty reviews in amazon dataset less than two letters
                    if len(l[0].strip()) < 2:
                        continue

                    if l[1] not in "12345":
                        continue
                except:
                    continue

                # get the sentence text
                sentence = self.tokenize(l[0].strip().lower())

                # Binarize the output usual score is from 1->5
                # Make 1,2,3 negative  -- 4,5 positive
                # (this is to accommodate the unbalance between positive and neg)
                if annotator is None:
                    # get the label review score
                    label = l[1]

                    if self.binary_output:
                        if label in "45":
                            label = "1"
                        else:
                            label = "0"
                else:

                    label = annotator.predict_instance(self.tokens_to_instance(sentence))
                    label = str(label["class"])

                try:
                    assert len(sentence) > 2
                    assert label in ["1", "0"]
                except AssertionError as e:
                    continue

                yield self.tokens_to_instance(sentence, label)

    def tokens_to_instance(self, tokens: List[Token], label: str = None) -> Instance:

        sentence_field = TextField(tokens, self.token_indexers)
        field = {"sentence": sentence_field}

        if label:
            label_field = LabelField(label)
            field["label"] = label_field

        return Instance(field)

    def text_to_instance(self, document: str, label: str = None) -> Instance:

        tokens = self.tokenize(document.strip().lower())
        sentence_field = TextField(tokens, self.token_indexers)
        field = {"sentence": sentence_field}

        if label:
            label_field = LabelField(label)
            field["label"] = label_field

        return Instance(field)


@DatasetReader.register('dd-reviews')
class DD_AmazonSentimentDatasetReader(DatasetReader):
    """
    Dataset reader of amazon sentiment analysis dataset
    Takes two separate file paths one of D1 and one D2
    Ignore the original labels and provide a label == 0 if from D1
    and label == 1 if from D1

    Provides equal number of samples for now between D1 and D2

    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, binary_output: bool = False) -> None:

        super().__init__(lazy=False)
        self.binary_output = binary_output
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tokenize = SpacyWordSplitter().split_words

    def _read(self, file_paths: str) -> Iterator[Instance]:
        """
        Take 2 file paths separated by comma (sorry this is a hack to adapt to readers of allennlp)
        """

        file_paths = file_paths.split(",")
        f1, f2 = file_paths

        # count number of lines in f1 and f2 to make equal batches
        f1_n = len(open(f1).readlines())
        f2_n = len(open(f2).readlines())

        def next(c):
            """
            grauantee equal dist of F1 and F2
            Return True if F1 False if F2
            """
            frac = f1_n/float(f2_n) if f1_n > f2_n else f2_n/float(f1_n)
            frac = round(frac)

            if frac != 1:
                return (c % frac) != 0
            else:
                return (c % 2) == 0

        with open(f1) as f1:
            with open(f2) as f2:

                for c in range(0, f1_n+f2_n):
                    l = f1.readline() if next(c) else f2.readline()

                    # get the label either it is d1 or d2
                    label = "0" if next(c) else "1"

                    # in case one of the files if over already keep going on the other file
                    if not l and next(c):
                        l = f2.readline()
                        label = "1"
                    if not l and not next(c):
                        l = f1.readline()
                        label = "0"
                    l = l.split("\t")

                    # get the sentence text
                    sentence = self.tokenize(l[0].strip().lower())

                    try:
                        assert len(sentence) > 2
                        assert label in ["1", "0"]
                    except AssertionError as e:
                        continue

                    yield self.tokens_to_instance(sentence, label)

    def tokens_to_instance(self, tokens: List[Token], label: str = None) -> Instance:

        sentence_field = TextField(tokens, self.token_indexers)
        field = {"sentence": sentence_field}

        if label:
            label_field = LabelField(label)
            field["label"] = label_field

        return Instance(field)
