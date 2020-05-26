# This file includes modifications to universal_dependencies_dataset_reader.py” 
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



from typing import Dict, Tuple, List, Iterator
import logging

from overrides import overrides
from conllu.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path

from readers.base_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from typing import TypeVar, Iterable, Tuple, Union


from allennlp.predictors import Predictor
from predictors.sequencetagging import SequenceTaggingPredictor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def lazy_parse(text: str, fields: Tuple = DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]


@DatasetReader.register("custom_universal_dependencies")
class UniversalDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_language_specific_pos: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos

    @overrides
    def _read(self, file_path: str, annotator: SequenceTaggingPredictor=None):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in  lazy_parse(conllu_file.read()):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if x["id"] is not None]
                sentence = [x["form"] for x in annotation]

                if annotator is None:
                    heads = [x["head"] for x in annotation]
                    tags = [x["deprel"] for x in annotation]
                    if self.use_language_specific_pos:
                        pos_tags = [x["xpostag"] for x in annotation]
                    else:
                        pos_tags = [x["upostag"] for x in annotation]

                    heads_tags = list(zip(tags, heads))

                else:
                    # if an annotator is given to the reader use labels exported by the reader
                    # as the annotation labels

                    annotated_instance = annotator.predict(sentence)
                    label_ids = annotated_instance["class"]
                    label_dict = annotator._model.vocab.get_index_to_token_vocabulary('pos')
                    pos_tags = [label_dict[i] for i in label_ids]

                    heads_tags = None
                yield self.text_to_instance(sentence, pos_tags)


    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: List[str],
                         upos_tags: List[str] = None,
                         dependencies: List[Tuple[str, int]] = None) -> Instance:
        """
        Parameters
        ----------
        sentence : ``List[str]``, required.
            The sentence in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing sentence, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in sentence], self._token_indexers)
        fields["sentence"] = tokens

        if upos_tags is not None:
            # labels
            fields["label"] = SequenceLabelField(upos_tags, tokens, label_namespace="pos")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                     tokens,
                                                     label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([int(x[1]) for x in dependencies],
                                                        tokens,
                                                        label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({"sentence": sentence, "pos": upos_tags})
        return Instance(fields)


@DatasetReader.register("custom_universal_dependencies_noun_classifier")
class UniversalDependenciesDatasetReaderBinary(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the words TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_language_specific_pos: bool = False,
                 positive_tags = ["NOUN","PROPN"],
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.positive_tags = positive_tags
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos

    @overrides
    def _read(self, file_path: str, annotator: SequenceTaggingPredictor=None):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in  lazy_parse(conllu_file.read()):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if x["id"] is not None]
                sentence = [x["form"] for x in annotation]

                if annotator is None:
                    heads = [x["head"] for x in annotation]
                    tags = [x["deprel"] for x in annotation]
                    if self.use_language_specific_pos:
                        pos_tags = [x["xpostag"] for x in annotation]
                    else:
                        pos_tags = [x["upostag"] for x in annotation]

                    # filter tags to binary (noun classification problem)
                    pos_tags = ["1" if i in self.positive_tags else "0" for i in pos_tags]
                    heads_tags = list(zip(tags, heads))

                else:
                    # if an annotator is given to the reader use labels exported by the reader
                    # as the annotation labels

                    annotated_instance = annotator.predict(sentence)
                    label_ids = annotated_instance["class"]
                    label_dict = annotator._model.vocab.get_index_to_token_vocabulary('pos')
                    pos_tags = [label_dict[i] for i in label_ids]

                    heads_tags = None
                yield self.text_to_instance(sentence, pos_tags)


    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: List[str],
                         upos_tags: List[str] = None,
                         dependencies: List[Tuple[str, int]] = None) -> Instance:
        """
        Parameters
        ----------
        sentence : ``List[str]``, required.
            The sentence in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing sentence, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in sentence], self._token_indexers)
        fields["sentence"] = tokens

        if upos_tags is not None:
            # labels
            fields["label"] = SequenceLabelField(upos_tags, tokens, label_namespace="pos")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                     tokens,
                                                     label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([int(x[1]) for x in dependencies],
                                                        tokens,
                                                        label_namespace="head_index_tags")

        fields["metadata"] = MetadataField({"sentence": sentence, "pos": upos_tags})
        return Instance(fields)


@DatasetReader.register("dd_custom_universal_dependencies")
class DDUniversalDependenciesDatasetReader(DatasetReader):
    """
    Reads a file in the conllu Universal Dependencies format.
    output 1 if it from domain 1 and outputs 0 if it is from domain 2

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        The token indexers to be applied to the sentence TextField.
    use_language_specific_pos : ``bool``, optional (default = False)
        Whether to use UD POS tags, or to use the language specific POS tags
        provided in the conllu format.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_language_specific_pos: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.use_language_specific_pos = use_language_specific_pos

    @overrides
    def _read(self, file_paths: str, annotator: Predictor = None) -> Iterator[Instance]:
        """
        Take 2 file paths separated by comma (sorry this is a hack to adapt to readers of allennlp)
        """

        # if `file_path` is a URL, redirect to the cache
        file_paths = file_paths.split(",")
        f1_path, f2_path = file_paths

        # count number of lines in f1 and f2 to make equal batches
        f1 = list(lazy_parse(open(f1_path).read()))
        f2 = list(lazy_parse(open(f2_path).read()))
        f1_n = len(f1)
        f2_n = len(f2)

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

        cf1 = 0
        cf2 = 0
        for c in range(0, f1_n + f2_n):
            if next(c) and cf1 < len(f1) or ((not next(c)) and cf2 >= len(f2)):
                annotation = f1[cf1]
                cf1 += 1
                label = "0"
            else:
                annotation = f2[cf2]
                cf2 += 1
                label = "1"

            # CoNLLU annotations sometimes add back in sentence that have been elided
            # in the original sentence; we remove these, as we're just predicting
            # dependencies for the original sentence.
            # We filter by None here as elided sentence have a non-integer word id,
            # and are replaced with None by the conllu python library.
            annotation = [x for x in annotation if x["id"] is not None]
            sentence = [x["form"] for x in annotation]
            sentence = [Token(x) for x in sentence]  # convert sentence into Token objects
            yield self.text_to_instance(sentence, label)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentence: List[Token], label: str = None) -> Instance:
        """
        Parameters
        ----------
        sentence : ``List[str]``, required.
            The words in the sentence to be encoded.
        label : ``str`` "0" if the document comes from domain 1 and "1" if it comes from
            domain 2

        Returns
        -------
        An instance containing words, and the domain label
        """

        sentence_field = TextField(sentence, self.token_indexers)
        field = {"sentence": sentence_field}

        if label:
            label_field = LabelField(label)
            field["label"] = label_field
        return Instance(field)
