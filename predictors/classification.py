

from overrides import overrides
from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader


@Predictor.register('document-classifier')
class DocumentClassificationPredictor(Predictor):
    """
    Predictor for the :class:` models.base.basetextclassifier
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        self._model = model
        self._dataset_reader = dataset_reader

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"source": "..."}``.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:

        document = inputs["source"]

        instance = self._dataset_reader.text_to_instance(document=document)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return {"instance": self.predict_instance(instance), "all_labels": all_labels}

