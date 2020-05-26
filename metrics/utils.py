
import numpy as np
from scipy import linalg

from typing import Dict, Any, Iterable

import torch
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator

from allennlp.commands.evaluate import evaluate as allennlp_evaluate
from models.base import basetextclassifier

def evaluate(model: basetextclassifier,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int) -> Dict[str, Any]:
    
    results = allennlp_evaluate(model, instances, data_iterator, cuda_device)
    return results

