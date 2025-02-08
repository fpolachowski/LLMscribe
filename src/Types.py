from enum import Enum
from typing import List
from dataclasses import dataclass


class SimilarityMeasure(Enum):
    COSINE = 1
    EULER = 2

@dataclass
class Document():
    content : str
    embedding : List[float] = None