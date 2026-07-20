from typing import Union, TypeVar

from .baselinedatabuilder import BaseLineDataBuilder
from .graphdatabuilder import GraphDataBuilder

DataBuilder = TypeVar(
    "DataBuilder",
    bound=Union[BaseLineDataBuilder, GraphDataBuilder]
)