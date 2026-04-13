from typing import TypeVar, Union

from .baselineloader.baselinedataloadermanager import BaseLineDataLoaderManager
# from .shallowloader.shallowdataloadermanager import ShallowDataLoaderManager
from .deepdataloaders.deepdataloader import DeepDataLoaderManager 
from .deepdataloaders.graphdataloader import GraphDataLoaderManager

DLM = TypeVar('DLM', bound=Union[BaseLineDataLoaderManager, DeepDataLoaderManager, GraphDataLoaderManager])