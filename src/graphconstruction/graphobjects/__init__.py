"""
GraphObjects

Provides fundamental classes that may hold data required for the computation of graph structures.
"""

from .graphconfig import GraphConfig, TopKConfig
from .graphstructure import GraphStructure
from .graphobject import GraphObject

import logging
logger = logging.getLogger(__name__)