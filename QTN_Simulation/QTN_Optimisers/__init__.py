# __init__.py

# Import main classes and functions
from .optimiserLib import *
# You can define what should be imported when using "from package import *"
__all__ = ['optEinsumRandomGreedy', 'applyCircuitSequentially', 'applyCircuitSequentiallyAdjacently']

# Package metadata
__version__ = '0.1.0'
__author__ = 'Aadi Patwardhan'
__description__ = 'A package with circuit optimisers for my simulation package'