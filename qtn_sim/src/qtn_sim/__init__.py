# __init__.py

# Import main classes and functions
from .circuits import *
from .tn_simulators import *
from .optimisers import *
from .qtn_plotting import *
# You can define what should be imported when using "from package import *"
__all__ = [name for name in dir() if not name.startswith('_')]

# Package metadata
__version__ = '0.0.1'
__author__ = 'Aadi Patwardhan'
__description__ = 'A package to plot performance of tensor network simulations'