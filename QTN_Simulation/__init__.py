# __init__.py

# Import main classes and functions
from .QTN_circuits import *
from .TN_Simulators import *
from .QTN_Optimisers import *
from .QTN_plotting import *
# You can define what should be imported when using "from package import *"
__all__ = [name for name in dir() if not name.startswith('_')]

# Package metadata
__version__ = '0.1.0'
__author__ = 'Aadi Patwardhan'
__description__ = 'A package to plot performance of tensor network simulations'