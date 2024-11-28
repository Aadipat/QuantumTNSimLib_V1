# __init__.py

# Import main classes and functions from MPSQLib.py
from .MPSQLib import QuantumMPS

# Import main classes and functions from OneTensorQLib.py
from .OneTensorQLib import QuantumTensor

from .einsumLib import (einsumForTensor, einsumForCircuit, apply_Gate_On_Tensor)
from .tensorUtils import  (svQiskitStyleToMine)

# You can define what should be imported when using "from package import *"
__all__ = ['QuantumMPS', 'QuantumTensor', 'einsumForCircuit', 'einsumForTensor', 'apply_Gate_On_Tensor', 'svQiskitStyleToMine']

# Package metadata
__version__ = '0.1.0'
__author__ = 'Aadi Patwardhan'
__description__ = 'A quantum computing package for tensor network quantum computation simulation'