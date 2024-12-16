# __init__.py

# Import main classes and functions from mps_qlib.py
from .mps_qlib import (QuantumMPS, split_tensor_SVD)

# Import main classes and functions from one_tensor_qlib.py
from .one_tensor_qlib import QuantumTensor

from .einsum_lib import (einsumForTensor, einsumForCircuit, applyGateTensorOnTensor)
from .tensor_utils import  (svQiskitStyleToMine)

# You can define what should be imported when using "from package import *"
__all__ = ['QuantumMPS', 'QuantumTensor', 'einsumForCircuit', 'einsumForTensor', 'split_tensor_SVD', 'applyGateTensorOnTensor', 'svQiskitStyleToMine']

# Package metadata
__version__ = '0.1.0'
__author__ = 'Aadi Patwardhan'
__description__ = 'A quantum computing package for tensor network quantum computation simulation'