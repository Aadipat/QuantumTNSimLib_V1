import numpy as np

from qtn_sim.circuits import *

npchars = [chr(ord('a')+i) for i in range(26)] + [chr(ord('A')+i) for i in range(26)]
opchars = [chr(i+48) for i in range(500)]
opchars.remove(">")

chars = opchars

INDICES = ""
valid_characters = [(65, 65+26), (97, 97+26), (192, 214), (216, 246), (248, 328), (330, 383), (477, 687), (913, 974), (1024, 1119)]
for tup in valid_characters:
    for i in range(tup[0], tup[1]):
        INDICES += chr(i)

chars = INDICES

def getChars(baseEinsumStrategy):
    # if baseEinsumStrategy is np.einsum:
    #     return npchars
    # return opchars
    return chars

def einsumForTensor(tensor : np.array, charIndexStart : int, strategy : str = None):
    """
    This gives the einsum string for a tensor. We need to pass the offset characters, the tensor itself
    strategy is a dummy string, not needed

    Returns:
        s : tensor string for einsum
    """
    s = ""
    for i in range(len(tensor.shape)):
        s += getChars(strategy)[charIndexStart+i]
    return s

def einsumForCircuit(tensorInputString : str, lines : list[int], circuit : QCircuit, strategy : str = None):
    """
    This method provides the einsum string for the circuit contraction on the state given as QCircuit object.
    We must pass the quantum state input string and this will be added
    We must also pass the list of qubit lines on which the state is dependendent on

    the strategy string is more of a dummy string, could be necessary for detrmining the characters for th einsum string.

    Returns: 
        s : einstein summation strigng for the circuit tensors
    """
    s = ""
    s += tensorInputString
    index = getChars(strategy).index(tensorInputString[-1]) + 1
    for i in range(len(circuit.gateList)):
        (gate,qubits) = circuit.gateList[i]
        gateInS = ""
        gateOutS = ""
        for line in qubits:
            gateInS += lines[line]
            lines[line] = getChars(strategy)[index]
            index += 1
            gateOutS += lines[line]
        s += "," + gateInS + gateOutS

    s += "->"
    s += tensorInputString[0]
    for line in lines:
        s += line
    s += tensorInputString[-1]
    return s
    

def applyGateTensorOnTensor(tensor, gate : np.array = np.array([]), qubits : list[int] = None, baseEinsumStrategy = np.einsum, es_str: str = None):
    """
    This method applies a tensor on another
    The first tensor is the state tensor or a tensor in the mps, the second being the gate tensor
    qubits is the lines and the tensors are contracted with the base einsum strategy
    we can also optionally pass the einsum string which will overide the contraction if provided.

    Returns: 
        tensorContracted : the contracted tensors
    """
    if qubits is None:
        qubits = []
    if es_str is not None:
        return baseEinsumStrategy(es_str, tensor,gate)

    # One strategy for unordered qubits, detect inversions and 
    # apply swap gates optimally before and after.
    
    es_str = ""
    mpsS = ""
    # Add MPS indices, row down
    for i in range(len(tensor.shape)):
        mpsS += getChars(baseEinsumStrategy)[i] 
    es_str += mpsS
    es_str += ','

    # Add gate indices, row down then column right to einstein summation string.
    # Add index if it is in qubits.
    mpoS = ""
    
    for i in qubits:
        mpoS += mpsS[i + 1]
    
    leftoverS = ""
    k = 1
    while len(mpoS) < len(gate.shape):
        c = getChars(baseEinsumStrategy)[getChars(baseEinsumStrategy).index(mpsS[-1]) + k]   #chr(ord(mpsS[-1])+k)
        mpoS += c
        leftoverS += c
        k += 1
    es_str += mpoS
    # Now we add the final result indices
    # We must map the gate indices leftover to output lines.

    outS = ""
    i = 0
    j = 0
    while i + j < len(leftoverS) + len(mpsS):
        if j >= len(mpsS):
            outS += leftoverS[i:]
            break
        if i >= len(leftoverS):
            outS += mpsS[j:]
            break
        if mpsS[j] in mpoS:
            outS += leftoverS[i]
            j += 1
            i += 1
        else:
            outS += mpsS[j]
            j += 1
    es_str += "->"
    es_str += outS
    # print(es_str)
    return baseEinsumStrategy(es_str, tensor,gate)





from abc import ABC, abstractmethod

# Abstract class for qSimulators

class QSimulator(ABC):
    
    @abstractmethod
    def get_state_vector(self):
        pass
    @abstractmethod
    def apply(self, gate : Gate, qubits : list[int] = None):
        pass
    @abstractmethod
    def applyCircuit(self, circuit : QCircuit):
        pass
    @abstractmethod
    def plot_prob(self):
        pass

