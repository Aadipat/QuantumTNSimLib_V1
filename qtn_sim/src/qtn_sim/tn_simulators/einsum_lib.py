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

def einsumForTensor(tensor,charIndexStart, strategy):
    s = ""
    for i in range(len(tensor.shape)):
        s += getChars(strategy)[charIndexStart+i]
    return s

def einsumForCircuit(tensorInputString, lines, circuit:QCircuit, strategy):
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
    

def applyGateTensorOnTensor(tensor, gate = np.array([]), qubits=None, baseEinsumStrategy = np.einsum, es_str: str = None):
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
