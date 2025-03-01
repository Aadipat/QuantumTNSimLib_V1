import json
import inspect
import random
import math

import numpy as np

from .circuit_utils import *

class Gate:
    def __init__(self, gateId, tensor, params=[]):
        self.id = gateId
        self.tensor = tensor
        self.params = params

class SingleQubitGate(Gate):
    def __init__(self, gateId, tensor, params=[]):
        super().__init__(gateId, tensor, params)


class TwoQubitGate(Gate):
    def __init__(self, gateId, tensor, params=[]):
        super().__init__(gateId, tensor, params)


class ThreeQubitGate(Gate):
    def __init__(self, gateId, tensor, params=[]):
        super().__init__(gateId, tensor, params)


class HGate(SingleQubitGate):
    def __init__(self):
        super().__init__("H", H())


class XGate(SingleQubitGate):
    def __init__(self):
        super().__init__("X", X())


class YGate(SingleQubitGate):
    def __init__(self):
        super().__init__("Y", Y())


class ZGate(SingleQubitGate):
    def __init__(self):
        super().__init__("Z", Z())


class SGate(SingleQubitGate):
    def __init__(self):
        super().__init__("S", S())


class TGate(SingleQubitGate):
    def __init__(self):
        super().__init__("T", T())


class RYGate(SingleQubitGate):
    def __init__(self, params):
        super().__init__("RY", RY(params[0]), params)


class GGate(SingleQubitGate):
    def __init__(self, params):
        super().__init__("G", G(params[0]), params)


class CNOTGate(TwoQubitGate):
    def __init__(self):
        super().__init__("CNOT", CNOT())


class SWAPGate(TwoQubitGate):
    def __init__(self):
        super().__init__("SWAP", SWAP())


class CPGate(TwoQubitGate):
    def __init__(self, params):
        super().__init__("CP", CP(params[0]), params)


class CZGate(TwoQubitGate):
    def __init__(self):
        super().__init__("CZ", CZ())


class CRYGate(TwoQubitGate):
    def __init__(self, params):
        super().__init__("CRY", CRY(params[0]), params)


class TOFFOLIGate(ThreeQubitGate):
    def __init__(self):
        super().__init__("CCX", TOFFOLI())


# Produce Controlled gate given a unitary gate
# Puts the gate in the right bottom corner. 
class CUnitaryTensorGate(Gate):

    def __init__(self, uGate : Gate):

        dim = len(uGate.tensor.shape)/2

        newDim = dim+1
        
        gateT = uGate.tensor
        
        matrix = np.reshape(gateT, (int(np.pow(2,dim)),int(np.pow(2,dim))))

        newMatrix = expandWithIdentity(matrix)

        tensor = np.reshape(newMatrix, tuple([2 for i in range(int(2*newDim))]))

        super().__init__("C" + uGate.id, tensor)


class QCircuit:
    def __init__(self, circuit=None):
        if circuit is None:
            circuit = []
        self.gateList = circuit

    def addGate(self, gate, indices):
        self.gateList.append((gate, indices))

    def toString(self):
        s = ""
        for (gate, indices) in self.gateList:
            s += "{" + str(gate.id) + ", " + str(indices) + "}\n"
        return s

    def toJSONDict(self):
        gatesJson = []
        for (gate, indices) in self.gateList:
            gateDict = {
                "id": gate.id,
                "params": gate.params,
                "indices": indices
            }
            gatesJson.append(gateDict)
        circuitJson = {
            "gates": gatesJson
        }
        return circuitJson


class QFTCircuit(QCircuit):
    def __init__(self, n):
        circuit = []
        for i in range(n):
            circuit.append((HGate(), [i]))
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                circuit.append((CPGate([angle]), [i, j]))

        for i in range(n // 2):
            circuit.append((SWAPGate(), [i, n - i - 1]))
        super().__init__(circuit)


class GHZCircuit(QCircuit):
    def __init__(self, n):
        circuit = [(HGate(), [0])] + [(CNOTGate(), [i, i + 1]) for i in range(n - 1)]
        super().__init__(circuit)


class WCircuitLinear(QCircuit):
    def __init__(self, n):
        
        circuit = []
        circuit.append((XGate(), [0]))

        def B(p, n1,n2):
            B = []
            UnitaryGate = CUnitaryTensorGate(GGate([1.0/p]))
            B.append((UnitaryGate, [n1,n2]))
            B.append((SWAPGate(), [n1,n2]))
            B.append((CNOTGate(), [n1,n2]))
            B.append((SWAPGate(), [n1,n2]))
            return B
        
        for i in range(n-1):
            
            for (g,i) in B(n-i, i,i+1):
                circuit.append((g,i))

        super().__init__(circuit)


circuitMap = {
    "H": HGate,
    "X": XGate,
    "Y": YGate,
    "Z": ZGate,
    "S": SGate,
    "T": TGate,
    "RY": RYGate,
    "CNOT": CNOTGate,
    "SWAP": SWAPGate,
    "CP": CPGate,
    "CZ": CZGate,
    "CRY": CRYGate,
    "TOFFOLI": TOFFOLIGate,
    "CCX": TOFFOLIGate
}

singleQGateMap = {
    "H": HGate,
    "X": XGate,
    "Y": YGate,
    "Z": ZGate,
    "S": SGate,
    "T": TGate,
    "RY": RYGate
}

twoQGateMap = {
    "CNOT": CNOTGate,
    "SWAP": SWAPGate,
    "CP": CPGate,
    "CZ": CZGate,
    "CRY": CRYGate
}

threeQGateMap = {
    "TOFFOLI": TOFFOLIGate,
    "CCX": TOFFOLIGate
}


# json representation:
# Circuit:
#     gates[]
#
# Gate:
#     id:
#     parameters:
#     indices:

def circuitFromJSONDict(JSONCircuitDict):
    circuit = []
    for gateApp in JSONCircuitDict["gates"]:
        gate = np.array([])
        if len(gateApp["params"]) > 0:
            gate = circuitMap[gateApp["id"]](gateApp["params"])
        else:
            gate = circuitMap[gateApp["id"]]()
        indices = gateApp["indices"]
        circuit.append((gate, indices))
    return QCircuit(circuit)


def getRandomIndices(n, k):
    indices = random.sample(range(n), k)
    return sorted(indices)


def getRandomCircuit(n, depth, gatesUsed = circuitMap, probablityOfAdjacent=0.0, withNonAdjacent = True, asJson=False):
    """
    Generates a random quantum circuit.
    Parameters:
    n (int): Number of qubits in the circuit.
    depth (int): Maximum depth of the circuit.
    gatesUsed (dict): Dictionary of gates to be used in the circuit. Defaults to circuitMap. str (gate.id) -> Gate object
    withNonAdjacent (bool): If True, allows non-adjacent qubit gates. Defaults to True.
    asJson (bool): If True, returns the circuit as a JSON string. Defaults to False.
    Returns:
    tuple: A tuple containing the QCircuit object and its JSON representation (if asJson is True).
    """
    if not withNonAdjacent:
        probablityOfAdjacent = 1.0

    circuit = []
    for d in range(int((random.random() * depth) + 1)):
        gate = np.array([])
        if n == 1:
            gateId = random.choice(list(set(singleQGateMap.keys()).intersection(gatesUsed.keys())))
        elif n == 2:
            gateId = random.choice(list(set((singleQGateMap | twoQGateMap).keys()).intersection(gatesUsed.keys())))
        else:
            gateId = random.choice(list(set(circuitMap.keys()).intersection(gatesUsed.keys())))
        gateFunc = circuitMap[gateId]
        signature = inspect.signature(gateFunc)
        parameters = []
        if len(signature.parameters.values()) >= 1:
            for i in range(len(signature.parameters.values())):
                angle = random.random() * math.pi / 2
                parameters.append(angle)
            gate = gateFunc(parameters)
        elif len(signature.parameters.values()) == 0:
            gate = gateFunc()
        indices = []
        p = random.random()
        if p > probablityOfAdjacent:
            indices = getRandomIndices(n, int(len(gate.tensor.shape) / 2))
        else:
            i = random.choice([i for i in range(n+1-int(len(gate.tensor.shape)/2))])
            indices = [i + j for j in range(int(len(gate.tensor.shape)/2))]
        circuit.append((gate, indices))
    circuitJson = QCircuit(circuit).toJSONDict()
    if asJson:
        circuitJson = json.dumps(circuitJson)
    return QCircuit(circuit), circuitJson


def createCircuits(numberOfCircuits, numberOfQubits, maxCircuitDepth, gatesUsed = circuitMap,  probablityOfAdjacent = 0.0, withNonAdjacent = True, filePath=None):
    """
    Generates a specified number of random quantum circuits and optionally saves them to a JSON file.
    Args:
        numberOfCircuits (int): The number of circuits to generate.
        numberOfQubits (int): The number of qubits in each circuit.
        maxCircuitDepth (int): The maximum depth of each circuit.
        gatesUsed (dict, optional): A mapping of gates to be used in the circuits. Defaults to circuitMap. str (gate.id) -> Gate object
        probablityOfAdjacent (float, optional): The probability of using adjacent gates. Defaults to 0.0.
        withNonAdjacent (bool, optional): Whether to include non-adjacent gates in the circuits. Defaults to True.
        filePath (str, optional): The file path to save the generated circuits in JSON format. If None, the circuits are not saved to a file. Defaults to None.
    Returns:
        list: A list of generated quantum circuits.
    """
    if not withNonAdjacent:
        probablityOfAdjacent = 1.0
    
    circuits = []
    circuitsJson = []
    for i in range(numberOfCircuits):
        circuit, circuitJson = getRandomCircuit(numberOfQubits, maxCircuitDepth, gatesUsed, probablityOfAdjacent=probablityOfAdjacent)
        circuitsJson.append(circuitJson)
        circuits.append(circuit)
    circuitsJsonDump = {
        "circuits": circuitsJson
    }
    if filePath is not None:
        with open(filePath, 'w', encoding='utf-8') as f:
            json.dump(circuitsJsonDump, f, indent=4)
    return circuits


def readCircuits(filePath):
    """
    Reads quantum circuits from a JSON file and returns them as a list of Qcircuit objects.
    Args:
        filePath (str): The path to the JSON file containing the circuits.
    Returns:
        list: A list of Qcircuit objects parsed from the JSON file.
    """
    circuitsJson = None
    with open(filePath, 'r') as file:
        circuitsJson = json.load(file)
    circuitsFromJson = []
    for circuitJson in circuitsJson["circuits"]:
        circuitsFromJson.append(circuitFromJSONDict(circuitJson))
    return circuitsFromJson
