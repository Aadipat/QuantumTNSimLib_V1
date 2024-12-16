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


# class WCircuit(QCircuit):
#     def __init__(self, n):
#         super().__init__(Wcircuit(n))


# DEPRECATED! NOT WORKING beyond n = 3!!!
# def Wcircuit(n):
#     circuit = []
#     theta = 2 * np.arccos(1 / np.sqrt(n))
#     circuit.append((RY(theta), [0]))
#     for i in range(1, n):
#         theta_i = 2 * np.arccos(1 / np.sqrt(n - i))
#         circuit.append((CRY(theta_i), [i - 1, i]))
#         for j in range(i - 1, -1, -1):
#             circuit.append((CNOT(), [j, i]))
#     return circuit

#
# def WRecursive(n):
#     circuit = QCircuit()
#     if n == 1:
#         circuit.addGate(XGate(), [0])
#         return circuit
#
#     # Create W state for n-1 qubits
#     sub_w = WRecursive(n - 1)
#     mini = n
#     for (gate,indices) in sub_w.gateList:
#         mini = min(mini, np.min(indices))
#     sW = QCircuit()
#     for i in range(len(sub_w.gateList)):
#         new_indices = [j - mini for j in sub_w.gateList[i][1]]
#         sW.addGate(sub_w.gateList[i][0],new_indices)
#
#     circuit.gateList += sW.gateList
#
#     # Apply rotation to distribute amplitude
#     theta = 2 * np.arcsin(1 / np.sqrt(n))
#     circuit.addGate(RYGate([theta]), [n-1])
#
#     # Distribute amplitude to the new qubit
#     for i in range(n - 1):
#         circuit.addGate(CNOTGate(), [i, n-1])
#
#     return circuit


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


def getRandomCircuit(n, depth, asJson=False):
    circuit = []
    for d in range(int((random.random() * depth) + 1)):
        gate = np.array([])
        if n == 1:
            gateId = random.choice(list(singleQGateMap.keys()))
        elif n == 2:
            gateId = random.choice(list((singleQGateMap | twoQGateMap).keys()))
        else:
            gateId = random.choice(list(circuitMap.keys()))
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
        indices = getRandomIndices(n, int(len(gate.tensor.shape) / 2))
        circuit.append((gate, indices))
    circuitJson = QCircuit(circuit).toJSONDict()
    if asJson:
        circuitJson = json.dumps(circuitJson)
    return QCircuit(circuit), circuitJson


def createCircuits(numberOfCircuits, numberOfQubits, maxCircuitDepth, filePath=None):
    circuits = []
    circuitsJson = []
    for i in range(numberOfCircuits):
        circuit, circuitJson = getRandomCircuit(numberOfQubits, maxCircuitDepth)
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
    circuitsJson = None
    with open(filePath, 'r') as file:
        circuitsJson = json.load(file)
    circuitsFromJson = []
    for circuitJson in circuitsJson["circuits"]:
        circuitsFromJson.append(circuitFromJSONDict(circuitJson))
    return circuitsFromJson
