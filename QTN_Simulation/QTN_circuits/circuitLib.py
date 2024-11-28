import numpy as np
import json


# SINGLE QUBIT gates
# Hadamard
def H():
    return 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
# Pauli gates
def X():
    return np.array([[0, 1], [1, 0]])
def Y():
    return np.array([[0, complex(0,-1)], [complex(0,1), 0]])
def Z():
    return np.array([[1, 0], [0, -1]])
# Phase gate
def S():
    return np.array([[1, 0], [0, complex(0,1)]])
# PI/8 gate
def T():
    return np.array([[1, 0], [0, np.exp((complex(0,1)*np.pi/4))]])
# RY gate
def RY(angle):
    return np.array([[np.cos(angle/2.0), -np.sin(angle/2.0)], [np.sin(angle/2.0), np.cos(angle/2.0)]])

#Get conjugate of gate
def Dagger(f):
    return np.conj(f)

# 2 QUBIT GATES
# Controlled not
def CNOT():
    cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    return np.reshape(cnot, (2,2,2,2))
# Controlled Z
def CZ():
    cz = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
    return np.reshape(cz, (2,2,2,2))
# Controlled P
def CP(angle):
    elamdba = complex(np.cos(angle),np.sin(angle))
    cp = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,elamdba]])
    return np.reshape(cp, (2,2,2,2))
# Swap gate
def SWAP():
    swapSV = np.array([[1,0,0,0],[0,0,1,0], [0,1,0,0], [0,0,0,1]])
    return np.reshape(swapSV,(2,2,2,2))
# Controlled RY
def CRY(angle):
    cp = np.array([[1,0,0,0],[0,1,0,0],[0,0,np.cos(angle/2.0),-np.sin(angle/2.0)],[0,0,np.sin(angle/2.0),np.cos(angle/2.0)]])
    return np.reshape(cp, (2,2,2,2))

# 3 QUBIT GATES
# Toffoli gate
def TOFFOLI():
    toff = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0]
                     ,[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
                     [0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]])
    return np.reshape(toff,(2,2,2,2,2,2))

# Some circuits
def GHZcircuit(n):
    circuit = [(H(),[0])] + [(CNOT(),[i,i+1]) for i in range(n-1)]
    return circuit

def QFTcircuit(n):
    circuit = []
    for i in range(n):
        circuit.append((H(), [i]))
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            circuit.append((CP(angle), [i,j]))
            
    for i in range(n // 2):
        circuit.append((SWAP(),[i, n - i - 1]))
    return circuit

def Wcircuit(n):
    circuit = []
    theta = 2 * np.arccos(1/np.sqrt(n))
    circuit.append((RY(theta), [0]))
    for i in range(1, n):
        theta_i = 2 * np.arccos(1/np.sqrt(n-i))
        circuit.append((CRY(theta_i), [i-1, i]))
        for j in range(i-1, -1, -1):
            circuit.append((CNOT(),[j, i]))
    return circuit



circuitMap = {
    "H":H,
    "X":X,
    "Y":Y,
    "Y": Z,
    "S": S,
    "T": T,
    "RY":RY,
    "CNOT":CNOT,
    "SWAP":SWAP,
    "CP":CP,
    "CZ":CZ,
    "CRY":CRY,
    "TOFFOLI":TOFFOLI,
    "CCX":TOFFOLI
}

def circuitFromJSON(JSONCircuit):
    circuit = []
    y = json.loads(JSONCircuit)
    for gateApp in y["gates"]:
        gate = circuitMap[gateApp["id"]]()
        indices = gateApp["indices"]
        circuit.append((gate, indices))
    return circuit