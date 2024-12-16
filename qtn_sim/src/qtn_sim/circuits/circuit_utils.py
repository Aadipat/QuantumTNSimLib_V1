import numpy as np

# SINGLE QUBIT gates
# Hadamard
def H():
    return 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])


# Pauli gates
def X():
    return np.array([[0, 1], [1, 0]])


def Y():
    return np.array([[0, complex(0, -1)], [complex(0, 1), 0]])


def Z():
    return np.array([[1, 0], [0, -1]])


# Phase gate
def S():
    return np.array([[1, 0], [0, complex(0, 1)]])


# PI/8 gate
def T():
    return np.array([[1, 0], [0, np.exp((complex(0, 1) * np.pi / 4))]])


# RY gate
def RY(angle):
    return np.array([[np.cos(angle / 2.0), -np.sin(angle / 2.0)], [np.sin(angle / 2.0), np.cos(angle / 2.0)]])


#Get conjugate of gate
def Dagger(f):
    return np.conj(f)


# 2 QUBIT GATES
# Controlled not
def CNOT():
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    return np.reshape(cnot, (2, 2, 2, 2))


# Controlled Z
def CZ():
    cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    return np.reshape(cz, (2, 2, 2, 2))


# Controlled P
def CP(angle):
    elamdba = complex(np.cos(angle), np.sin(angle))
    cp = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, elamdba]])
    return np.reshape(cp, (2, 2, 2, 2))


# Swap gate
def SWAP():
    swapSV = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.reshape(swapSV, (2, 2, 2, 2))


# Controlled RY
def CRY(angle):
    cp = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(angle / 2.0), -np.sin(angle / 2.0)],
                   [0, 0, np.sin(angle / 2.0), np.cos(angle / 2.0)]])
    return np.reshape(cp, (2, 2, 2, 2))


# 3 QUBIT GATES
# Toffoli gate
def TOFFOLI():
    toff = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]
                        , [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]])
    return np.reshape(toff, (2, 2, 2, 2, 2, 2))



