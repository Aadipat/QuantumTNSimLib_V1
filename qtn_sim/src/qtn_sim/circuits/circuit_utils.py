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

# G gate for the W circuit:
def G(p):
    return np.array([[complex(np.sqrt(p),0), complex(-np.sqrt(1-p),0)], [complex(np.sqrt(1-p),0), complex(np.sqrt(p),0)]])


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



# Expanded matrix, to create a controlled gate for any unitary matrix.
def expandWithIdentity(matrix):
    original_shape = matrix.shape
    new_shape = (original_shape[0] * 2, original_shape[1] * 2)
    # Create a new identity matrix of double the size
    result = np.eye(new_shape[0], dtype=matrix.dtype)
    # Place the original matrix in the bottom right corner
    result[-original_shape[0]:, -original_shape[1]:] = matrix
    return result



# 3 QUBIT GATES
# Toffoli gate
def TOFFOLI():
    toff = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]
                        , [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]])
    return np.reshape(toff, (2, 2, 2, 2, 2, 2))



