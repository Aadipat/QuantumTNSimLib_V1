import numpy as np
import matplotlib.pyplot as plt

def truncate(U : np.array, Sigma : np.array, Vh : np.array, bond_dimension : int):
    """
    Pass the U, S, Vh obtained by SVD and the bond dimension,
    Computes the truncation using bond dimensions

    Returns:
        U, S, V : the truncated tensors
    """
    b = len(Sigma)
    Vh = np.reshape(Vh, (b, 2, -1))
    U = np.reshape(U, (-1, 2, b))

    b = np.min([b, bond_dimension])
    return U[:, :, :b], Sigma[:b], Vh[:b]


def split_tensor_SVD(bond_dimension : int, n : int, new_Tensor : np.array = np.array([]), leftDim : int = 1, rightDim : int = 1):
    """
    Pass the bond dimension, number of qubits represented by tensor, contracted tensor (e.g. gate applied already), 
    left and right bond dimensions.
    Computes the SVD splitting for MPS

    Returns:
        tensors_split : the state split by svd into n tensors 
    """
    tensors_split = []
    splits = n - 1
    last_B_dim = leftDim

    for _ in range(splits):
        # Reshape into a matrix
        new_Tensor = np.reshape(new_Tensor, (last_B_dim * 2, -1))
        U, Sigma, Vh = np.linalg.svd(new_Tensor, full_matrices=False)
        U, Sigma, Vh = truncate(U, Sigma, Vh, bond_dimension)

        U = np.reshape(U, (last_B_dim, 2, -1))
        tensors_split.append(U)

        new_Tensor = np.tensordot(np.diag(Sigma), Vh, 1)

        last_B_dim = Vh.shape[0]

    new_Tensor = np.reshape(new_Tensor, (-1, 1))
    U = np.reshape(new_Tensor, (-1, 2, rightDim))
    tensors_split.append(U)
    return tensors_split


def svQiskitStyleToMine(sv : np.array):
    """
    Statevectors are np.arrays
    Converts the Qiskit statevector style to mine
    Returns:
        s : state vector My package style
    """
    # Qiskit: last..3rd,2nd,1st
    # Mine: 1st2nd3rd...last
    n = int(np.log2(len(sv)))
    newSv = [0 for _ in range(len(sv))]
    for s in range(len(sv)):
        b = bin(s)
        string = b[2:]
        for i in range(n-len(string)):
            string = "0" + string
        newS = string[::-1]
        newInt = eval('0b' + newS)
        newSv[s] = sv[newInt]
    # print(newSv)
    # print(sv)
    return np.array(newSv)

def plot_prob_dist(prob_dist : np.array):
    """
    Plots the distribution of probabilities for states from the given np statevector
    """
    x_lbs = []

    num = len(prob_dist.shape)

    i = 0
    probs = np.ravel(prob_dist)
    for _ in probs:
        s = "{:0" + str(num) + "b}"
        x_lbs.append(f"|" + s.format(i) + ">")
        i += 1

    plt.bar(x_lbs, probs)
    plt.show()