from typing import List
from QTN_Simulation.QTN_Optimisers.optimiserLib import *
from .einsumLib import *
import numpy as np
import matplotlib.pyplot as plt
import opt_einsum as oe

# This lib allows users to create MPS states,
# apply gates on certain qubits and
# plot probabilities using numpy einstein summation and MPS networks.

class QuantumMPS():

    # MPS Constructor
    def __init__(self,n, bond_dimension = 3, einsumOptimiser = oe.contract, baseEinsumStrategy = oe.contract, tensors = None, state_vector = None ):
        self.bond_dimension = bond_dimension
        self.einsumOptimiser = einsumOptimiser
        self.baseEinsumStrategy = baseEinsumStrategy
        self.tensors = None
        self.n = n
        if tensors is not None:
            # if initialised as mps
            self.tensors = tensors
            return
        if state_vector is not None:
            l = [1]
            for i in range(int(np.log2(len(state_vector)))):
                l.append(2)
            l.append(1)
            oneTensor = np.reshape(state_vector, tuple(l))
            self.tensors = split_tensor_SVD(self.bond_dimension, self.n, oneTensor)
            return
        self.tensors = [np.reshape(np.array([1,0]),(1,2,1)) for i in range(n)]
        return
    # Get state tensor string and lines for circuit
    def getTensorStringAndLines(self):
        s = ""
        lines = []
        for t in range(len(self.tensors)):
            tensorS = einsumForTensor(self.tensors[t],t*3, self.einsumOptimiser)
            s += tensorS
            if t < len(self.tensors)-1 :
                s += ","
            lines.append(tensorS[1])
        return s,lines
    # Convert to 1 tensor representation. 
    def convert_To_One_Tensor(self):
        # We contract on all the intermediate indices
        tensor = self.contract_tensors([i for i in range(len(self.tensors))])
        return tensor
    # Contract adjacent mps
    def contract_tensors(self, indices = []):
        # If the mps nodes are not entangled => vectors. 
        # We must outer product them to form the tensor.
        if(len(indices) < 2):
            return self.tensors[indices[0]]
        
        # We contract on specific the intermediate indices
        tensor = self.tensors[indices[0]]
        t = indices[1]
        k = 1
        while(k < len(indices)):
            # Need to build ein_sum string.
            s = ""
            o = ""
            for i in range(len(tensor.shape)):
                s += getChars(self.baseEinsumStrategy)[i] # chr(ord('a') + i)
                if i is not len(tensor.shape) -1:
                    o += getChars(self.baseEinsumStrategy)[i]
            s += ","
            for i in range(len(self.tensors[indices[k]].shape)):
                s += getChars(self.baseEinsumStrategy)[i + len(tensor.shape) -1] #chr(ord('a') + i + len(tensor.shape) - 1)
                if i != 0:
                    o += getChars(self.baseEinsumStrategy)[i + len(tensor.shape) -1] 
            s += "->" + o
            tensor = self.baseEinsumStrategy(s, tensor, self.tensors[indices[k]], optimize='greedy')
            k += 1
        return tensor
    # Apply a certain circuit list(gate,indices)
    def applyCircuit(self,circuit:List[tuple]):
        newTensors = []
        if self.einsumOptimiser is applyCircuitSequentially:
            newTensors = self.einsumOptimiser(self, circuit)
        elif self.einsumOptimiser is applyCircuitSequentiallyAdjacently:
            newTensors = self.einsumOptimiser(self, circuit)
    
        else:
            #Apply as one einsum
            tensorS,lines = self.getTensorStringAndLines()
            einsum_str = einsumForCircuit(tensorS,lines,circuit,self.einsumOptimiser)
            # print(len(einsum_str.split(",")))
            tensors = self.tensors + [c[0] for c in circuit]
            tensor = self.einsumOptimiser(einsum_str,*tensors,optimize='greedy')
            newTensors = split_tensor_SVD(self.bond_dimension,self.n, tensor, 1, 1)
        self.tensors = newTensors
    # Apply gate on certain qubits
    def apply(self, gate = np.array([]), qubits=[]):
        # To apply a gate, we must contract tensors with the gate 
        # We contract each tensor with the resepective index of the gate on the tensor 
        # then we use SVD to split the the tensor. 

        # We contract the adjacent tensors
        # Then contract with the gate
        contraction_list = [i for i in range(qubits[0], qubits[-1]+1)]
        split_num = len(contraction_list)
        tensor = self.contract_tensors(contraction_list)
        qubitsShifted = [q - qubits[0] for q in qubits]
        new_Tensor = apply_Gate_On_Tensor(tensor, gate, qubitsShifted, self.baseEinsumStrategy)

        self.tensors[qubits[0]] = new_Tensor
        # Then use svd to split the tensors again.
        leftBdim = 1
        if qubits[0] != 0:
            leftBdim = self.tensors[qubits[0]-1].shape[-1]
        rightBdim = 1
        if qubits[-1] != self.n-1:
            rightBdim = self.tensors[qubits[-1]+1].shape[0]
        tensors_split = []
        if(split_num > 1):
            tensors_split = split_tensor_SVD(self.bond_dimension,split_num, new_Tensor, leftBdim, rightBdim)
            tensors = []
            i = 0
            j = qubits[0]
            m = j
            k = qubits[-1] + 1
            while i < j:
                tensors.append(self.tensors[i])
                i += 1
            while j < k:
                tensors.append(tensors_split[j - m])
                j += 1
            while k < self.n:
                tensors.append(self.tensors[k])
                k += 1
            self.tensors = tensors
            return      
        
        tensors = self.tensors
        tensors[qubits[0]] = new_Tensor
        self.tensors = tensors
    # Convert to state vector
    def get_state_vector(self):
        return abs(np.ravel(np.square(np.squeeze(self.convert_To_One_Tensor()))))
    # Plot probabilities. 
    def plot_prob(self):

        prob_distr = self.get_state_vector()

        x_lbls = []

        num = self.n
        i = 0
        probs = np.ravel(prob_distr)
        for elem in probs:
            s = "{:0"+str(num)+"b}"
            x_lbls.append(f"|" + s.format(i) + ">")
            i +=1

        plt.bar(x_lbls, probs)
        plt.show()


def truncate(U,S,Vh, bond_dimension):
    b = len(S)
    Vh = np.reshape(Vh,(b, 2, -1))
    U = np.reshape(U,(-1, 2, b))

    b = np.min([b, bond_dimension])
    return U[:, :, :b], S[:b], Vh[:b]

def split_tensor_SVD(bond_dimension, n, new_Tensor = np.array([]), leftDim = 1, rightDim = 1):
    tensors_split = []
    splits = n-1
    lastBdim = leftDim

    for i in range(splits):
        # Reshape into a matrix
        new_Tensor = np.reshape(new_Tensor, (lastBdim*2,-1))
        U, S, Vh = np.linalg.svd(new_Tensor, full_matrices=False)
        U, S, Vh = truncate(U,S,Vh, bond_dimension)

        U = np.reshape(U, (lastBdim, 2, -1))
        tensors_split.append(U)
    
        new_Tensor = np.tensordot(np.diag(S), Vh, 1)

        lastBdim = Vh.shape[0]
    
    new_Tensor = np.reshape(new_Tensor, (-1,1))
    U, S, Vh = np.linalg.svd(new_Tensor, full_matrices=False)
    # U = np.reshape(U, (-1, 2, rightDim))
    U = np.reshape(new_Tensor, (-1,2,rightDim))
    tensors_split.append(U)
    return tensors_split