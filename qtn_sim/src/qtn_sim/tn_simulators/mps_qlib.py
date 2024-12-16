from qtn_sim.optimisers.optimiser_lib import *
from .einsum_lib import *
from qtn_sim.tn_simulators.tensor_utils import *


# This lib allows users to create MPS states,
# apply gates on certain qubits and
# plot probabilities using numpy einstein summation and MPS networks.

class QuantumMPS:

    # MPS Constructor
    def __init__(self, n,
                 bond_dimension=3,
                 einsumOptimiser:MyOptimiser =SequentialOptimiser(),
                 baseEinsumStrategy=np.einsum,
                 tensors=None, state_vector=None):
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
            for _ in range(int(np.log2(len(state_vector)))):
                l.append(2)
            l.append(1)
            oneTensor = np.reshape(state_vector, tuple(l))
            self.tensors = split_tensor_SVD(self.bond_dimension, self.n, oneTensor)
            return
        self.tensors = [np.reshape(np.array([1, 0]), (1, 2, 1)) for _ in range(n)]
        return

    # Get state tensor string and lines for circuit
    def getTensorStringAndLines(self):
        s = ""
        lines = []
        for t in range(len(self.tensors)):
            tensorS = einsumForTensor(self.tensors[t], t * 3, self.einsumOptimiser)
            s += tensorS
            if t < len(self.tensors) - 1:
                s += ","
            lines.append(tensorS[1])
        return s, lines

    # Convert to 1 tensor representation.
    def convert_To_One_Tensor(self):
        # We contract on all the intermediate indices
        tensor = self.contract_tensors([t for t in range(len(self.tensors))])
        return tensor

    # Contract adjacent mps
    def contract_tensors(self, indices=None):
        # If the mps nodes are not entangled => vectors. 
        # We must take the outer product to form the tensor.
        if indices is None:
            indices = []
        if len(indices) < 2:
            return self.tensors[indices[0]]

        # We contract on specific the intermediate indices
        tensor = self.tensors[indices[0]]
        k = 1
        while k < len(indices):
            # Need to build ein_sum string.
            s = ""
            o = ""
            for ti in range(len(tensor.shape)):
                s += getChars(self.baseEinsumStrategy)[ti]  # chr(ord('a') + i)
                if ti is not len(tensor.shape) - 1:
                    o += getChars(self.baseEinsumStrategy)[ti]
            s += ","
            for ti in range(len(self.tensors[indices[k]].shape)):
                s += getChars(self.baseEinsumStrategy)[
                    ti + len(tensor.shape) - 1]  #chr(ord('a') + i + len(tensor.shape) - 1)
                if ti != 0:
                    o += getChars(self.baseEinsumStrategy)[ti + len(tensor.shape) - 1]
            s += "->" + o
            tensor = self.baseEinsumStrategy(s, tensor, self.tensors[indices[k]], optimize='greedy')
            k += 1
        return tensor

    # Apply a certain circuit list(gate,indices)
    def applyCircuit(self, circuit : QCircuit):
        newTensors = self.einsumOptimiser.evaluate(self, circuit)
        self.tensors = newTensors

    # Apply gate on certain qubits
    def apply(self, gate, qubits=None):
        # To apply a gate, we must contract tensors with the gate 
        # We contract each tensor with the respective index of the gate on the tensor
        # then we use SVD to split the tensor.
        if qubits is None:
            qubits = []
        if isinstance(gate, Gate):
            gate = gate.tensor

        # We contract the adjacent tensors
        # Then contract with the gate
        contraction_list = [_ for _ in range(qubits[0], qubits[-1] + 1)]
        split_num = len(contraction_list)
        tensor = self.contract_tensors(contraction_list)
        qubitsShifted = [q - qubits[0] for q in qubits]
        new_Tensor = applyGateTensorOnTensor(tensor, gate, qubitsShifted, self.baseEinsumStrategy)

        self.tensors[qubits[0]] = new_Tensor
        # Then use svd to split the tensors again.
        left_B_dim = 1
        if qubits[0] != 0:
            left_B_dim = self.tensors[qubits[0] - 1].shape[-1]
        right_B_dim = 1
        if qubits[-1] != self.n - 1:
            right_B_dim = self.tensors[qubits[-1] + 1].shape[0]
        if split_num > 1:
            tensors_split = split_tensor_SVD(self.bond_dimension, split_num, new_Tensor, left_B_dim, right_B_dim)
            tensors = []
            ti = 0
            j = qubits[0]
            m = j
            k = qubits[-1] + 1
            while ti < j:
                tensors.append(self.tensors[ti])
                ti += 1
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

        prob_dist = self.get_state_vector()

        plot_prob_dist(prob_dist)