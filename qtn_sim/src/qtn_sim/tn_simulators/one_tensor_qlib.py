from typing import List
from .einsum_lib import *
import opt_einsum as oe
from qtn_sim.circuits import *
from qtn_sim.tn_simulators.tensor_utils import *

# This lib allows users to simulate a statevector like 1 tensor approach, 
# apply gates on certain qubits and 

class QuantumTensor:

    def __init__(self, n: int = 1, einsumOptimiser = oe.contract, baseEinsumStrategy = oe.contract, init_arr: List[float] = None):
        self.einsumOptimiser = einsumOptimiser
        self.baseEinsumStrategy = baseEinsumStrategy
        # Start with a simple 2 x 2^(n-1) matrix that represents the initial state
        zero_qbit = np.array([1, 0])
        
        state_list = [zero_qbit]*n
     
        if init_arr is not None:
            state_list = [np.array([1-x, x]) for x in init_arr]

        state = state_list[0]

        for qbit in state_list[1:]:
            state = np.tensordot(state, qbit, axes=-1)
        shape = [1]
        for qi in range(n):
            shape.append(2)
        shape.append(1)
        state = np.reshape(state, tuple(shape))
        self.state = state

    def get_state_vector(self):
        return np.ravel(self.state)

    def apply(self, gate, qubits=None):
        if qubits is None:
            qubits = []
        if isinstance(gate, Gate):
            gate = gate.tensor
        self.state = applyGateTensorOnTensor(self.state, gate, qubits, self.baseEinsumStrategy)

    def applyCircuit(self,circuit:QCircuit):
        for (gate, indices) in circuit.gateList:
            self.apply(gate,indices)
     
    def plot_prob(self):
        
        prob_dist = abs(np.square(self.state))

        plot_prob_dist(prob_dist)

