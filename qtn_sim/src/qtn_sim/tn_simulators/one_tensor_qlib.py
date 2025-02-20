from typing import List
from .einsum_lib import *
import opt_einsum as oe
from qtn_sim.circuits import *
from qtn_sim.tn_simulators.tensor_utils import *
from qtn_sim.optimisers.optimiser_lib import *

# This lib allows users to simulate a statevector like 1 tensor approach, 
# apply gates on certain qubits and 

class QuantumTensor(QSimulator):

    def __init__(self, n: int = 1, 
                 einsumOptimiser : MyOptimiser = SequentialOptimiser(), 
                 baseEinsumStrategy = np.einsum, 
                 init_arr : np.array = None):
        """
        Creates a quantum 1 tensor object with n qubit lines
        The einsumOptimiser: is how the entire 
        einstein summation of the circuit
        is going to be computed. We pass a MyOptimiser object. Default it is SequentialOptimiser
        which applies gates sequentially and takes each gate einsum seperately. 
        The baseEinsum strategy is how the individual einstein summations are computed
        
        This takes in a function that can be called 
        : baseEinsumStrategy(es_str, state_tensor, gate_tensor)
        where these tensors are numpy arrays

        init_arr : any input state

        Args:

        Returns:
            QuantumTensor : 1 tensor / statevector simulator object
        """
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
        self.tensors = [state]
        self.n = n
    
    # Get state tensor string and lines for circuit
    def getTensorStringAndLines(self):
        """
        This method gives the tensor string for the mps and the lines list
        May be used for one shot simulation

        Returns : the einsum string for the state and the lines
        """
        s = ""
        lines = []
        tensorS = einsumForTensor(self.tensors[0], 0, self.baseEinsumStrategy)
        s += tensorS
        for i in range(self.n):
            lines.append(tensorS[i+1])
        return s, lines

    def get_state_vector(self):
        """
        Gives back a vetcor of probabilities of each state
        If you want amplitudes, you must run np.ravel(_.convert_To_One_Tensor())

        It does not alter the state
        
        Returns : the state vector as a numpy array of probabilities, not amplitudes.
        """
        return np.ravel(np.square(self.state))

    def apply(self, gate : Gate, qubits : list[int] = None):
        """
        Applies a given Gate object 
        to the QuantumMPS object
        
        Returns : void method. The state is altered
        """
        if qubits is None:
            qubits = []
        if isinstance(gate, Gate):
            gate = gate.tensor
        self.state = applyGateTensorOnTensor(self.state, gate, qubits, self.baseEinsumStrategy)
        self.tensors = [self.state]

    def applyCircuit(self, circuit : QCircuit):
        """
        Applies a given QCircuit object (a quantum circuit)
        to the QuantumMPS object

        Returns : void method. The state is altered.
        """
        # for (gate, indices) in circuit.gateList:
        #     self.apply(gate,indices)
        self = self.einsumOptimiser.evaluate(self, circuit)
        self.state = self.tensors[0]
     
    def plot_prob(self):
        """
        Uses matplotlib to plot a histogram of state probabilities. 
        It does not alter the state tensor
        """
        prob_dist = abs(np.square(self.state))

        plot_prob_dist(prob_dist)

