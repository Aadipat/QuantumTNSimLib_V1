from typing import List
# from circuitLib import *
from .einsumLib import *
import numpy as np
import matplotlib.pyplot as plt
import opt_einsum as oe

# This lib allows users to simulate a statevector like 1 tensor approach, 
# apply gates on certain qubits and 

class QuantumTensor():

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
        for i in range(n):
            shape.append(2)
        shape.append(1)
        state = np.reshape(state, tuple(shape))
        self.state = state

    def apply(self, gate = np.array([]), qubits=[], es_str: str = None):
        self.state = apply_Gate_On_Tensor(self.state,gate,qubits, self.baseEinsumStrategy)

    def applyCircuit(self,circuit:List[tuple]):
        for (gate, indices) in circuit:
            self.apply(gate,indices)
     
    def plot_prob(self):
        
        prob_distr = abs(np.square(self.state))

        x_lbls = []

        num = len(prob_distr.shape)
        
        i = 0
        probs = np.ravel(prob_distr)
        for elem in probs:
            s = "{:0"+str(num)+"b}"
            x_lbls.append(f"|" + s.format(i) + ">")
            i +=1 

        plt.bar(x_lbls, probs)
        plt.show()
