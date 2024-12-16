from qtn_sim import *
import numpy as np

n = 4
bond_dimension = 2
mps = None
state_vector = None

einsumOptimiser = SequentialOptimiser(swapping=True)
baseEinsumStrategy = np.einsum
q1 = QuantumMPS(n,bond_dimension,einsumOptimiser,baseEinsumStrategy,mps, state_vector)

q2 = QuantumTensor(n,einsumOptimiser,baseEinsumStrategy)

circuit = GHZCircuit(n)

print(circuit.toString())


q1.applyCircuit(circuit)
q2.applyCircuit(circuit)

print(q1.get_state_vector())
print(np.square(q2.state.ravel()))