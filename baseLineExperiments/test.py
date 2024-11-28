from QTN_Simulation import *
import numpy as np

n = 2
bond_dimension = 5
mps = None
state_vector = None

# einsumOptimiser = applyCircuitSequentially
einsumOptimiser = np.einsum
baseEinsumStrategy = np.einsum
# baseEinsumStrategy = np.einsum

q1 = QuantumMPS(n,bond_dimension,einsumOptimiser,baseEinsumStrategy,mps, state_vector)

q2 = QuantumTensor(n,einsumOptimiser,baseEinsumStrategy)



# q1.apply(H(),[0])
# q1.apply(H(),[0])
# q1.apply(H(),[1])
# q1.apply(CNOT(),[0,2])
# q1.apply(TOFFOLI(),[0,2,3])    
# q1.apply(SWAP(), [0,3])

# q2.apply(H(),[0])
# q2.apply(H(),[0])
# q2.apply(H(),[1])
# q2.apply(CNOT(),[0,2])
# q2.apply(TOFFOLI(),[0,2,3])    
# q2.apply(SWAP(), [0,3])

circuit = []

circuit.append((H(),[0]))
# circuit.append((H(),[1]))
# circuit.append((H(),[2]))
# circuit.append((H(),[3]))

# circuit.append((CNOT(),[0,1]))
# circuit.append((TOFFOLI(),[1,2,3]))
# circuit.append((SWAP(), [0,2]))
# # # circuit = QFTcircuit(n)
# circuit.append((TOFFOLI(),[0,2,5]))

# circuit = QFTcircuit(n)


q1.applyCircuit(circuit)
q2.applyCircuit(circuit)


# print(einsumForCircuit(q1.getTensorStringAndLines()[0], q1.getTensorStringAndLines()[1], circuit))


print(q1.get_state_vector())
print(np.square(q2.state.ravel()))