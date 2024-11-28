from QTN_Simulation.QTN_plotting import *
from QTN_Simulation.TN_Simulators import *
from QTN_Simulation.QTN_circuits import *
from QTN_Simulation.QTN_Optimisers import *

n = 10
bd = 2
avgIt = 10

f = np.log2
# f = np.sqrt
# f = lambda x:x

# x = [i+1 for i in range(200)]

x = [i+1 for i in range(20)]

data = DataObj(x, n, applyCircuitSequentiallyAdjacently, np.einsum, bd)


# fig = plot("Applying QFT on the " + str(n) + " qubits",
#         "max bond dimension",
#         [(QuantumMPS, "MPS"),(QuantumTensor, "1TensorLib")],
#         getPerformanceForQubitNumber,QFTcircuit,avgIt,data)

# fig = plot("Taking the qubits into GHZ state",
#         "number of qubits",
#         [(QuantumMPS, "MPS")],
#         getPerformanceForQubitNumber,GHZcircuit,avgIt,data)

# fig = plot("Taking the qubits into QFT",  
#         "number of qubits",
#         [(QuantumMPS, "MPS"), (QuantumTensor,"1Tensor")], 
#         getPerformanceForQubitNumber,QFTcircuit,avgIt,data)

fig = plot("Taking the qubits into QFT" + str(applyCircuitSequentiallyAdjacently),
        "number of qubits",
        [(QuantumMPS, "MPS")],
        getPerformanceForQubitNumber,QFTcircuit,avgIt,data,f)

# fig = plot("Taking the qubits into GHZ",  
#         "number of qubits",
#         [(QuantumMPS, "MPS")], 
#         getPerformanceForQubitNumber,GHZcircuit,avgIt,data)

# fig = plot("Taking the qubits into W",  
#         "number of qubits",
#         [(QuantumMPS, "MPS")], 
#         getPerformanceForQubitNumber,Wcircuit,avgIt,data,f)

# fig = plot("Applying QFT on the " + str(n) + " qubits", 
#         "max bond dimension",
#         [(QuantumMPS, "MPS")],
#         getPerformanceForBondDimension,QFTcircuit,avgIt,data)

# fig = plot("Applying GHZ on the " + str(n) + " qubits", 
#         "max bond dimension",
#         [(QuantumMPS, "MPS")],
#         getPerformanceForBondDimension,GHZcircuit,avgIt,data)

# fig = plot("Applying W State on the " + str(n) + " qubits",
#         "max bond dimension",
#         [(QuantumMPS, "MPS")],
#         getPerformanceForBondDimension,Wcircuit,avgIt,data)

# fig = plot("Applying QFT on the " + " qubits", "number of qubits",
#         [#((QuantumMPS,np.einsum,np.einsum), "MPS np.einsum"),
#          #((QuantumMPS,oe.contract,oe.contract), "MPS oe.contract"),
#          #((QuantumMPS,applyCircuitSequentiallyAdjacently,np.einsum), "MPS np.einsum + sequentialOptimiser"),
#          ((QuantumMPS,applyCircuitSequentiallyAdjacently,oe.contract), "MPS oe.contract + sequentialOptimiser"),
#          ((QuantumMPS,applyCircuitSequentiallyAdjacently,optEinsumRandomGreedy), "MPS oe.contract Random greedy + sequentialOptimiser")],
#         getPerformanceForDifferentEinsum,QFTcircuit,avgIt,data)

# fig = plot("Applying GHZ on the " + " qubits", "number of qubits",
#         [#((QuantumMPS,np.einsum,np.einsum), "MPS np.einsum"),
#          ((QuantumMPS,oe.contract,oe.contract), "MPS oe.contract"),
#         #  ((QuantumMPS,optEinsumRandomGreedy,oe.contract), "MPS opt_einsum RandomGreedy"),
#          ((QuantumMPS,applyCircuitSequentiallyAdjacently,np.einsum), "MPS np.einsum + sequentialOptimiser"),
#          ((QuantumMPS,applyCircuitSequentiallyAdjacently,oe.contract), "MPS oe.contract + sequentialOptimiser"),
#          ((QuantumMPS,applyCircuitSequentiallyAdjacently,optEinsumRandomGreedy), "MPS oe.contract Random greedy + sequentialOptimiser")
#         ],
#         getPerformanceForDifferentEinsum,GHZcircuit,avgIt,data)


# fig = plot("Applying GHZ on the " + " qubits", "number of qubits",
#         [((QuantumMPS,applyCircuitSequentiallyAdjacently,np.einsum), "MPS np.einsum + sequentialOptimiser"),
#          ((QuantumMPS,applyCircuitSequentiallyAdjacently,oe.contract), "MPS oe.contract + sequentialOptimiser")
#          ],
#         getPerformanceForDifferentEinsum,GHZcircuit,avgIt,data)


fig.show()

plt.show()


# print(q1.get_state_vector())
# print(np.square(q2.state.ravel()))