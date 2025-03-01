from qtn_sim import *
import numpy as np


n = 200
bd = 10
avgIt = 1

# f = np.log2
# f = np.sqrt
# f = lambda x:x

# fi = np.exp2

# fi = np.square

# x = [i+1 for i in range(200)]

x = [i+1 for i in range(n)]


optimiser = SequentialOptimiser(swapping=True)
data = DataObj(x, n, optimiser, np.einsum, bd)


# fig = plot("Applying QFT on the " + str(n) + " qubits",
#         "max bond dimension",
#         [(QuantumMPS, "MPS"),(QuantumTensor, "1TensorLib")],
#         getPerformanceForQubitNumber,QFTCircuit,avgIt,data)

fig = plot("Taking the qubits into GHZ state",
        "number of qubits",
        [(QuantumMPS, "MPS")],
        getPerformanceForQubitNumber,GHZCircuit,avgIt,data)

# fig = plot("Taking the qubits into QFT",
#         "number of qubits",
#         [(QuantumMPS, "MPS")],
#         getPerformanceForQubitNumber,QFTCircuit,avgIt,data, f, fi)

# fig = plot("Taking the qubits into QFT with" + optimiser.toString(),
#         "number of qubits",
#         [(QuantumMPS, "MPS")],
#         getPerformanceForQubitNumber,QFTCircuit,avgIt,data, f, fi)

# fig = plot("Taking the qubits into W state",
#         "number of qubits",
#         [(QuantumMPS, "MPS")],
#         getPerformanceForQubitNumber,WCircuitLinear,10, data)


# fig = plot("Applying QFT on the " + str(n) + " qubits",
#         "max bond dimension",
#         [(QuantumMPS, "MPS")],
#         getPerformanceForBondDimension,QFTCircuit,avgIt,data)

# fig = plot("Applying QFT on the " + " qubits", "number of qubits",
#         [#((QuantumMPS,np.einsum,np.einsum), "MPS np.einsum"),
#         #  ((QuantumMPS,OneShotOptimiser(),oe.contract), "One shot oe.contract"),
#         #  ((QuantumMPS,optEinsumRandomGreedy,oe.contract), "MPS opt_einsum RandomGreedy"),
#          ((QuantumMPS,SequentialOptimiser(),np.einsum), "MPS np.einsum + sequentialOptimiser"),
#          ((QuantumMPS,SequentialOptimiser(),oe.contract), "MPS oe.contract + sequentialOptimiser"),
#          ((QuantumMPS,SequentialOptimiser(), optEinsumRandomGreedy), "MPS oe.contract Random greedy + sequentialOptimiser")
#         ],
#         getPerformanceForDifferentEinsum,QFTCircuit,avgIt,data)

fig.show()

plt.show()
