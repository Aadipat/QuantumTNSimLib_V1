from qtn_sim.tn_simulators.tensor_utils import *
from qtn_sim.tn_simulators.einsum_lib import *
from qtn_sim.circuits import *
import numpy as np
import opt_einsum as oe

class MyOptimiser:

    def __init__(self):
        return

    def evaluate(self, qsimulator, circuit):
        return

    def toString(self):
        return "myOptimiser";

class SequentialOptimiser(MyOptimiser):

    def __init__(self, swapping = True):
        super().__init__()
        self.swapping = swapping
        return

    def evaluate(self, qsimulator, circuit):
        if self.swapping is True:
            return applyCircuitSequentiallyAdjacently(qsimulator, circuit)
        return applyCircuitSequentially(qsimulator, circuit)

    def toString(self):
        s = super().toString() + " : Sequential "
        if self.swapping is True:
            s += "with swapping"
        return s

class RearrangeOptimiser(MyOptimiser):

    def __init__(self):
        super().__init__()
        return


class OneShotOptimiser(MyOptimiser):
    def __init__(self, strategy = oe.contract):
        super().__init__()
        self.strategy = strategy
        return
    def evaluate(self, qsimulator, circuit : QCircuit):
        # Apply as one einsum
        tensorS, lines = qsimulator.getTensorStringAndLines()
        einsum_str = einsumForCircuit(tensorS, lines, circuit, qsimulator.einsumOptimiser)
        tensors = qsimulator.tensors + [c[0].tensor for c in circuit.gateList]
        tensor = self.strategy(einsum_str, *tensors, optimize='greedy')
        return split_tensor_SVD(qsimulator.bond_dimension, qsimulator.n, tensor, 1, 1)



    




def applyCircuitSequentially(qSimulator, circuit:QCircuit):
    for (gate, indices) in circuit.gateList:
        qSimulator.apply(gate,indices)
    return qSimulator.tensors

swapGate = SWAPGate()
def applyCircuitSequentiallyAdjacently(qSimulator, circuit:QCircuit):
    # return applyCircuitSequentially(qSimulator, circuit)
    for (gate, indices) in circuit.gateList:
        # print(indices)
        nonadjacent =  np.max([0]+[indices[i] -indices[i-1] for i in range(len(indices))]) > 1
        swaps = []
        if nonadjacent:
            # Qubits are non adjacent, will swap. O(n) overhead.
            q = indices[0]
            for i in range(1,len(indices)):
                # print(q)
                if indices[i] - q > 1:
                    for j in range(indices[i] - q-1):
                        # print([indices[i]-j-1,indices[i]-j])
                        qSimulator.apply(swapGate, [indices[i]-j-1,indices[i]-j])
                        swaps.append([indices[i]-j-1,indices[i]-j])
                q += 1
            # qSimulator.applyCircuit(swaps)
        # print([s[1] for s in swaps])
        qSimulator.apply(gate,[i + indices[0] for i in range(len(indices))])
        swaps.reverse()
        if nonadjacent and len(swaps) > 0:
            for i in swaps:
                qSimulator.apply(swapGate,i)
    return qSimulator.tensors




def getSizesForIndices(einsum_string, tensors):
    dict = {}
    t = 0
    strings = einsum_string.split("->")[0].split(",")
    # print(len(strings))
    # print(len(tensors))
    for tensor in tensors:
        i = 0
        for i in range(len(tensor.shape)):
            dict[strings[t][i]] = tensor.shape[i]
            i += 1
        t += 1
    return dict

def buildViews(einsum_string, sizes_dict):
    views = []
    for term in einsum_string.split("->")[0].split(","):
        if term == '->':
            break
        shape = tuple(sizes_dict[idx] for idx in term)
        views.append(np.random.rand(*shape))
    return views

def optEinsumRandomGreedy(einsum_string, *tensors, optimize = ""):
    opt_rg = oe.RandomGreedy(max_repeats=256, parallel=False)
    sizes_dict = getSizesForIndices(einsum_string, tensors)
    views = buildViews(einsum_string, sizes_dict)
    path, path_info = oe.contract_path(einsum_string, *tensors, optimize=opt_rg)
    return oe.contract(einsum_string, *tensors, optimize=path)

