from qtn_sim.tn_simulators.tensor_utils import *
from qtn_sim.tn_simulators.einsum_lib import *
from qtn_sim.circuits import *
import numpy as np
import opt_einsum as oe

class MyOptimiser:

    def __init__(self):
        """
        Creates a MyOptimiser object
        It defines how a circuit will be applied onto a quantum state data structure
        """
        return

    def evaluate(self, qsimulator : QSimulator, circuit : QCircuit):
        """
        Evaluates the circuit on the QSimulator (quantum state abstraction) 
        """
        return

    def toString(self):
        """
        String version for the object
        """
        return "myOptimiser"

class SequentialOptimiser(MyOptimiser):

    def __init__(self, swapping = True):
        """
        Creates a SequentialOptimiser object
        Applies gates sequentially from a circuit
        
        Boolean swapping : whether swapping will be used for non adjacent qubit gates

        """
        super().__init__()
        self.swapping = swapping
        return

    def evaluate(self, qsimulator : QSimulator, circuit : QCircuit):
        """
        Right now works with QuantumMPS only
        Evaluates the circuit on the QSimulator (quantum state abstraction) 
        
        Returns:
            result : the new mps tensors
        """
        if self.swapping is True:
            return applyCircuitSequentiallyAdjacently(qsimulator, circuit)
        return applyCircuitSequentially(qsimulator, circuit)

    def toString(self):
        """
        String version for the object
        """
        s = super().toString() + " : Sequential "
        if self.swapping is True:
            s += "with swapping"
        return s

class RearrangeOptimiser(MyOptimiser):

    def __init__(self):
        """
        Creates a RearrangeOptimiser object
        It defines how a circuit will be applied onto a quantum state data structure
        """
        super().__init__()
        return


class OneShotOptimiser(MyOptimiser):
    def __init__(self, strategy = oe.contract):
        """
        Creates a OneShotoptimiser object
        1 einsum string will be used
        """
        super().__init__()
        self.strategy = strategy
        return
    def evaluate(self, qsimulator : QSimulator, circuit : QCircuit):
        """
        Alters the tensors list to 1 tensor!
        Evaluates the circuit on the QSimulator (quantum state abstraction) 
        
        Returns:
            result : the new mps tensors
        """
        # Apply as one einsum
        tensorS, lines = qsimulator.getTensorStringAndLines()
        einsum_str = einsumForCircuit(tensorS, lines, circuit, qsimulator.einsumOptimiser)
        tensors = qsimulator.tensors + [c[0].tensor for c in circuit.gateList]
        tensor = self.strategy(einsum_str, *tensors, optimize='greedy')
        # return split_tensor_SVD(qsimulator.bond_dimension, qsimulator.n, tensor, 1, 1)
        qsimulator.tensors = [tensor]
        return qsimulator



    




def applyCircuitSequentially(qSimulator : QSimulator, circuit:QCircuit):
    for (gate, indices) in circuit.gateList:
        qSimulator.apply(gate,indices)
    return qSimulator

swapGate = SWAPGate()
def applyCircuitSequentiallyAdjacently(qSimulator : QSimulator, circuit:QCircuit):
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
    return qSimulator




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

