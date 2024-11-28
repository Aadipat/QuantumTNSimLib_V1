from QTN_Simulation.QTN_circuits.circuitLib import *
import opt_einsum as oe
import numpy as np

def optEinsumRandomGreedy(einsum_string, *tensors, optimize = ""):
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
    opt_rg = oe.RandomGreedy(max_repeats=256, parallel=False)
    sizes_dict = getSizesForIndices(einsum_string, tensors)
    views = buildViews(einsum_string, sizes_dict)
    path, path_info = oe.contract_path(einsum_string, *tensors, optimize=opt_rg)
    return oe.contract(einsum_string, *tensors, optimize=path)


class myOptimiser:

    def __init__(self):
        return

class sequentialOptimiser(myOptimiser):

    def __init__(self):
        return
    

class rearrangeOptimiser(myOptimiser):

    def __init__(self):
        return
    




def applyCircuitSequentially(qSimulator, circuit):
    for (gate, indices) in circuit:
        qSimulator.apply(gate,indices)
    return qSimulator.tensors

swapGate = SWAP()
def applyCircuitSequentiallyAdjacently(qSimulator, circuit):
    # return applyCircuitSequentially(qSimulator, circuit)
    for (gate, indices) in circuit:
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
            for indices in swaps:
                qSimulator.apply(swapGate,indices)
    return qSimulator.tensors