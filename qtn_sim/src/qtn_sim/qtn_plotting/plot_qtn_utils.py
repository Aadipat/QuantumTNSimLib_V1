from qtn_sim.tn_simulators.mps_qlib import *
from qtn_sim.tn_simulators.one_tensor_qlib import *
from qtn_sim.circuits import *
import numpy as np

import tracemalloc
import time 

# let us plot the performance on time and memory of mps vs statevector
# for different circuits.

class DataObj:
    def __init__(self,x,n,eo,bs,o):
        self.x = x
        self.n = n
        self.eo = eo 
        self.bs = bs 
        self.o = o 

# print(chars)

def linearise(Y, f):
    return [f(y) for y in Y]

def getPerformanceForQubitNumber(qSimulator, circuit, avgIt, data):
    yTime = []
    yMem = []

    for xi in data.x:
        
        times = []
        mems = []
        for it in range(avgIt):
            q = qSimulator(xi, data.eo, data.bs)
            if qSimulator is QuantumMPS:
                q = qSimulator(xi, data.o, data.eo, data.bs)

            circuitApplied = circuit(xi)

            start = time.perf_counter_ns()
            tracemalloc.start()

            q.applyCircuit(circuitApplied)

            mem = tracemalloc.get_traced_memory()

            tracemalloc.stop()

            end = time.perf_counter_ns()
            timeIt = end - start

            times.append(timeIt)
            mems.append(mem[1])

        yTime.append(np.mean(times))
        yMem.append(np.mean(mems))

    return yTime, yMem

def getPerformanceForBondDimension(qSimulator, circuit, avgIt, data):
    yTime = []
    yMem = []
    
    circuitApplied = circuit(data.n)

    for xi in data.x:

        times = []
        mems = []
        for it in range(avgIt):
            q = qSimulator(data.n, xi, data.eo, data.bs)
            
            start = time.perf_counter_ns()
            tracemalloc.start()

            q.applyCircuit(circuitApplied)

            mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            end = time.perf_counter_ns()
            timeIt = end - start

            times.append(timeIt)
            mems.append(mem[1])

        yTime.append(np.mean(times))
        yMem.append(np.mean(mems))

    return yTime, yMem

def getPerformanceForDifferentEinsum(qSimulator, circuit, avgIt, data):
    yTime = []
    yMem = []
    for xi in data.x:

        times = []
        mems = []
        for it in range(avgIt):
            # if qSimulator is QuantumMPS:
            q = qSimulator[0](xi, data.o,qSimulator[1],qSimulator[2])

            circuitApplied = circuit(xi)

            start = time.perf_counter_ns()
            tracemalloc.start()

            q.applyCircuit(circuitApplied)

            mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            end = time.perf_counter_ns()
            timeIt = end - start

            times.append(timeIt)
            mems.append(mem[1])

        yTime.append(np.mean(times))
        yMem.append(np.mean(mems))

    return yTime, yMem

def plot(title, xlabel,
         simulators=None,
         getData = getPerformanceForQubitNumber,
         circuit = GHZCircuit,avgIt =1,
         data=None,f = lambda y:y, fi = lambda y:y):
    
    if simulators is None:
        simulators = [()]
    if data is None:
        data = {}
    t1 = "Execution time"
    t2 = "Memory usage"

    ylabel1 = "Time (ns)"
    ylabel2 = "Memory (bytes)"

    fig = plt.figure()
    fig.suptitle(title)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.set_title(t1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    
    ax2.set_title(t2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel2)

    for (simulator,simlabel) in simulators:
        yTime, yMem = getData(simulator,circuit,avgIt,data)

        ax1.scatter(data.x,yTime,label=simlabel)
        ax1.set_yscale('function', functions=(f, fi))

        ax2.scatter(data.x,yMem,label=simlabel)
        ax2.set_yscale('function', functions=(f, fi))

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    return fig


