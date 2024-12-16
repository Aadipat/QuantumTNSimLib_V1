import tracemalloc
import opt_einsum as oe
from opt_einsum import contract
import time
from qtn_sim import  *
import matplotlib.pyplot as plt

# for 4 lined ghzcircuit Quantum domain
print("Quantum tensor")
n = 2
circuit = GHZCircuit(n)
q = QuantumMPS(n)
s,l = q.getTensorStringAndLines()

es = einsumForCircuit(s,l,circuit, None)
print(es)
tensors = q.tensors + [c[0].tensor for c in circuit.gateList]
 
# tensors = [np.random.rand(*t.shape) for t in q.tensors]

t = time.perf_counter_ns()
np.einsum(es, *tensors)
t1 = time.perf_counter_ns() - t

t = time.perf_counter_ns()
contract(es, *tensors)
t2 = time.perf_counter_ns() - t

print(t1)
print(t2)

x = []
y1 = []
y2 = []
y3 = []
y4 = []

bd = 1
r = 15
print("Random tensor") 
for n in range(r):
    N = n+1

    tensors = [np.random.rand(bd,N,bd) for t in q.tensors] + [np.random.rand(*[N for i in c[0].tensor.shape]) for c in circuit.gateList]

    opt_rg = oe.RandomGreedy(max_repeats=256, parallel=False)
    sizes_dict = getSizesForIndices(es, tensors)
    views = buildViews(es, sizes_dict)
    path, path_info = oe.contract_path(es, *tensors)

    t = time.perf_counter_ns()
    tracemalloc.start()
    np.einsum(es, *tensors)
    mem1 = tracemalloc.get_traced_memory()
    t1 = time.perf_counter_ns() - t

    t = time.perf_counter_ns()
    tracemalloc.start()
    contract(es, *tensors, optimize=path)
    mem2 = tracemalloc.get_traced_memory()
    t2 = time.perf_counter_ns() - t
    
    x.append(N)
    y1.append(t1)
    y2.append(t2)
    y3.append(mem1[1])
    y4.append(mem2[1])


print("np:" + str(t1))
print("oe:" + str(t2))


plt.suptitle("Plotting performance of np vs oe for random tensors analogous to GHZ format of type (bd,N,bd)")

plt.subplot(1,2,1)
plt.title("Time")
plt.scatter(x,y1, label="np")
plt.scatter(x,y2, label="oe")
plt.axvline(x = 2, color = 'b', label = 'N=2 for our tensors in MPS')
plt.xlabel("middle component of a tensor: (bd,N,bd)")
plt.ylabel("Time (ns)")

plt.subplot(1,2,2)
plt.title("Memory")
plt.scatter(x,y3, label="np")
plt.scatter(x,y4, label="oe")
plt.axvline(x = 2, color = 'b', label = 'N=2 for our tensors in MPS')
plt.xlabel("middle component of a tensor: (bd,N,bd)")
plt.ylabel("Memory (bytes)")

plt.legend()
plt.show()

