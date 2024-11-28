import unittest
import qiskit.circuit
import qiskit.providers
from QTN_Simulation.TN_Simulators import *
from QTN_Simulation.QTN_circuits import *
from QTN_Simulation.QTN_Optimisers import *
from QTN_Simulation.TN_Simulators.tensorUtils import *

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import opt_einsum as op

class QCTest(unittest.TestCase):
    def test1XQubit(self):
        n = 4
        bond_dimension = 3
        mps = None
        state_vector = None
        state_vector = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
        q = QuantumMPS(n,bond_dimension)

        q1 = QuantumTensor(n)

        q1.apply(X(),[0])

        q.apply(X(),[0])
        
        o = svQiskitStyleToMine(q.get_state_vector())
        o1 = svQiskitStyleToMine(np.square(q1.state.ravel()))

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
        self.assertTrue(np.isclose(o1.sum(), 1))

        # Check with qiskit
        qc = QuantumCircuit(n)
        qc.x(0) 
        statevector = Statevector(qc)
        result = np.square(statevector.data)

        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
            self.assertTrue(np.isclose(result[p], o1[p]))

    def test2SWAPQubit(self):
        n = 3
        bond_dimension = 2
        mps = None
        state_vector = None

        q = QuantumMPS(n,bond_dimension,tensors=mps, state_vector=state_vector)

        q.apply(X(), [2])
        q.apply(SWAP(),[0,1])

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))

        # Check with qiskit
        qc = QuantumCircuit(n)
        qc.x(2)
        qc.swap(0,1) 
        statevector = Statevector(qc)
        result = np.square(statevector.data)

        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))

    def test3TOFFOLIQubits(self):
        n = 3
        bond_dimension = 1
        mps = None
        state_vector = None

        q = QuantumMPS(n,bond_dimension)
        
        q.apply(X(), [0])
        q.apply(X(), [1])
        q.apply(TOFFOLI(),[0,1,2])

        o = svQiskitStyleToMine(q.get_state_vector())
        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))

        # Check with qiskit
        circ = QuantumCircuit(n)
        circ.x(0)
        circ.x(1)
        circ.ccx(0, 1, 2)
        statevector = Statevector(circ)
        result = np.square(statevector.data)

        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))

    def testGHZ(self):
        n = 3
        bond_dimension = 3
        mps = None
        state_vector = None

        q = QuantumMPS(n,bond_dimension)
        q1 = QuantumTensor(n)
        
        q.apply(H(), [0])
        q.apply(CNOT(), [0,1])
        q.apply(CNOT(),[1,2])

        q1.applyCircuit(GHZcircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())
        o1 = svQiskitStyleToMine(np.square(q1.state.ravel()))
        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
        self.assertTrue(np.isclose(o1.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        circ.h(0)
        circ.cx(0,1)
        circ.cx(1, 2)

        statevector = Statevector(circ)
        result = np.square(statevector.data)
        
        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
            self.assertTrue(np.isclose(result[p], o1[p]))

    def testQFT(self):
        n = 3
        bond_dimension = 3
        mps = None
        state_vector = None

        q = QuantumMPS(n,bond_dimension)
        
        q.applyCircuit(QFTcircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        # for i in range(n):
        #     circ.h(i)
        # for j in range(i + 1, n):
        #     angle = np.pi / (2 ** (j - i))
        #     circ.cp(angle, i,j)
            
        # for i in range(n // 2):
        #     circ.swap(i, n - i - 1)

        qft = qiskit.circuit.library.QFT(n)
        circ.append(qft, range(n))

        statevector = Statevector(circ)
        result = np.square(statevector.data)

        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
    
    def testW(self):
        n = 3
        bond_dimension = 3
        mps = None
        state_vector = None

        q = QuantumMPS(n,bond_dimension)
        
        q.applyCircuit(Wcircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        qc = QuantumCircuit(n)
    
        # Apply initial rotation to the first qubit
        theta = 2 * np.arccos(1/np.sqrt(n))
        qc.ry(theta, 0)
        
        # Apply controlled rotations and CNOT gates
        for i in range(1, n):
            theta_i = 2 * np.arccos(1/np.sqrt(n-i))
            
            # Controlled rotation
            qc.cry(theta_i, i-1, i)
            
            # CNOT gates to propagate the state
            for j in range(i-1, -1, -1):
                qc.cx(j, i)
                
        statevector = Statevector(qc)
        result = np.square(statevector.data)
        
        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
        return
    
    def testDeeperCircuit(self):
        n = 4
        bond_dimension = 3
        mps = None
        state_vector = None
        # state_vector = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        q = QuantumMPS(n,bond_dimension)
        # q = QuantumCircuit(n,None)

        q.apply(H(),[0])
        q.apply(H(),[1])
        q.apply(CNOT(),[1,2])
        q.apply(TOFFOLI(),[1,2,3])
        q.apply(SWAP(), [2,3])
        q.apply(H(),[0])
        q.apply(H(),[1])
        q.apply(CNOT(),[0,1])
        q.apply(X(),[0])
        q.apply(X(),[1])
        q.apply(CNOT(),[1,2])   
        q.apply(SWAP(), [0,1])
        q.apply(SWAP(), [1,2])

        # o = q.get_state_vector()
        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        circ.h(0)
        circ.h(1)
        circ.cx(1,2)
        circ.ccx(1,2,3)
        circ.swap(2,3)
        circ.h(0)
        circ.h(1)
        circ.cx(0,1)
        circ.x(0)
        circ.x(1)
        circ.cx(1,2)
        circ.swap(0,1)
        circ.swap(1,2)
        
        statevector = Statevector(circ)
        result = np.square(statevector.data)
        
        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))

    def testDeeperWholeCircuit(self):
        n = 4
        bond_dimension = 3
        mps = None
        state_vector = None
        # state_vector = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        q = QuantumMPS(n,bond_dimension)
        # q = QuantumCircuit(n,None)

        c = []
        c.append((H(),[0]))
        c.append((H(),[1]))
        c.append((CNOT(),[1,2]))
        c.append((TOFFOLI(),[1,2,3]))
        c.append((SWAP(), [2,3]))
        c.append((H(),[0]))
        c.append((H(),[1]))
        c.append((CNOT(),[0,1]))
        c.append((X(),[0]))
        c.append((X(),[1]))
        c.append((CNOT(),[1,2]))   
        c.append((SWAP(), [0,1]))
        c.append((SWAP(), [1,2]))

        q.applyCircuit(c)

        # o = q.get_state_vector()
        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        circ.h(0)
        circ.h(1)
        circ.cx(1,2)
        circ.ccx(1,2,3)
        circ.swap(2,3)
        circ.h(0)
        circ.h(1)
        circ.cx(0,1)
        circ.x(0)
        circ.x(1)
        circ.cx(1,2)
        circ.swap(0,1)
        circ.swap(1,2)
        
        statevector = Statevector(circ)
        result = np.square(statevector.data)
        
        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
        
    def testSwapCnotHCircuit(self):
        n = 4
        bond_dimension = 3
        mps = None
        state_vector = None
        state_vector = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        q = QuantumMPS(n,bond_dimension,tensors=mps, state_vector=state_vector)

        q.apply(H(),[0])
        q.apply(H(),[1])
        q.apply(CNOT(),[0,1])
        q.apply(X(),[0])
        q.apply(X(),[1])
        q.apply(CNOT(),[1,2])   
        q.apply(SWAP(), [0,1])

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))

        circ = QuantumCircuit(n)
        circ.h(0)
        circ.h(1)
        circ.cx(0,1)
        circ.x(0)
        circ.x(1)
        circ.cx(1,2)
        circ.swap(0,1)

        statevector = Statevector(circ)
        result = np.square(statevector.data)

        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))

    def testToffoliCnotH(self):
        n = 4
        bond_dimension = 3
        mps = None
        state_vector = None
        # state_vector = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        q = QuantumMPS(n,bond_dimension,tensors=mps, state_vector=state_vector)
        # q = QuantumCircuit(n,None)

        circuit = []

        circuit.append((H(),[0]))
        circuit.append((H(),[1]))
        circuit.append((H(),[2]))
        circuit.append((H(),[3]))

        circuit.append((CNOT(),[0,1]))
        circuit.append((TOFFOLI(),[1,2,3]))
        circuit.append((SWAP(), [1,2]))


        q.applyCircuit(circuit)



        # o = q.get_state_vector()
        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        
        circ.h(0)
        circ.h(1)
        circ.h(2)
        circ.h(3)
        
        circ.cx(0,1)
        circ.ccx(1,2,3)
        circ.swap(1,2)
        
        statevector = Statevector(circ)
        result = np.square(statevector.data)
        
        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))

    def testOptEinsumMPSghz(self):
        n = 3
        bond_dimension = 3
        mps = None
        state_vector = None
        einsumOptimiser = op.contract
        q = QuantumMPS(n,bond_dimension,einsumOptimiser=einsumOptimiser,tensors=mps, state_vector=state_vector)
        
        q.applyCircuit(GHZcircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        circ.h(0)
        circ.cx(0,1)
        circ.cx(1, 2)

        statevector = Statevector(circ)
        result = np.square(statevector.data)
        
        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
        return
    
    def testSequentialMPSghz(self):
        n = 3
        bond_dimension = 3
        mps = None
        state_vector = None
        einsumOptimiser = applyCircuitSequentially
        baseEinsumStrategy = op.contract
        q = QuantumMPS(n,bond_dimension,einsumOptimiser=einsumOptimiser,tensors=mps, state_vector=state_vector)
        
        q.applyCircuit(GHZcircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        circ.h(0)
        circ.cx(0,1)
        circ.cx(1, 2)

        statevector = Statevector(circ)
        result = np.square(statevector.data)
        
        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
        return
    
    def testNLargeHs(self):
        n = 10
        bond_dimension = 3
        mps = None
        state_vector = None
        einsumOptimiser = applyCircuitSequentially
        baseEinsumStrategy = op.contract

        q = QuantumMPS(n,bond_dimension,einsumOptimiser,baseEinsumStrategy,mps, state_vector)

        circuit = []

        circuit.append((H(),[0]))
        circuit.append((H(),[1]))
        circuit.append((H(),[2]))
        circuit.append((H(),[3]))

        q.applyCircuit(circuit)

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        circ.h(0)
        circ.h(1)
        circ.h(2)
        circ.h(3)

        statevector = Statevector(circ)
        result = np.square(statevector.data)
        
        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
        return

    def testQFTWithadjacentSwaps(self):
        n = 5
        bond_dimension = 4
        mps = None
        state_vector = None

        q = QuantumMPS(n, bond_dimension, einsumOptimiser=applyCircuitSequentiallyAdjacently)
        
        q.applyCircuit(QFTcircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)
        # for i in range(n):
        #     circ.h(i)
        # for j in range(i + 1, n):
        #     angle = np.pi / (2 ** (j - i))
        #     circ.cp(angle, i,j)
            
        # for i in range(n // 2):
        #     circ.swap(i, n - i - 1)

        qft = qiskit.circuit.library.QFT(n)
        circ.append(qft, range(n))

        statevector = Statevector(circ)
        result = np.square(statevector.data)
        # print(o - result)
        # for p in range(len(result)):
        self.assertTrue(np.allclose(result, o))

    def testQFTWithadjacentSwapsAndWithout(self):
        n = 7
        bond_dimension = 5
        mps = None
        state_vector = None

        q = QuantumMPS(n, bond_dimension, einsumOptimiser=applyCircuitSequentiallyAdjacently)
        
        q.applyCircuit(QFTcircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())

        bond_dimension = 3
        q = QuantumMPS(n,bond_dimension, einsumOptimiser=applyCircuitSequentially)
        
        q.applyCircuit(QFTcircuit(n))

        o1 = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        # print(o.sum())
        self.assertTrue(np.isclose(o.sum(), 1, rtol=1e-1))
        self.assertTrue(np.isclose(o1.sum(), 1, rtol=1e-1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)

        qft = qiskit.circuit.library.QFT(n)
        circ.append(qft, range(n))

        statevector = Statevector(circ)
        result = np.square(statevector.data)
        # print(o - result)
        for p in range(len(result)):
            self.assertTrue(np.square(result[p]-o[p]) < 1e-4)
            self.assertTrue(np.isclose(result[p], o1[p], rtol=1e-2))
    
if __name__ =='__main__':
    unittest.main()
