import unittest

import qiskit.circuit
import qiskit.providers
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import opt_einsum as oe
import numpy as np
from qtn_sim import *



class QCTest(unittest.TestCase):
    def test1XQubit(self):
        n = 4
        bond_dimension = 3
    
        q = QuantumMPS(n,bond_dimension, OneShotOptimiser(oe.contract), np.einsum)

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

        q.apply(XGate(), [2])
        q.apply(SWAPGate(),[0,1])

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

    def test2SWAPGateClass(self):
        n = 3
        bond_dimension = 2
        mps = None
        state_vector = None

        q = QuantumMPS(n,bond_dimension,tensors=mps, state_vector=state_vector)

        q.apply(XGate(), [2])
        q.apply(SWAPGate(),[0,1])

        q1 = QuantumTensor(n)

        q1.apply(XGate(), [2])
        q1.apply(SWAPGate(), [0, 1])

        o = svQiskitStyleToMine(q.get_state_vector())
        o1 = svQiskitStyleToMine(q1.get_state_vector())

        self.assertTrue(np.isclose(o.sum(), 1))
        self.assertTrue(np.isclose(o1.sum(), 1))

        # Check with qiskit
        qc = QuantumCircuit(n)
        qc.x(2)
        qc.swap(0,1)
        statevector = Statevector(qc)
        result = np.square(statevector.data)

        self.assertTrue(np.allclose(result, o))
        self.assertTrue(np.allclose(result, o1))

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
        
        q.apply(HGate(), [0])
        q.apply(CNOTGate(), [0,1])
        q.apply(CNOTGate(),[1,2])

        q1.applyCircuit(GHZCircuit(n))

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
        
        q.applyCircuit(QFTCircuit(n))

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

        qft = qiskit.circuit.library.QFTGate(n)
        circ.append(qft, range(n))

        statevector = Statevector(circ)
        result = np.square(statevector.data)

        for p in range(len(result)):
            self.assertTrue(np.isclose(result[p], o[p]))
    
    def testW(self):
        n = 10
        bond_dimension = 3
        mps = None
        state_vector = None
        #
        q = QuantumMPS(n,bond_dimension)
        #
        q.applyCircuit(WCircuitLinear(n))

        o = q.get_state_vector()    
        for i in o:
            if np.isclose(0,i):
                continue

            self.assertTrue(np.isclose(1.0/n,i))

        self.assertTrue(np.isclose(o.sum(), 1))
    
    def testDeeperCircuit(self):
        n = 4
        bond_dimension = 3
        mps = None
        state_vector = None
        # state_vector = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        q = QuantumMPS(n,bond_dimension)
        # q = QuantumCircuit(n,None)

        q.apply(HGate(),[0])
        q.apply(HGate(),[1])
        q.apply(CNOTGate(),[1,2])
        q.apply(TOFFOLIGate(),[1,2,3])
        q.apply(SWAPGate(), [2,3])
        q.apply(HGate(),[0])
        q.apply(HGate(),[1])
        q.apply(CNOTGate(),[0,1])
        q.apply(XGate(),[0])
        q.apply(XGate(),[1])
        q.apply(CNOTGate(),[1,2])
        q.apply(SWAPGate(), [0,1])
        q.apply(SWAPGate(), [1,2])

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

        c = QCircuit()

        c.addGate(HGate(), [0])
        c.addGate(HGate(),[1])
        c.addGate(CNOTGate(),[1,2])
        c.addGate(TOFFOLIGate(),[1,2,3])
        c.addGate(SWAPGate(), [2,3])
        c.addGate(HGate(),[0])
        c.addGate(HGate(),[1])
        c.addGate(CNOTGate(),[0,1])
        c.addGate(XGate(),[0])
        c.addGate(XGate(),[1])
        c.addGate(CNOTGate(),[1,2])
        c.addGate(SWAPGate(), [0,1])
        c.addGate(SWAPGate(), [1,2])

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
        state_vector = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        q = QuantumMPS(n,bond_dimension,tensors=mps, state_vector=state_vector)

        q.apply(HGate(),[0])
        q.apply(HGate(),[1])
        q.apply(CNOTGate(),[0,1])
        q.apply(XGate(),[0])
        q.apply(XGate(),[1])
        q.apply(CNOTGate(),[1,2])
        q.apply(SWAPGate(), [0,1])

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

        state_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        QTN = QuantumMPS(n, bond_dimension, tensors=None, state_vector=state_vector)
        # q = QuantumCircuit(n,None)

        circuitToApply = QCircuit()
        circuitToApply.addGate(HGate(), [0])
        circuitToApply.addGate(HGate(), [1])
        circuitToApply.addGate(HGate(), [2])
        circuitToApply.addGate(HGate(), [3])
        circuitToApply.addGate(CNOTGate(), [0,1])
        circuitToApply.addGate(TOFFOLIGate(), [1,2,3])
        circuitToApply.addGate(SWAPGate(), [0,1])
        QTN.applyCircuit(circuitToApply)
        o = svQiskitStyleToMine(QTN.get_state_vector())
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

    def testOptEinsumMPS_ghz(self):
        n = 3
        bond_dimension = 3
        mps = None
        state_vector = None
        einsumOptimiser = OneShotOptimiser(oe.contract)
        q = QuantumMPS(n,bond_dimension,einsumOptimiser=einsumOptimiser,tensors=mps, state_vector=state_vector)
        
        q.applyCircuit(GHZCircuit(n))

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
    
    def testSequentialMPS_ghz(self):
        n = 3
        bond_dimension = 3
        mps = None
        state_vector = None
        einsumOptimiser = SequentialOptimiser(swapping=False)
        baseEinsumStrategy = oe.contract
        q = QuantumMPS(n,bond_dimension,einsumOptimiser=einsumOptimiser,tensors=mps, state_vector=state_vector)
        
        q.applyCircuit(GHZCircuit(n))

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
        einsumOptimiser = SequentialOptimiser(swapping=False)
        baseEinsumStrategy = oe.contract

        q = QuantumMPS(n,bond_dimension,einsumOptimiser,baseEinsumStrategy,mps, state_vector)

        circuit = QCircuit()
        circuit.addGate(HGate(), [0])
        circuit.addGate(HGate(), [1])
        circuit.addGate(HGate(), [2])
        circuit.addGate(HGate(), [3])

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

    def testQFTWithAdjacentSwaps(self):
        n = 5
        bond_dimension = 4
        mps = None
        state_vector = None

        q = QuantumMPS(n, bond_dimension, einsumOptimiser=SequentialOptimiser(swapping=True), state_vector=state_vector)
        
        q.applyCircuit(QFTCircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        self.assertTrue(np.isclose(o.sum(), 1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)

        qft = qiskit.circuit.library.QFTGate(n)
        circ.append(qft, range(n))

        statevector = Statevector(circ)
        result = np.square(statevector.data)
        # print(o - result)
        # for p in range(len(result)):
        self.assertTrue(np.allclose(result, o))

    def testQFTWithAdjacentSwapsAndWithout(self):
        n = 7
        bond_dimension = 7
        mps = None
        state_vector = None

        q = QuantumMPS(n, bond_dimension, einsumOptimiser=SequentialOptimiser(swapping=False))
        
        q.applyCircuit(QFTCircuit(n))

        o = svQiskitStyleToMine(q.get_state_vector())

        bond_dimension = 3
        q = QuantumMPS(n,bond_dimension, einsumOptimiser=SequentialOptimiser(swapping=True))
        
        q.applyCircuit(QFTCircuit(n))

        o1 = svQiskitStyleToMine(q.get_state_vector())

        # q.plot_prob()
        # print(o.sum())
        self.assertTrue(np.isclose(o.sum(), 1, rtol=1e-1))
        self.assertTrue(np.isclose(o1.sum(), 1, rtol=1e-1))
 
        # Check with qiskit
        circ = QuantumCircuit(n)

        qft = qiskit.circuit.library.QFTGate(n)
        circ.append(qft, range(n))

        statevector = Statevector(circ)
        result = np.square(statevector.data)
        # print(o - result)
        for p in range(len(result)):
            self.assertTrue(np.square(result[p]-o[p]) < 1e-4)
            self.assertTrue(np.square(result[p]-o1[p]) < 1e-4)
    
    def testLargeRandom(self):
        
        n = 10
        bd = 100
        depth = 100
        for i in range(100):
            circuit, circuitJson = getRandomCircuit(n, depth)
            q = QuantumMPS(n, bd, einsumOptimiser=SequentialOptimiser(swapping=True))
            q.applyCircuit(circuit)
            s = q.get_state_vector().sum()
            self.assertTrue(np.isclose(s, 1, rtol=1e-1))

    def testCU(self):
        
        U = XGate()

        CU = CUnitaryTensorGate(U)
        self.assertTrue(np.allclose(CNOTGate().tensor, CU.tensor))


        U = CNOTGate()

        CU = CUnitaryTensorGate(U)
        self.assertTrue(np.allclose(TOFFOLIGate().tensor, CU.tensor))
     
if __name__ =='__main__':
    unittest.main()
