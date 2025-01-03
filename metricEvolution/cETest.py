# Looking at BERT transformer's embedding semantics:


from circuitEmbedding import *

from qtn_sim import *



c1 = QCircuit()

c1.addGate(HGate(), [0])
c1.addGate(HGate(), [1])


c2 = QCircuit()

c1.addGate(HGate(), [1])


c3 = QCircuit()

c1.addGate(HGate(), [0])


embeddings = embedCircuits([c1,c2,c3], 1)


print(embeddings)