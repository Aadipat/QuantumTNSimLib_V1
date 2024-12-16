
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)


def embed(input):
    return model(input)



# We will use in circuit gate variance (basically the variance on how many qubits the gates are acting.)
# We can use the gate ids t weight the variance : Corresponds to chaos/ complexity in circuit.

def embedCircuit(circuit):
    # embeddedCircuit = []

    # gateIndices = []
    # for (gate, indices) in circuit.circuit:
    #     gateIndices.append(np.var(indices))
    #
    # embeddedCircuit.append(np.sum(gateIndices))
    # return embeddedCircuit

    # return len(circuit.gateList)

    return hash(json.dumps(circuit.toJSONDict()))

def embedCircuits(circuits, components):
    sentences = []
    for circuit in circuits:
        s = circuit.toString()
        sentences.append(s)
    embeddings = embed(sentences)
    print(embeddings.shape)
    pca = PCA(n_components=components)
    dense = pca.fit_transform(sc.fit_transform(embeddings))
    explained_variance = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(explained_variance)
    print(cum_sum_eigenvalues)
    return dense