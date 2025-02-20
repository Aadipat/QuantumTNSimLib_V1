from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


from qtn_sim import *
n = 10
circuitsGHZ = [GHZCircuit(i).toString() for i in range(1,n)]
circuitsQFT = [QFTCircuit(i).toString() for i in range(1,n)]
circuitsWState = [WCircuitLinear(i).toString() for i in range(1,n)]


embeddingsGHZ = model.encode(circuitsGHZ)
embeddingsQFT = model.encode(circuitsQFT)
embeddingsWState = model.encode(circuitsWState)


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
x = pca.fit_transform(embeddingsGHZ)
pca = PCA(n_components=3)
y = pca.fit_transform(embeddingsQFT)
z = pca.fit_transform(embeddingsWState)


import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x[:, 0],x[:, 1],x[:, 2])
ax.scatter3D(y[:, 0],y[:, 1],y[:, 2])
ax.scatter3D(z[:, 0],z[:, 1],z[:, 2])
ax.legend()
plt.show()