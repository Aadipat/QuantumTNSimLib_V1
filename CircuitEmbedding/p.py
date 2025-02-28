import matplotlib.pyplot as plt
import pandas as pd
from qtn_sim import *

# circuits = readCircuits("C:/Users/aadik/Downloads/Tests/Tests/data2/circuitslongDistance_depth100_num10000_q20.json")


# Load the data
df = pd.read_csv("C:/Users/aadik/Downloads/Tests/Tests/data2/allDatalongDistance_depth100_num10000_q20.csv")

# Select a random subset of the dataframe
df = df.sample(frac=0.1, random_state=1)  # Adjust frac to the desired fraction of the dataframe

# Extract pc1, pc2, and sparsity
pc1 = df['pc1']
pc2 = df['pc2']
sparsity = df['sparsity']

# Create the scatter plot with a heatmap
plt.figure()
scatter = plt.scatter(pc1, pc2, c=sparsity, cmap='viridis')
plt.colorbar(scatter, label='Sparsity')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PC1 vs PC2 with Sparsity Heatmap')
plt.show()