from qtn_sim import *

def getQubits(qcircuit : QCircuit):
    # Calculate the number of qubits
    num_qubits = 0
    for qubits in map((lambda x: x[1]), qcircuit.gateList):
        num_qubits = max(num_qubits, max(qubits))
    num_qubits += 1  # Since qubits are zero-indexed
    return num_qubits


# We can also partition a circuit into partitions
# where each partition is a list of gates and the partitions can be run in parallel
# This is useful for parallelizing the execution of the circuit
# The function below converts a QCircuit to a list of partitions
# where each partition is a list of gates that can be run in parallel
# We need to go through the gateList
# and add gates to new partitions if they act on qubits that are not occupied
# else we add them to the respective partition

# this is basically finding the connected components in the circuit graph
# where the vertices are the qubits and the edges are the gates
# and the partitions are the connected components, just we store the 
# gates instead of the vertices

# gate is my Gate class from qtn_sim
# qcircuit is my QCircuit class from qtn_sim
def qcircuit_to_partitions(qcircuit : QCircuit) -> list[QCircuit]:
    partitions = []
    occupied_qubits_per_partition = []

    for gate, qubits in qcircuit.gateList:
        # Find the partitions that contain the qubits
        partitions_with_qubits = [i for i, partition in enumerate(occupied_qubits_per_partition) if any(q in partition for q in qubits)]
        
        # If there are no partitions with the qubits, create a new partition
        if not partitions_with_qubits:
            partitions.append([(gate, qubits)])
            occupied_qubits_per_partition.append(set(qubits))
        else:
            # Add the gate to the first partition that contains the qubits
            partitions[partitions_with_qubits[0]].append((gate, qubits))
            occupied_qubits_per_partition[partitions_with_qubits[0]].update(qubits)
            # If there are multiple partitions with the qubits, merge them
            if len(partitions_with_qubits) > 1:
                partitions_with_qubits = sorted(partitions_with_qubits, reverse=True)
                for i in partitions_with_qubits[1:]:
                    if partitions_with_qubits[0] < len(partitions):
                        partitions[partitions_with_qubits[0]].extend(partitions[i])
                        occupied_qubits_per_partition[partitions_with_qubits[0]].update(occupied_qubits_per_partition[i])
                        if i < len(partitions):
                            del partitions[i]
                        if i < len(occupied_qubits_per_partition):
                            del occupied_qubits_per_partition[i]
                    else: 
                        partitions.append([(gate, qubits)])
                        occupied_qubits_per_partition.append(set(qubits))
    
    return list(map(lambda p: QCircuit(p), partitions))
# This function converts a
# QCircuit from my qtn_sim library
# to a list of layers
# using a topological sort

# The layers are a list of lists
# where each list is a layer of gates
# that can be applied in parallel
# gate is my Gate class from qtn_sim
# qcircuit is my QCircuit class from qtn_sim

# QCircuit has a gateList attribute
# which is a list of (Gate, list[int])
# where Gate is a Gate object
# and list[int] is the list of qubits
# that the gate acts on

# qCircuit_to_layers : QCircuit -> List[List[Gate]]
def qcircuit_to_layers(qcircuit : QCircuit):
    layers = []
    current_layer = []
    occupied_qubits = set()

    # Go through the gates in the qcircuit and add them to the layers
    for gate, qubits in qcircuit.gateList:
        # If the gate acts on qubits that are occupied in the current layer, start a new layer
        if any(q in occupied_qubits for q in qubits):
            layers.append(current_layer)
            current_layer = []
            occupied_qubits = set()
        
        current_layer.append(gate)
        occupied_qubits.update(qubits)

    # Add the last layer if it's not empty
    if current_layer:
        layers.append(current_layer)

    return layers

# Test the function
# Let us consider the following large circuit
qcircuit = getRandomCircuit(10,10,False)[0]
print(qcircuit.toString())
layers = qcircuit_to_layers(qcircuit)
partitions = qcircuit_to_partitions(qcircuit)

# for i, layer in enumerate(layers):
#     print(f"Layer {i}:")
#     for gate in layer:
#         print(gate.id)
#     print()

for i, partition in enumerate(partitions):
    print(f"Partition {i}:")
    print(partition.toJSONDict())
    print()