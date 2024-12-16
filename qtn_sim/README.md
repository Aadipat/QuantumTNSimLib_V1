This library contains modules to simulate quantum circuits using tensor networks as well as some benchmarking api.

I am exploring specifically tensor native representations to optimise simulation:

- Different tensor networks - QLibs
- Tensor contraction orders and strategies. - EinsumLib

Author: Aadi Patwardhan


The API:

Tensor networks:
    QuantumMPS
    QuantumTensor
Einstein summation library:
    get einsum string for tensor and circuit + apply a gate on a tensor.

Some plotting functions:
    Over bond dimension, qubit number einstein summation strategy.
    Plots memory usage and execution time.

Optimisers: 
    sequential + with swapping for non-adjacent
    oneShot
    

I have a small testSuite that can be run.



Install build package: pip install build
Install the wheel file pip install <path_to_wheel>
Now you can import qtn_sim