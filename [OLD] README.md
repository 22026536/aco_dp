# [Two-phase GRASP for the Multi-Constraint Graph Partitioning problem](https://doi.org/10.1016/j.cor.2024.106946)
_Published in Computers & Operations Research_

### Authors
- Alberto Herrán
- J. Manuel Colmenar
- Mauricio G.C. Resende

### Abstract
The Multi-Constraint Graph Partitioning (MCGP) problem seeks a partition of the node set of a graph into a fixed number of clusters such that each cluster satisfies a collection of node-weight constraints and the total cost of the edges whose end nodes are in the same cluster is minimized. In this paper we propose a two-phase reactive GRASP heuristic to find near-optimal solutions to the MCGP problem. Our proposal is able to reach all the best known results for state-of-the-art instances, obtaining all the certified optimum values while spending only a fraction of the time in relation to the previous methods. To reach these results we have implemented an efficient computation method applied in the improvement phase. Besides, we have created a new set of larger instances for the MCGP problem and provided detailed results for future comparisons.

## Datasets
Instances are located inside the `instances` folder, where two different sets of instances are provided, **pollster** and **tsplib**.

### Instance format

Instances in set **pollster** begin with a `p` followed by the number od nodes `V`, the number of clusters `K` and the number of attributes `T`. Next `V` lines begin with a `n` denoting the data associated with every node, i.e. the weights `w` of its `T` attributes and the cost `d` to any other node in the graph.

```
p V K T
n w11..w1T d11..d1V
n w21..w2T d21..d2V
:
n wV1..wVT dV1..dVV
```

Instances in set **tsplib** begin with a `t` and also include `V`, `K`, `T` and the corresponding weights `w` of every node `n`. However, instead the cost `d` between every pair of nodes, the tsplib instances include the node coordinates to compute the cost `d` as the Euclidean distance between every pair of nodes.

```
t V K T
n w11..w1T x1..y1
n w21..w2T x2..y2
:
n wV1..wVT xV..yV
```

## Source code
Source code is located inside the `src` folder. Moreover a `CMakeLists` file is provided to compile it using CMake. Remember to edit the correct path to your C++ compiler inside the `CMakeLists` file.

### Compiling

You can easily compile and build an executable artifact of this project using CMake and a recent version of the GCC compiler. For easy of use just run the script `compile.sh` and it will build an executable file in the root folder.

### Executing

To run the executable file for a specific instance and algorithm configuration, we provide the script `run.sh` that sets both the instance an the configurable parameters of the algorithm.

The list of configurable parameters are described in the paper. Additional parameters as termination criteria, logs to console, etc, are also explained in the `includes.h` file inside the `src` folder.

The results of the execution will be stored in the `results/logs` folder, with a timestamped file for each execution (`results/logs/evolution`). The corresponding solutions will be stored in the `results/logs/solutions` folder and the associated costs in the `results/logs/objectives`.

Moreover, the `results` folder also includes the experimental results used in the paper for both **pollster** and **tsplib** sets of instances.

## Cite

Consider citing our paper if used in your own work:

### DOI
[https://doi.org/10.1016/j.cor.2024.106946](https://doi.org/10.1016/j.cor.2024.106946)

### Bibtex
```bibtex
@article{HERRAN2024106946,
title = {Two-phase GRASP for the Multi-Constraint Graph Partitioning problem},
journal = {Computers & Operations Research},
pages = {106946},
year = {2024},
issn = {0305-0548},
doi = {https://doi.org/10.1016/j.cor.2024.106946},
url = {https://www.sciencedirect.com/science/article/pii/S0305054824004180},
author = {Alberto Herrán and J. Manuel Colmenar and Mauricio G.C. Resende},
keywords = {Metaheuristics, Greedy Randomized Adaptive Search Procedure, Graph partitioning},
abstract = {The Multi-Constraint Graph Partitioning (MCGP) problem seeks a partition of the node set of a graph into a fixed number of clusters such that each cluster satisfies a collection of node-weight constraints and the total cost of the edges whose end nodes are in the same cluster is minimized. In this paper we propose a two-phase reactive GRASP heuristic to find near-optimal solutions to the MCGP problem. Our proposal is able to reach all the best known results for state-of-the-art instances, obtaining all the certified optimum values while spending only a fraction of the time in relation to the previous methods. To reach these results we have implemented an efficient computation method applied in the improvement phase. Besides, we have created a new set of larger instances for the MCGP problem and provided detailed results for future comparisons.}
}
```
