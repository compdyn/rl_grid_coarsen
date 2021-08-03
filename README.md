# Optimization-Based Algebraic Multigrid Coarsening Using Reinforcement Learning
Code for reproducing the experimental results in our paper:
https://arxiv.org/pdf/2106.01854.pdf

## Requirements
 * Python = 3.8.10
 * PyTorch = 1.7.1
 * Pytorch_Geometric = 1.6.3
 * pyamg = 4.1.0
 * networkx = 2.6.2
 * pygmsh = 7.1.6
 * gmsh = 4.8.0
 
 

## Training
```
python main.py
```

## Files

* Batch_Graph.py: The class for making a batch of grids to be the input to the agent's network.
* DuelingNet.py: The netwrok architecture that the agent utilizes (TAGCN network).
* Lloyd_Unstructured.py: The code for the generating Lloyd agregations on grids.
* MG_Agent.py: The class for Dueling Double DQN agent.
* Optim.py: The code for running the grid coarsinging using Lloyed aggregation.
* Solve_MG.py: The code for the two grid cycle AMG algorithm.
* Unstructured.py: The class for defining grids and making unstructured meshes. 
* fem.py: Constructs a finite element discretization of a 2D Poisson problem.
* main.py: The main driver for training the RL agent and test functions.
