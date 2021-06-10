# rl_grid_coarsen

* Batch_Graph.py: The class for making a batch of grids to be the input to the agent's network
* DuelingNet.py: The netwrok architecture that the agent utilizes (TAGCN network)
* Lloyd_Unstructured.py: The code for the generating Lloyd agregations on grids
* MG_Agent.py: The class for Dueling Double DQN agent
* Optim.py: The code for running the grid coarsinging using Lloyed aggregation
* Solve_MG.py: The code for the two grid cycle AMG algorithm
* Unstructured.py: The class for defining grids. 
* fem.py: Constructs a finite element discretization of a 2D Poisson problem
* main.py: The main driver for training the RL agent
