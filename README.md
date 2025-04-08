# Spiking Artifical Ant

This repo looks at Koza's Artificial Ant problem and attempts to solve the Santa Fe Trail with a Spiking Neural network.

# Network Architecture

The Spiking Neural Network (SNN) is composed of SLSTM neurons, enabling the network to retain memory. The architecture consists of an input layer, one hidden layer, and an output layer. The input layer contains 69 neurons, representing the state of the ant. The output layer has 3 neurons, each corresponding to a possible action the ant can take.

# Results
** WIP **
Test Trail 1 - test1.txt
![Fig1](GA-SNN/test1.plot.png)
Figure 1: DSQN solved test trail 1 in 68 generation with 115 moves

# References:
[Chemical Reaction Network Control Systems for Agent-Based Foraging Tasks](https://pdxscholar.library.pdx.edu/open_access_etds/2203/)

[Integrating Deep Q-Networks and Reinforcement Learning for Autonomous Navigation in the Santa Fe Trail Problem](https://core.ac.uk/download/621339497.pdf)
