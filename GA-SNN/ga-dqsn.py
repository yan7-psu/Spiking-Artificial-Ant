#########################################
# GA for SNN Weight Initialization - Santa Fe Trail
# Inspired by Sudoku-SFT-GA and DEAP GP Ant Trail Example
#########################################
import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import random
import logging
import sys
import matplotlib.pyplot as plt
import operator
import copy
import multiprocessing
from snntorch import surrogate

from deap import algorithms, base, creator, tools
from functools import partial

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

# Define Spiking Neural Network (SNN)
beta = 0.9        # Decay rate for Leaky Integrate-and-Fire (LIF) neurons

class SpikingNet(nn.Module):
    def __init__(self, num_inputs=69, num_hidden=128, num_outputs=3):
        super().__init__()

        spike_grad_lstm = surrogate.straight_through_estimator()

        # SLSTM Layers
        self.slstm1 = snn.SLSTM(num_inputs, num_hidden, spike_grad=spike_grad_lstm)
        self.slstm2 = snn.SLSTM(num_hidden, num_outputs, spike_grad=spike_grad_lstm)

    def forward(self, x, num_steps=10):
        """ Forward pass through the SLSTM SNN """

        # Initialize SLSTM states at t=0
        syn1, mem1 = self.slstm1.init_slstm()
        syn2, mem2 = self.slstm2.init_slstm()

        # Store outputs for multiple time steps
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            spk1, syn1, mem1 = self.slstm1(x, syn1, mem1)  # First SLSTM layer
            spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)  # Output SLSTM layer

            spk2_rec.append(spk2)  # Record spikes
            mem2_rec.append(mem2)  # Record membrane potential

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)  # Stack over time

# Genetic Algorithm Parameters

# Final Genome Length Calculation (Use convert_genome_to_snn error to find value)
GENOME_LENGTH = 103490

# Define DEAP types for GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, GENOME_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#3. Convert GA Genome to SNN
import torch
import numpy as np

def convert_genome_to_snn(genome):
    """Converts a genome (flat list of weights) into an SLSTM-based spiking network model."""
    model = SpikingNet()
    state_dict = model.state_dict()  # Get model parameters
    genome_array = np.array(genome)  # Convert genome to NumPy array
    index = 0  # Track genome position

    # Debugging: Check actual parameter count
    actual_params = sum(np.prod(state_dict[key].shape) for key in state_dict)
    #print(f" Expected GENOME_LENGTH: {GENOME_LENGTH}")
    #print(f" Actual network parameter count: {actual_params}")

    if len(genome_array) != actual_params:
        raise ValueError(f"Genome size {len(genome_array)} does not match network parameter size {actual_params}.")

    for key in state_dict:
        shape = state_dict[key].shape  # Expected shape of weights
        num_params = np.prod(shape)  # Number of parameters in this layer

        # Exclude `reset_mechanism_val` and `graded_spikes_factor`
        if "reset_mechanism_val" in key or "graded_spikes_factor" in key:
            continue  # Skip these parameters

        if index + num_params > len(genome_array):
            raise ValueError(f"Genome size {len(genome_array)} is too small for network parameters {index+num_params}")

        # Handle scalar parameters separately
        if shape == torch.Size([]):  # Scalars (threshold)
            state_dict[key] = torch.tensor(genome_array[index], dtype=torch.float32)
            index += 1
        else:
            weights = genome_array[index: index + num_params].reshape(shape)
            state_dict[key] = torch.tensor(weights, dtype=torch.float32)
            index += num_params

    model.load_state_dict(state_dict)  # Load the modified weights
    return model


# Santa Fe Trail Environment
import copy
import random

class SantaFeEnvironment:
    direction = ["north", "east", "south", "west"]
    dir_row = [1, 0, -1, 0]
    dir_col = [0, 1, 0, -1]

    def __init__(self, max_moves=600):
        self.max_moves = max_moves
        self.moves = 0
        self.eaten = 0
        self.routine = None
        self.row = 0
        self.col = 0
        self.dir = 1  # Ensure direction is always initialized

    def _reset(self):
        """Resets the environment state for a new episode."""
        self.row = self.row_start
        self.col = self.col_start
        self.dir = 1  # Always reset direction
        self.moves = 0
        self.eaten = 0
        self.matrix_exc = copy.deepcopy(self.matrix)
        # Print food locations after reset
        #food_count = sum(row.count("food") for row in self.matrix_exc)
        #print(f"Food Reloaded: {food_count} Pieces → Reset Complete")

    @property
    def position(self):
        """Returns the current position and direction of the agent."""
        return (self.row, self.col, self.direction[self.dir])

    def turn_left(self):
        """Turns the agent left (counterclockwise)."""
        if self.moves < self.max_moves:
            self.moves += 1
            old_dir = self.dir  # Store previous direction
            self.dir = (self.dir - 1) % 4  # Rotate left
            #print(f"Turn Left: {self.direction[old_dir]} → {self.direction[self.dir]}")

    def turn_right(self):
        """Turns the agent right (clockwise)."""
        if self.moves < self.max_moves:
            self.moves += 1
            old_dir = self.dir  # Store previous direction
            self.dir = (self.dir + 1) % 4  # Rotate right
            #print(f"Turn Right: {self.direction[old_dir]} → {self.direction[self.dir]}")

    def move_forward(self):
        """Moves the agent forward in the current direction."""
        #print(f"move_forward() called. matrix_row={getattr(self, 'matrix_row', 'NOT SET')}, self ID={id(self)}")
        if self.moves < self.max_moves:
            self.moves += 1
            old_row, old_col = self.row, self.col  # Store old position
            self.row = (self.row + self.dir_row[self.dir]) % self.matrix_row
            self.col = (self.col + self.dir_col[self.dir]) % self.matrix_col
            if self.matrix_exc[self.row][self.col] == "food":
                self.eaten += 1  # Increase food count when food is collected
                self.matrix_exc[self.row][self.col] = "empty"  # Remove food
            self.matrix_exc[self.row][self.col] = "passed"
            #print(f"Move: ({old_row},{old_col}) → ({self.row},{self.col}), Food Collected: {self.eaten}")        

    def sense_food(self):
        """Checks if food is ahead in the direction the agent is facing."""
        ahead_row = (self.row + self.dir_row[self.dir]) % self.matrix_row
        ahead_col = (self.col + self.dir_col[self.dir]) % self.matrix_col
        return self.matrix_exc[ahead_row][ahead_col] == "food"

    def if_food_ahead(self, out1, out2):
        """Performs one of two actions based on whether food is ahead."""
        return out1() if self.sense_food() else out2()

    def run(self, routine):
        """Runs a given routine for a complete episode."""
        self._reset()
        while self.moves < self.max_moves:
            routine()

    def parse_matrix(self, matrix):
        self.matrix = list()
        self.total_food = 0
        for i, line in enumerate(matrix):
            self.matrix.append(list())
            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("food")
                    self.total_food += 1
                elif col == ".":
                    self.matrix[-1].append("empty")
                elif col == "S":
                    self.matrix[-1].append("empty")
                    self.row_start = self.row = i
                    self.col_start = self.col = j
                    self.dir = 1
        self.matrix_row = len(self.matrix)
        self.matrix_col = len(self.matrix[0])
        self.matrix_exc = copy.deepcopy(self.matrix)

        #print(f" parse_matrix() executed: matrix_row={self.matrix_row}, matrix_col={self.matrix_col}, self ID={id(self)}")

    def get_state(self):
        """Returns the state representation of the ant's position, direction, and food presence ahead."""

        # One-hot encode the ant's X and Y position (32 each)
        ant_x_encoding = [0] * 32
        ant_y_encoding = [0] * 32
        ant_x_encoding[self.row] = 1
        ant_y_encoding[self.col] = 1

        # One-hot encode the ant's direction (4 total)
        direction_encoding = [0, 0, 0, 0]
        direction_encoding[self.dir] = 1

        # Determine if food is ahead
        ahead_x = self.row + self.dir_row[self.dir]
        ahead_y = self.col + self.dir_col[self.dir]

        food_ahead = 0
        if 0 <= ahead_x < self.matrix_row and 0 <= ahead_y < self.matrix_col:
            food_ahead = 1 if self.matrix_exc[ahead_y][ahead_x] == "food" else 0

        # Combine all state information into a single list (32 + 32 + 4 + 1 = 69)
        state = ant_x_encoding + ant_y_encoding + direction_encoding + [food_ahead]

        # Convert to PyTorch tensor and add batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 69)

        # 🔍 Extract Debugging Values
        # extracted_x = ant_x_encoding.index(1)  # Find the index where 1 is placed
        # extracted_y = ant_y_encoding.index(1)  # Find the index where 1 is placed
        # extracted_direction = direction_encoding.index(1)  # Find index of 1 in direction
        # extracted_food = food_ahead  # This is already a boolean

        # print(f" Extracted State - Ant_X: {extracted_x}, Ant_Y: {extracted_y}, Direction: {extracted_direction}, Food Ahead: {extracted_food}")
        return state_tensor


def evaluate_fitness(individual, environment):
    """Runs the GA-evolved SNN in the Santa Fe Trail environment and evaluates performance."""
    model = convert_genome_to_snn(individual)

    environment._reset() 

    time_steps = 0
    while environment.moves < environment.max_moves and environment.eaten < environment.total_food:
        state_tensor = environment.get_state()
        spk_out, mem_out = model(state_tensor)  # Get Q-values
        action_values = spk_out.sum(dim=0)
        action = torch.argmax(action_values).item()  # Select best action

        # Execute chosen action
        if action == 0:
            environment.move_forward()
        elif action == 1:
            environment.turn_left()
        elif action == 2:
            environment.turn_right()

        time_steps += 1  # Track number of steps

    collected_food = environment.eaten

    if collected_food >= environment.total_food:
        print(f"All food pellets eaten at {environment.moves} moves!")
        torch.save(model.state_dict(), "solved_solution.pth")
        sys.exit("Solved solution found. Exiting.")

    fitness = collected_food - (0.01 * environment.moves)
    fitness = max(fitness, 0)
    
    #print(f" Fitness Evaluation: Collected Food={collected_food}, Time Steps={time_steps}, Final Fitness={fitness}")

    return (fitness,)

import multiprocessing
import os 

t_path = os.path.join(os.path.dirname(__file__),"santafe_trail.txt")

# Wrapper function to pass environment explicitly
def evaluate_fitness_wrapper(individual):
    """Each worker creates its own environment instance."""
    environment = SantaFeEnvironment()  
    with open(t_path) as trail_file:
        environment.parse_matrix(trail_file)  
    
    return evaluate_fitness(individual, environment)  

toolbox.register("evaluate", evaluate_fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.05)


#  Run GA

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def main():
    random.seed(70)

    mu = 200
    lambda_ = 300
    cxpb = 0.6
    mutpb = 0.2
    ngen = 5000

    pop = toolbox.population(n=mu)

    pool = multiprocessing.Pool(processes=6)
    toolbox.register("map", pool.map)
    toolbox.register("evaluate", evaluate_fitness_wrapper)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Evaluate initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    # === Main Evolution Loop ===
    for gen in range(1, ngen + 1):
        # Variation Strategy: Manual crossover + mutation (Variation 5)
        offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, lambda_)]

        # Crossover (pairwise)
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i-1], offspring[i] = toolbox.mate(
                    offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values

        # Mutation (per individual)
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Evaluate the new offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Combine parents + offspring and apply (mu + lambda) selection
        combined = pop + offspring
        pop[:] = toolbox.select(combined, mu)

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    # Plot and Log
    logging.info("Best genome found!")
    print(logbook)

    gen = logbook.select("gen")
    fit_mins = logbook.select("Min")
    fit_maxs = logbook.select("Max")
    fit_avgs = logbook.select("Avg")

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_avgs, label="Avg Fitness")
    ax1.plot(gen, fit_mins, label="Min Fitness")
    ax1.plot(gen, fit_maxs, label="Max Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend(loc="lower right")

    best_snn = convert_genome_to_snn(hof[0])
    torch.save(best_snn.state_dict(), "best_snn.pth")

    pool.close()
    pool.join()
    plt.show()

    return pop, hof, stats

if __name__ == "__main__":
    main()
