import torch
import random
import numpy as np
from collections import deque
from model import DSQN, SNNTrainer
from environment import SantaFeEnvironment
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4

class Agent:
    def __init__(self):
        self.num_runs = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        # Create DSQN Networks
        self.model = DSQN(69, 256, 3)  # Correct input size from environment
        self.trainer = SNNTrainer(self.model, self.model, LEARNING_RATE, self.gamma)

    def get_state(self, environment):
        return environment.get_state()  # Now correctly returns a NumPy array

    def remember(self, state, action, reward, next_state, run):
        self.memory.append((state, action, reward, next_state, run))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, runs = zip(*mini_sample)
        loss = self.trainer.train_step(states, actions, rewards, next_states, runs)

        return loss

    def train_short_memory(self, state, action, reward, next_state, run):
        return self.trainer.train_step(state, action, reward, next_state, run)

    def get_action(self, state):
        self.epsilon = max(0.01, 0.9 - 0.01 * self.num_runs)  #  Greedy-epsilon
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            spk2_rec = self.model(state_tensor)
            move = torch.argmax(spk2_rec).item()

        final_move[move] = 1
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_losses = []
    plot_mean_losses = []
    total_score = 0
    max_score = 0
    min_time_steps = 600
    agent = Agent()
    environment = SantaFeEnvironment()

    # Load Santa Fe Trail from text file
    with open("santafe_trail.txt") as trail_file:
        environment.parse_matrix(trail_file)

    while True:
        state_old = agent.get_state(environment)
        final_move = agent.get_action(state_old)

        # Take a step in the environment
        run, reward, score, time_steps = environment.step(final_move)
        state_new = agent.get_state(environment)

        # Train agent
        agent.train_short_memory(state_old, final_move, reward, state_new, run)
        agent.remember(state_old, final_move, reward, state_new, run)

        # Check if episode ended
        if not run:
            if environment.eaten == 89:
                agent.save_model('best_model.pth')

            # Reset environment and train long memory
            environment.reset(environment.col_start, environment.row_start, 1)  # 1 = Right (East)
            agent.num_runs += 1
            long_loss = agent.train_long_memory()
            plot_losses.append(long_loss)

            if score > max_score:
                max_score = score
            if time_steps < min_time_steps:
                min_time_steps = time_steps

            print(f"Run #{agent.num_runs} - Score: {score}, Highest Score: {max_score} | "
                  f"Time Steps: {time_steps}, Lowest Time Steps: {min_time_steps}", flush=True)

            # Update plots
            mean_loss = sum(plot_losses) / len(plot_losses)
            plot_mean_losses.append(mean_loss)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_runs
            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores, plot_losses, plot_mean_losses)

if __name__ == "__main__":
    train()
