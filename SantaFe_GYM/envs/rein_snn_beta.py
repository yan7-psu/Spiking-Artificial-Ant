from bindsnet.encoding import bernoulli
#from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDP
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import select_softmax
from minigrid.wrappers import ImgObsWrapper
from SantaFe_env import FoodCollectorEnv  # Import your custom environment
import gymnasium as gym
from bindsnet_minigridenv import GymEnvironment
import numpy as np

from gymnasium.spaces.utils import flatten
import torch
import matplotlib.pyplot as plt

class CustomGymEnvironment(GymEnvironment):
    def step(self, action):
        # Get outputs from the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Combine `terminated` and `truncated` into a single `done` flag
        done = terminated or truncated

        # Extract only the 2nd row and 2nd column of the 'image' field
        if 'image' in obs and isinstance(obs['image'], np.ndarray):
            image_tensor = torch.from_numpy(obs['image']).float() / 11.0
            obs = image_tensor[1, 1]  # Access the 2nd row, 2nd column (0-indexed)
        else:
            raise ValueError("Expected 'image' in observation, but got: {}".format(obs.keys()))

        # Convert reward to Python float
        reward = float(reward)

        # Return processed data (obs now only contains the specific tensor value)
        return obs, reward, done, info




# Build network.
network = Network(dt=1.0)

# Layers of neurons.
observation_space = 3  # Example: Flattened RGB input (7x7x3).
action_space = 7  # Example: Move forward, turn left, turn right.

inpt = Input(n=observation_space, shape=[1,1,3],traces=True)
middle = LIFNodes(n=100, traces=True)
out = LIFNodes(n=action_space, refrac=0, traces=True)

# Connections between layers.
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDP,
    nu=1e-1,
    norm=0.5 * middle.n,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Load the FoodCollector environment.
# Register your custom environment (if not already registered)
gym.register(
    id="MiniGrid-FoodCollector-v0",
    entry_point=FoodCollectorEnv,
    max_episode_steps=100,
)

# Replace environment instantiation with the environment ID

# Load and wrap the environment
#env = gym.make("MiniGrid-FoodCollector-v0", size=7, num_food=3, render_mode="human")
#env = ImgObsWrapper(env)  # Ensures observations are arrays

#environment = GymEnvironment(name="MiniGrid-FoodCollector-v0", render_mode="human")
environment = CustomGymEnvironment(name="MiniGrid-FoodCollector-v0", render_mode="human")
environment.reset()

# Build pipeline from specified components.
environment_pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=bernoulli,
    action_function=select_softmax,
    output="Output Layer",
    time=100,
    history_length=1,
    delta=1,
    plot_interval=1,
    render_interval=1,
)

reward_hist = []

def run_pipeline(pipeline, episode_count):
    for i in range(episode_count):
        total_reward = 0
        pipeline.reset_state_variables()
        is_done = False
        while not is_done:
            # Get the environment step result
            result = pipeline.env_step()

            # Unpack the result
            obs, reward, done, info = result

            # Ensure obs is a scalar tensor (already processed in CustomGymEnvironment step)
            if not isinstance(obs, torch.Tensor):
                raise ValueError("Expected obs to be a PyTorch tensor, but got: {}".format(type(obs)))

            # Pass the result to the pipeline
            pipeline.step([obs.unsqueeze(0), reward, done, info])  # Add batch dimension if needed

            # Update reward and check termination
            total_reward += reward
            is_done = done
            #print(f"obs:\n{obs}")

        print(f"Episode {i} total reward: {total_reward}")
        reward_hist.append(total_reward)


# Enable MSTDP for training.
environment_pipeline.network.learning = True

print("Training: ")
run_pipeline(environment_pipeline, episode_count=100)

plt.plot(reward_hist)
plt.show()

# Stop MSTDP for testing.
# environment_pipeline.network.learning = False

# print("Testing: ")
# run_pipeline(environment_pipeline, episode_count=100)

