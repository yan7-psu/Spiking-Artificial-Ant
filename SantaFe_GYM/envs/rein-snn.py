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

class CustomGymEnvironment(GymEnvironment):
    def step(self, action):
        # Get outputs from the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Combine `terminated` and `truncated` into a single `done` flag
        done = terminated or truncated

        # Process 'image' field in obs (convert to PyTorch tensor)
        if 'image' in obs and isinstance(obs['image'], np.ndarray):
            obs['image'] = torch.from_numpy(obs['image']).float()

        # Process 'direction' field (convert to tensor)
        if 'direction' in obs:
            obs['direction'] = torch.tensor(obs['direction']).float()

        # Remove or handle 'mission' field (string is not supported by BindsNET)
        obs.pop('mission', None)  # Remove the 'mission' field entirely, or process it as needed

        # Convert reward to Python float
        reward = float(reward)

        # Convert info dictionary values (if necessary)
        if isinstance(info, dict):
            info = {k: torch.tensor(v).float() if isinstance(v, (np.ndarray, float)) else v for k, v in info.items()}

        # Return processed data
        return obs, reward, done, info




# Build network.
network = Network(dt=1.0)

# Layers of neurons.
observation_space = 28  # Example: Flattened RGB input (7x7x3).
action_space = 7  # Example: Move forward, turn left, turn right.

inpt = Input(n=observation_space, traces=True)
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

            # Flatten or process obs dictionary into a format the pipeline expects
            if isinstance(obs, dict):
                # Example: Flatten the observation dictionary into a single tensor
                obs = torch.cat([value.flatten() for key, value in obs.items() if isinstance(value, torch.Tensor)])

            # Repackage the result with the flattened obs
            result = [obs, reward, done, info]

            # Pass the result to the pipeline
            pipeline.step(result)

            # Update reward and check termination
            total_reward += reward
            is_done = done
            print(f"OBS:\n {obs}")


        print(f"Episode {i} total reward: {total_reward}")


# Enable MSTDP for training.
environment_pipeline.network.learning = True

print("Training: ")
run_pipeline(environment_pipeline, episode_count=100)

# Stop MSTDP for testing.
environment_pipeline.network.learning = False

print("Testing: ")
run_pipeline(environment_pipeline, episode_count=100)

