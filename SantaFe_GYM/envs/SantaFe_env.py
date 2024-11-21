import gymnasium as gym
from minigrid.core.grid import Grid, WorldObj, OBJECT_TO_IDX
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper

# Define the color-to-index mapping
COLOR_TO_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5,
}

# Define the color RGB mapping
IDX_TO_RGB = {
    0: (255, 0, 0),    # red
    1: (0, 255, 0),    # green
    2: (0, 0, 255),    # blue
    3: (160, 32, 240), # purple
    4: (255, 255, 0),  # yellow
    5: (128, 128, 128) # grey
}

# Register the custom object type
OBJECT_TO_IDX["food"] = len(OBJECT_TO_IDX)

class Food(WorldObj):
    """
    Custom Food object that disappears when the agent overlaps it.
    """

    def __init__(self):
        super().__init__("food", "green")

    def can_overlap(self):
        # The agent can walk over the food
        return True

    def render(self, img):
        # Render the food as a solid green square
        color = IDX_TO_RGB[COLOR_TO_IDX[self.color]]
        img[:, :, 0] = color[0]  # Red channel
        img[:, :, 1] = color[1]  # Green channel
        img[:, :, 2] = color[2]  # Blue channel


class FoodCollectorEnv(MiniGridEnv):
    """
    Custom MiniGrid environment where the agent must collect all food pellets to complete the task.
    """

    def __init__(self, size=7, num_food=5, **kwargs):
        self.num_food = num_food
        self.food_positions = set()
        super().__init__(
            grid_size=size,
            max_steps=4 * size**2,
            see_through_walls=True,
            agent_view_size=3,  # The minimum required view size (kept as 3)
            mission_space=MissionSpace(
                mission_func=lambda: "collect all food pellets"
            ),
            **kwargs,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the outer walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent at a random position
        self.place_agent()

        # Randomly place food pellets in the environment
        for _ in range(self.num_food):
            while True:
                x, y = self._rand_int(1, width - 1), self._rand_int(1, height - 1)
                if (x, y) not in self.food_positions and (x, y) != self.agent_pos:
                    self.grid.set(x, y, Food())
                    self.food_positions.add((x, y))
                    break

        # Track collected food
        self.collected_food = set()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        reward -= 0.1

        # Check if the agent is on a food pellet
        agent_pos = tuple(self.agent_pos)
        if agent_pos in self.food_positions:
            self.collected_food.add(agent_pos)
            self.food_positions.remove(agent_pos)
            self.grid.set(*agent_pos, None)  # Remove the food pellet
            reward += 1  # Add reward for collecting food

        # Check if all food pellets have been collected
        if not self.food_positions:
            reward += self._reward()  # Additional reward for task completion
            terminated = True


        return obs, reward, terminated, truncated, info


# Register the environment
gym.register(
    id="MiniGrid-FoodCollector-v0",
    entry_point=FoodCollectorEnv,
    max_episode_steps=100,
)

# Test the environment
# for Image obs, the square in front of agent is row 2, column 2
if __name__ == "__main__":
    env = gym.make("MiniGrid-FoodCollector-v0", size=7, num_food=3, render_mode="human")
    #env = ImgObsWrapper(env)
    obs,_ = env.reset()
    done = False
    env.render()
    #env_obs = ViewSizeWrapper(env, agent_view_size=7)

    while not done:
        action = env.action_space.sample()  # Random actions
        obs, reward, done, _, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}")
        print("Observation:")
        print(obs)
        print(f"observation front: \n{obs[1][1]}")
        

    env.close()
