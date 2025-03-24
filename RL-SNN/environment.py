import copy
import numpy as np

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
        # ✅ Print food locations after reset
        #food_count = sum(row.count("food") for row in self.matrix_exc)
        #print(f"🍎 Food Reloaded: {food_count} Pieces → Reset Complete")
    
    def reset(self, start_x, start_y, direction):
        self.col_start = start_x
        self.row_start = start_y
        self.dir = direction
        self._reset()  # Call the private reset method

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
            #print(f"↩️ Turn Left: {self.direction[old_dir]} → {self.direction[self.dir]}")

    def turn_right(self):
        """Turns the agent right (clockwise)."""
        if self.moves < self.max_moves:
            self.moves += 1
            old_dir = self.dir  # Store previous direction
            self.dir = (self.dir + 1) % 4  # Rotate right
            #print(f"↪️ Turn Right: {self.direction[old_dir]} → {self.direction[self.dir]}")

    def move_forward(self):
        """Moves the agent forward in the current direction."""
        #print(f"🔹 move_forward() called. matrix_row={getattr(self, 'matrix_row', 'NOT SET')}, self ID={id(self)}")
        if self.moves < self.max_moves:
            self.moves += 1
            old_row, old_col = self.row, self.col  # Store old position
            self.row = (self.row + self.dir_row[self.dir]) % self.matrix_row
            self.col = (self.col + self.dir_col[self.dir]) % self.matrix_col
            if self.matrix_exc[self.row][self.col] == "food":
                self.eaten += 1  # Increase food count when food is collected
                self.matrix_exc[self.row][self.col] = "empty"  # Remove food
            self.matrix_exc[self.row][self.col] = "passed"
            #print(f"🚶 Move: ({old_row},{old_col}) → ({self.row},{self.col}), Food Collected: {self.eaten}")        

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
        for i, line in enumerate(matrix):
            self.matrix.append(list())
            for j, col in enumerate(line):
                if col == "#":
                    self.matrix[-1].append("food")
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

        #print(f"✅ parse_matrix() executed: matrix_row={self.matrix_row}, matrix_col={self.matrix_col}, self ID={id(self)}")

    def get_state(self):
        """Returns the state as a NumPy array instead of a PyTorch tensor."""
        
        # One-hot encode the ant's X and Y position (32 each)
        ant_x_encoding = [0] * 32
        ant_y_encoding = [0] * 32
        ant_x_encoding[self.row] = 1
        ant_y_encoding[self.col] = 1

        # One-hot encode the ant's direction (4 total)
        direction_encoding = [0, 0, 0, 0]
        direction_encoding[self.dir] = 1

        # Check if food is ahead using `sense_food()`
        food_ahead = int(self.sense_food())  # Convert boolean to int

        # Combine all state information into a single list (32 + 32 + 4 + 1 = 69)
        state = ant_x_encoding + ant_y_encoding + direction_encoding + [food_ahead]

        return np.array(state, dtype=np.float32)  # ✅ Convert to NumPy array
    
    def step(self, action):

        # Interpret action (one-hot encoded)
        if action == [1, 0, 0]:  # Move Forward
            self.move_forward()
        elif action == [0, 1, 0]:  # Turn Left
            self.turn_left()
        elif action == [0, 0, 1]:  # Turn Right
            self.turn_right()

        # Reward system
        reward = 0.0

        # Reward for eating food
        if self.eaten > 0:
            reward += 1.0  # Positive reward for eating food

        # Small negative reward for each move (encourages efficiency)
        #reward -= 0.01  

        # Check if max moves reached
        done = self.moves >= self.max_moves  

        return not done, reward, self.eaten, self.moves  # Return environment state

