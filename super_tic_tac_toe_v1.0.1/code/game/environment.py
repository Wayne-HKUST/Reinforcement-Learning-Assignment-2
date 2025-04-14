import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Filtering tensorFlow warning messages
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import logging

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'environment.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SuperTicTacToeEnv(py_environment.PyEnvironment):
    def __init__(self):
        # Initialize valid area (cross-shaped board)
        self.valid_area = np.zeros((12, 12), dtype=bool)
        self.valid_area[4:8, 4:8] = True  # center
        self.valid_area[0:4, 4:8] = True   # up
        self.valid_area[8:12, 4:8] = True  # down
        self.valid_area[4:8, 0:4] = True   # left
        self.valid_area[4:8, 8:12] = True  # right
        
        self._board = np.zeros((12, 12), dtype=np.int8)
        self._current_player = 1
        self._episode_ended = False
        self._current_time_step = None
        self.valid_positions = [(i,j) for i in range(12) for j in range(12) if self.valid_area[i,j]]
        self.num_valid_positions = len(self.valid_positions)
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.num_valid_positions - 1, name='action')

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(12, 12, 3), dtype=np.float32, minimum=0.0, maximum=2.0, name='observation')

    def _check_consecutive(self, row, col):
        """Check maximum consecutive count in four directions for the newly placed piece"""
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # main diagonal
            (1, -1)   # anti-diagonal
        ]
        max_consecutive = 1
        player = self._current_player
        
        for dx, dy in directions:
            count = 1
            # forward check
            i, j = row + dx, col + dy
            while 0 <= i < 12 and 0 <= j < 12 and self.valid_area[i][j]:
                if self._board[i][j] == player:
                    count += 1
                    i += dx
                    j += dy
                else:
                    break
            
            # backward check
            i, j = row - dx, col - dy
            while 0 <= i < 12 and 0 <= j < 12 and self.valid_area[i][j]:
                if self._board[i][j] == player:
                    count += 1
                    i -= dx
                    j -= dy
                else:
                    break
            
            max_consecutive = max(max_consecutive, count)
        
        return max_consecutive

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        logging.info("Resetting the environment")
        self._board = np.zeros((12, 12), dtype=np.int8)
        self._current_player = 1
        self._episode_ended = False
        self._current_time_step = ts.restart(self._get_obs())
        logging.info("Environment reset completed")
        return self._current_time_step

    def _step(self, action):
        logging.info(f"Received action: {action}")
        if self._episode_ended:
            return self.reset()

        if 0 <= action < len(self.valid_positions):
            row, col = self.valid_positions[action]
            if not self.valid_area[row][col] or self._board[row][col] != 0:
                # illegal move, give a large negative reward
                reward = -128
                return ts.transition(self._get_obs(), reward=reward)

            placed = self._safe_place(row, col)
            if placed:
                # legal placement, give a small positive reward
                consecutive = self._check_consecutive(row, col)
                reward = 2 ** consecutive  # Rewards are proportional to the number of consecutive counts
            else:
                # placement failure, give a small negative reward
                reward = -2

            if self._check_win():
                reward = 64 if self._current_player == 1 else -64

                self._episode_ended = True
            elif np.sum(self._board == 0) == 0:
                reward = -16
                self._episode_ended = True
            else:
                self._current_player = 3 - self._current_player

            logging.info(f"Step result: placed={placed}, reward={reward}, episode_ended={self._episode_ended}")

            if self._episode_ended:
                self._current_time_step = ts.termination(self._get_obs(), reward)
            else:
                self._current_time_step = ts.transition(self._get_obs(), reward=reward)

            return self._current_time_step
        else:
            # Action index out of range, give a large negative reward
            reward = -10
            logging.error(f"Invalid action received in environment: {action}")
            return ts.transition(self._get_obs(), reward=reward)

    def _get_obs(self):
        obs = np.zeros((12, 12, 3), dtype=np.float32)
        obs[..., 0] = (self._board == 1).astype(np.float32)
        obs[..., 1] = (self._board == 2).astype(np.float32)
        obs[..., 2] = self.valid_area.astype(np.float32)
        return obs

    def _safe_place(self, row, col, main_prob=0.5):
        """Probability placement mechanism implementation"""
        if np.random.random() < main_prob:  # Main position attempt
            if self._board[row, col] == 0:
                self._board[row, col] = self._current_player
                return True
            return False
        
        # Random attempts at adjacent positions
        neighbors = [
            (row-1, col-1), (row-1, col), (row-1, col+1),
            (row, col-1),          (row, col+1),
            (row+1, col-1), (row+1, col), (row+1, col+1)
        ]
        valid_neighbors = [
            (i,j) for i,j in neighbors 
            if 0<=i<12 and 0<=j<12 
            and self.valid_area[i,j]
            and self._board[i,j]==0
        ]
        
        if not valid_neighbors:
            return False
        
        # Probability of 1/16 for each neighbor
        for _ in range(16):
            i,j = valid_neighbors[np.random.randint(len(valid_neighbors))]
            if self._board[i,j] == 0:
                self._board[i,j] = self._current_player
                return True
        return False

    def _check_line(self, cells, win_count):
        player = self._current_player
        max_count = current = 0
        for cell in cells:
            current = current + 1 if cell == player else 0
            max_count = max(max_count, current)
        return max_count >= win_count

    def _check_win(self):
        """Exact winning condition check"""
        # Horizontal check (4-continuous)
        for i in range(12):
            for j in range(9):
                if self._board[i,j] == self._current_player and \
                   np.all(self._board[i,j:j+4] == self._current_player):
                    return True
        
        # Vertical check (4-continuous)
        for j in range(12):
            for i in range(9):
                if self._board[i,j] == self._current_player and \
                   np.all(self._board[i:i+4,j] == self._current_player):
                    return True
        
        # Diagonal check (5-continuous)
        for i in range(8):
            for j in range(8):
                if np.all([self._board[i+k,j+k] == self._current_player for k in range(5)]):
                    return True
                if np.all([self._board[i+k,j+4-k] == self._current_player for k in range(5)]):
                    return True
        return False

    def get_human_view(self):
        return self._board.copy(), self._current_player