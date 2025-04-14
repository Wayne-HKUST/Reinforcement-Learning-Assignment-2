import os
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # filter the warning of TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # set the TensorFlow logging level to ERROR
import pygame
import sys
from tf_agents.trajectories import time_step as ts
from game.environment import SuperTicTacToeEnv
from game.model_manager import ModelManager
import logging
import numpy as np

# Configure the TensorFlow logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'play.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")


class GameGUI:
    def __init__(self):
        pygame.init()
        # Reduce window size.
        self.screen = pygame.display.set_mode((800, 800))
        self.env = SuperTicTacToeEnv()
        self.manager = ModelManager()
        self.ai_policy = self.manager.load_latest_model()
        if self.ai_policy:
            try:
                # Try to get the 'action' signature.
                self.ai_policy = self.ai_policy.signatures['action']
            except KeyError:
                logging.error("The 'action' signature was not found in the saved model.")
                self.ai_policy = None
        logging.info(f"Loaded model type: {type(self.ai_policy)}")
        if self.ai_policy:
            logging.info(f"Model attributes: {dir(self.ai_policy)}")
        # Reduce cell size
        self.cell_size = 60
        # Adjust the offset
        self.offset = 30
        self.font = pygame.font.Font(None, 36)

    def draw_board(self):
        self.screen.fill((255, 255, 255))

        # Draw a chessboard
        for i in range(12):
            for j in range(12):
                rect = pygame.Rect(
                    self.offset + j * self.cell_size,
                    self.offset + i * self.cell_size,
                    self.cell_size, self.cell_size)

                # Draw only the valid area
                if self.env.valid_area[i][j]:
                    pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

                    # Draw the chess pieces
                    if self.env._board[i][j] == 1:
                        pygame.draw.circle(self.screen, (255, 0, 0), rect.center, 20)
                    elif self.env._board[i][j] == 2:
                        pygame.draw.circle(self.screen, (0, 0, 255), rect.center, 20)

        pygame.display.flip()

    # Calculates the row and column indices based on the clicked position
    def get_clicked_pos(self, pos):
        x, y = pos
        col = (x - self.offset) // self.cell_size
        row = (y - self.offset) // self.cell_size
        # Ensure the calculation results are within a valid range
        col = max(0, min(int(col), 11))
        row = max(0, min(int(row), 11))
        print(f"Calculated row: {row}, col: {col}")
        return row, col

    # Select an action for the AI based on its policy, ensuring the action is valid and handling any exceptions.
    def ai_move(self):
        if self.ai_policy:
            try:
                time_step = ts.restart(self.env._get_obs())
                time_step = {
                    'step_type': tf.convert_to_tensor([time_step.step_type], dtype=tf.int32),
                    'reward': tf.convert_to_tensor([time_step.reward], dtype=tf.float32),
                    'discount': tf.convert_to_tensor(time_step.discount, dtype=tf.float32),
                    'observation': tf.convert_to_tensor(time_step.observation, dtype=tf.float32)
                }
                action_step = self.ai_policy(**time_step)
                try:
                    action = action_step['output_0'].numpy()[0]
                except KeyError:
                    action = action_step.numpy()[0]

                # Filter out illegal moves
                valid_actions = []
                for i, (row, col) in enumerate(self.env.valid_positions):
                    if self.env._board[row][col] == 0:
                        valid_actions.append(i)

                if action not in valid_actions:
                    if valid_actions:
                        action = np.random.choice(valid_actions)
                    else:
                        return

                print(f"AI action: {action}")
                self.env.step(action)
            except Exception as e:
                logging.error(f"Error in AI move: {e}")

    def show_result(self, winner):
        # Create a semi-transparent surface
        overlay = pygame.Surface((800, 800), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 128))  # Set a semi-transparent white background

        if winner == 2:
            text = self.font.render("You win!", True, (0, 128, 0))  # Green text
        else:
            text = self.font.render("AI wins!", True, (128, 0, 0))  # Red text
        overlay.blit(text, (350, 300))

        continue_text = self.font.render("Continue", True, (0, 0, 0))
        exit_text = self.font.render("Exit", True, (0, 0, 0))

        # Draw a continue button
        continue_button_rect = pygame.Rect(350, 350, continue_text.get_width() + 20, continue_text.get_height() + 20)
        pygame.draw.rect(overlay, (200, 200, 200), continue_button_rect)
        overlay.blit(continue_text, (360, 360))

        # Draw an exit button
        exit_button_rect = pygame.Rect(350, 400, exit_text.get_width() + 20, exit_text.get_height() + 20)
        pygame.draw.rect(overlay, (200, 200, 200), exit_button_rect)
        overlay.blit(exit_text, (360, 410))

        # Displays the overlay, handle events for quitting or clicking buttons to continue or exit the game
        self.screen.blit(overlay, (0, 0))
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    if continue_button_rect.collidepoint(mouse_pos):
                        return True
                    elif exit_button_rect.collidepoint(mouse_pos):
                        pygame.quit()
                        sys.exit()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if self.env._current_player == 1:
                    if event.type == pygame.MOUSEBUTTONDOWN: # Detect mouse click events
                        pos = self.get_clicked_pos(event.pos) # Get the position of the user's click
                        row, col = pos
                        if self.env.valid_area[row][col] and self.env._board[row][col] == 0: # Check if the clicked position is within a valid area and that the position is empty (unoccupied)
                            action = self.env.valid_positions.index(pos)
                            print(f"Player 1 action: {action}")
                            self.env.step(action)
                else:
                    self.ai_move()

                self.draw_board()
                clock.tick(30)

                if self.env._episode_ended:
                    winner = 1 if self.env._current_player == 2 else 2
                    if self.show_result(winner):
                        logging.info("Starting a new game...")
                        self.env.reset()
                    else:
                        pygame.quit()
                        sys.exit()


if __name__ == "__main__":
    game = GameGUI()
    game.run()