import gymnasium as gym
import numpy as np
import cv2

class TetrisEnv(gym.Env):
    pieces = [
        [[1, 1],
         [1, 1]],  # Square
        [[0, 2, 0],
         [2, 2, 2]],  # T
        [[0, 3, 3],
         [3, 3, 0]],  # S
        [[4, 4, 0],
         [0, 4, 4]],  # Z
        [[5, 5, 5, 5]],  # I
        [[0, 0, 6],
         [6, 6, 6]],  # L
        [[7, 0, 0],
         [7, 7, 7]]  # J
    ]
    
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    def __init__(self):
        super().__init__()
        self.width = 10
        self.height = 20
        self.action_space = gym.spaces.Discrete(self.width * 4)
        self.observation_space = gym.spaces.Dict({
            'board': gym.spaces.Box(low=0, high=7, shape=(self.height, self.width), dtype=np.int32),
            'holes': gym.spaces.Box(low=0, high=self.width*self.height, shape=(1,), dtype=np.int32),
            'bumpiness': gym.spaces.Box(low=0, high=self.width*self.height, shape=(1,), dtype=np.int32),
            'height': gym.spaces.Box(low=0, high=self.height, shape=(1,), dtype=np.int32)
        })
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.height, self.width), dtype=np.int32)
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.piece = None
        self.current_pos = {"x": 0, "y": 0}
        self.gameover = False
        self.spawn_piece()
        return self.get_state(), {}

    def spawn_piece(self):
        self.piece = np.array(self.pieces[np.random.randint(len(self.pieces))])
        # Ensure piece spawns in the middle of the board
        self.current_pos = {
            "x": (self.width // 2) - (len(self.piece[0]) // 2),
            "y": 0
        }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def step(self, action):
        if self.gameover:
            return self.get_state(), 0, True, False, {}

        # Convert action to x position and rotation
        x = action // 4
        num_rotations = action % 4

        # Copy current piece and position
        piece = self.piece.copy()
        pos = {"x": x, "y": 0}

        # Rotate piece
        for _ in range(num_rotations):
            piece = self.rotate_piece(piece)

        # Validate position
        if x + len(piece[0]) > self.width:
            x = self.width - len(piece[0])
        pos["x"] = max(0, min(x, self.width - len(piece[0])))

        # Drop piece
        while not self.check_collision(piece, pos):
            pos["y"] += 1

        # Move back up one step
        pos["y"] -= 1

        # Check if game is over
        if pos["y"] < 0:
            self.gameover = True
            return self.get_state(), -10, True, False, {}

        # Place piece
        self.board = self.place_piece(piece, pos)

        # Clear lines and calculate reward
        lines_cleared = self.clear_lines()
        reward = self.calculate_reward(lines_cleared)

        # Spawn new piece
        self.spawn_piece()

        return self.get_state(), reward, self.gameover, False, {}

    def check_collision(self, piece, pos):
        future_y = pos["y"]
        future_x = pos["x"]

        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] == 0:
                    continue
                if future_y + y >= self.height:
                    return True
                if future_x + x < 0 or future_x + x >= self.width:
                    return True
                if future_y + y >= 0 and self.board[future_y + y][future_x + x] != 0:
                    return True
        return False

    def place_piece(self, piece, pos):
        board = self.board.copy()
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] != 0:
                    if 0 <= pos["y"] + y < self.height and 0 <= pos["x"] + x < self.width:
                        board[pos["y"] + y][pos["x"] + x] = piece[y][x]
        return board

    def rotate_piece(self, piece):
        return np.rot90(piece, -1)

    def clear_lines(self):
        lines = 0
        for i in range(self.height):
            if np.all(self.board[i] != 0):
                self.board = np.vstack((np.zeros((1, self.width)), self.board[:i], self.board[i+1:]))
                lines += 1
        return lines

    def calculate_reward(self, lines_cleared):
        if self.gameover:
            return -10
        return lines_cleared * 10

    def get_state(self):
        holes = self.count_holes()
        bumpiness, height = self.get_bumpiness_and_height()
        return {
            'board': self.board.copy(),
            'holes': np.array([holes]),
            'bumpiness': np.array([bumpiness]),
            'height': np.array([height])
        }

    def count_holes(self):
        holes = 0
        for col in range(self.width):
            block_found = False
            for row in range(self.height):
                if self.board[row][col] != 0:
                    block_found = True
                elif block_found and self.board[row][col] == 0:
                    holes += 1
        return holes

    def get_bumpiness_and_height(self):
        heights = []
        for col in range(self.width):
            for row in range(self.height):
                if self.board[row][col] != 0:
                    heights.append(self.height - row)
                    break
            if len(heights) < col + 1:
                heights.append(0)
        
        total_height = sum(heights)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        return bumpiness, total_height

    def render(self):
        if not hasattr(self, 'screen'):
            cv2.namedWindow('Tetris', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tetris', 300, 600)

        image = np.zeros((self.height * 30, self.width * 30, 3), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                color = self.piece_colors[self.board[y][x]]
                cv2.rectangle(image,
                            (x * 30, y * 30),
                            ((x + 1) * 30, (y + 1) * 30),
                            color,
                            -1)
                cv2.rectangle(image,
                            (x * 30, y * 30),
                            ((x + 1) * 30, (y + 1) * 30),
                            (128, 128, 128),
                            1)

        cv2.putText(image,
                    f'Score: {self.score}',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        cv2.imshow('Tetris', image)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
