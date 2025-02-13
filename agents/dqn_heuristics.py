from .dqn import DQNAgent
import numpy as np

class DQNHeuristicsAgent(DQNAgent):
    def __init__(self, num_features, num_actions):
        super().__init__((num_features,), num_actions)
        
    def extract_features(self, state):
        # Implement enhanced feature extraction
        return np.array([
            self.calculate_holes(state),
            self.calculate_bumpiness(state),
            self.calculate_height(state),
            self.calculate_lines_cleared(state),
            self.calculate_well_depth(state)
        ])
    
    def act(self, state):
        features = self.extract_features(state)
        return super().act(features)
    
    def update(self, state, action, reward, next_state, done):
        features = self.extract_features(state)
        next_features = self.extract_features(next_state)
        super().update(features, action, reward, next_features, done)
    
    # Additional heuristic methods
    def calculate_lines_cleared(self, state):
        return np.sum(np.all(state == 1, axis=1))
    
    def calculate_well_depth(self, state):
        heights = np.argmax(state == 1, axis=0)
        return np.max(heights) - np.min(heights)