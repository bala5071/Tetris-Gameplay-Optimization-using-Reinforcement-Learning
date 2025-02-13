import numpy as np

class QLearningAgent:
    def __init__(self, action_dim):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.action_dim = action_dim
        
    def get_state_key(self, state):
        # Discretize state (example: hash of simplified features)
        return hash(self.extract_features(state).tostring())
    
    def extract_features(self, state):
        # Implement feature extraction (heuristics)
        return np.array([
            self.calculate_holes(state),
            self.calculate_bumpiness(state),
            self.calculate_height(state)
        ])
    
    def act(self, state):
        state_key = self.get_state_key(state)
        if np.random.rand() < self.epsilon or state_key not in self.q_table:
            return np.random.randint(self.action_dim)
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
            
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table.get(next_state_key, np.zeros(self.action_dim)))
            
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])
    
    # Implement heuristic calculations
    def calculate_holes(self, state):
        return np.sum(state == 0)
    
    def calculate_bumpiness(self, state):
        heights = np.argmax(state == 1, axis=0)
        return np.sum(np.abs(np.diff(heights)))
    
    def calculate_height(self, state):
        return np.max(np.argmax(state == 1, axis=0))