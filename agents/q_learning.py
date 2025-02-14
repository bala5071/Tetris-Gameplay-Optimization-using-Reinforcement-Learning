import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def preprocess_state(self, state):
        # Convert dictionary observation to flat array
        board = state['board'].flatten()
        holes = state['holes']
        bumpiness = state['bumpiness']
        height = state['height']
        return np.concatenate([board, holes, bumpiness, height])

    def get_state_key(self, state):
        return tuple(state.astype(int))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q * (1 - done) - current_q)
        self.q_table[state_key][action] = new_q

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
