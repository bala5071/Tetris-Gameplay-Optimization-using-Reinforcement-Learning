import gymnasium as gym
from envs.tetris_env import TetrisEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.dqn_heuristics import DQNHeuristicsAgent
import numpy as np

def train(agent_type, episodes=1000):
    env = TetrisEnv()
    if agent_type == "q_learning":
        agent = QLearningAgent(env.action_space.n)
    elif agent_type == "dqn":
        agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    elif agent_type == "dqn_heuristics":
        agent = DQNHeuristicsAgent(num_features=5, num_actions=env.action_space.n)
    else:
        raise ValueError("Invalid agent type")

    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            if isinstance(agent, DQNAgent):
                agent.update_target()
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {episode}, Reward: {total_reward}")
    return rewards

if __name__ == "__main__":
    q_learning_rewards = train("q_learning", 1)
    dqn_rewards = train("dqn", 1)
    dqn_heuristics_rewards = train("dqn_heuristics", 1)
    np.savez("results.npz", 
             q_learning=q_learning_rewards,
             dqn=dqn_rewards,
             dqn_heuristics=dqn_heuristics_rewards)