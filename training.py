import gymnasium as gym
from envs.tetris_env import TetrisEnv
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.dqn_heuristics import DQNHeuristicsAgent
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_rewards(rewards_dict):
    plt.figure(figsize=(10, 5))
    for agent_name, rewards in rewards_dict.items():
        plt.plot(rewards, label=agent_name)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards by Agent')
    plt.legend()
    plt.savefig('training_rewards.png')
    plt.close()

def train(agent_type, episodes=1000):
    env = TetrisEnv()
    state_size = env.height * env.width + 3
    action_size = env.action_space.n
    
    if agent_type == "q_learning":
        agent = QLearningAgent(state_size, action_size)
    elif agent_type == "dqn":
        agent = DQNAgent(state_size, action_size)
    elif agent_type == "dqn_heuristics":
        agent = DQNHeuristicsAgent(state_size, action_size)
    else:
        raise ValueError("Invalid agent type")

    rewards = []
    for episode in tqdm(range(episodes), desc=f"Training {agent_type}"):
        state, _ = env.reset()
        state = agent.preprocess_state(state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = agent.preprocess_state(next_state)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if isinstance(agent, (DQNAgent, DQNHeuristicsAgent)):
                agent.update_target()
        
        rewards.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}")
    
    return rewards

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    episodes = 100
    rewards_dict = {}
    
    # Train each agent
    for agent_type in ["q_learning", "dqn", "dqn_heuristics"]:
        rewards = train(agent_type, episodes)
        rewards_dict[agent_type] = rewards
    
    # Save results
    np.savez("results.npz", **rewards_dict)
    plot_rewards(rewards_dict)
