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
    best_reward = float('-inf')
    no_improvement_count = 0
    
    for episode in tqdm(range(episodes), desc=f"Training {agent_type}"):
        state, _ = env.reset()
        state = agent.preprocess_state(state)
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, base_reward, done, _, _ = env.step(action)
            next_state = agent.preprocess_state(next_state)
            
            # Calculate composite reward
            reward = calculate_composite_reward(
                env, 
                base_reward, 
                done, 
                step_count
            )
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if isinstance(agent, (DQNAgent, DQNHeuristicsAgent)):
                agent.update_target()
        
        # Track best performance and implement early stopping
        if episode_reward > best_reward:
            best_reward = episode_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Best: {best_reward:.2f}")
        
        # Early stopping if no improvement for many episodes
        if episode>800 and no_improvement_count > 300:
            print(f"Early stopping at episode {episode} due to no improvement")
            break
    
    return rewards

def calculate_composite_reward(env, base_reward, done, step_count):
    reward = 0
    
    # Increase base rewards for successful actions
    reward += base_reward * 2
    
    # Progressive line clearing rewards
    if hasattr(env, 'cleared_lines'):
        lines_cleared = env.cleared_lines
        reward += pow(2, lines_cleared) * 100  # Exponential scaling
    
    # Survival reward based on step count
    reward += min(step_count * 0.5, 100)  # Cap at 100 to prevent exploitation
    
    # Reduce structural penalties
    holes = env.count_holes()
    bumpiness, height = env.get_bumpiness_and_height()
    reward -= holes * 2
    reward -= bumpiness * 0.25
    reward -= (height/env.height) * 1
    
    # Add positive rewards for good structure
    if height < env.height/2:
        reward += 50
    
    # Game over penalty
    if done:
        reward -= 30 * (1 - min(step_count/500, 1))  # Lower penalty for longer survival
    
    return reward



if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    episodes = 1000
    rewards_dict = {}
    
    # Train each agent with increased batch size and learning rate
    for agent_type in ["q_learning", "dqn", "dqn_heuristics"]:
        print(f"\nTraining {agent_type}")
        rewards = train(agent_type, episodes)
        rewards_dict[agent_type] = rewards
    
    # Save results
    np.savez("results.npz", **rewards_dict)
    plot_rewards(rewards_dict)
