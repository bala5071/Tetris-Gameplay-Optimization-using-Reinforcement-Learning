import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

data = np.load("results.npz")
plt.figure(figsize=(12, 6))

for name in data.files:
    rewards = data[name]
    smoothed = smooth(rewards)
    plt.plot(smoothed, label=name)

plt.title("Tetris Performance Comparison")
plt.xlabel("Episode")
plt.ylabel("Smoothed Reward")
plt.legend()
plt.grid()
plt.savefig("results.png")
plt.show()