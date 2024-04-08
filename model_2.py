import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import os, sys
import time
import text_flappy_bird_gym
import pandas as pd
import itertools

np.random.seed(11)


class SarsaLambdaAgent:
    def __init__(self, action_space, alpha=0.012, gamma=0.85, lam=0.9, epsilon=0.3, epsilon_decay=0.999995, epsilon_min=0.0):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam # Lambda for eligibility traces (0 for MC, 1 for TD(0))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = {}  # Initialize Q table with a default dictionary to handle states
        self.E = {}  # Eligibility traces

    def choose_action(self, state):
        # Implement ε-greedy policy
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = [self.get_Q(state, a) for a in range(self.action_space.n)]
            return np.argmax(q_values)

    def get_Q(self, state, action):
        # Using a tuple (state, action) as a key for the Q-table
        return self.Q.get((state, action), 0.0)

    def update(self, state, action, reward, next_state, next_action, done):
        # Update rule for Q and eligibility trace
        sa_pair = (state, action)
        next_sa_pair = (next_state, next_action)
        
        # Get the current Q value
        q_current = self.get_Q(state, action)
        q_next = self.get_Q(next_state, next_action) if not done else 0
        
        # Calculate delta
        delta = reward + self.gamma * q_next - q_current
        
        # Update the eligibility trace for the state-action pair
        self.E[sa_pair] = self.E.get(sa_pair, 0.0) + 1

        for sa in self.E:
            self.Q[sa] = self.Q.get(sa, 0.0) + self.alpha * delta * self.E[sa]
            self.E[sa] *= self.gamma * self.lam  # Decay the trace value

        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)



def plot_policy(agent, state_space):
    grid = np.zeros(state_space)
    for state in itertools.product(range(state_space[0]), range(state_space[1])):
        action = agent.choose_action(state)
        grid[state] = action

    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.title('Policy Visualization')
    plt.colorbar()
    plt.show()


def make_the_rewards_plot(choice):
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    agent = SarsaLambdaAgent(env.action_space , lam=choice)

    episodes = 30000
    rewards = []

    log_epsilon = []
    action_counts = {a: 0 for a in range(2)}

    for _ in tqdm(range(episodes)):
        state = env.reset()[0]
        action = agent.choose_action(state)
        done = False

        total_reward = 0

        while not done:
            next_state, _, done, _ , _ = env.step(action)

            y_diff = next_state[1]
            reward = -abs(y_diff) / 15

            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_reward += reward
            action_counts[action] += 1

        rewards.append(total_reward)
        log_epsilon.append(agent.epsilon)

    env.close()
    return rewards


from multiprocessing import Pool

if __name__ == "__main__":
    values_to_test = np.linspace(0, 1, 11)
    
    # Setup process pool and execute
    with Pool() as pool:
        results = pool.map(make_the_rewards_plot, values_to_test)
    
    # Plot the rewards in subplots
    fig, axs = plt.subplots(1, len(values_to_test), figsize=(20, 5))
    for i, rewards in enumerate(results):
        smoothed_rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')
        axs[i].plot(rewards)
        axs[i].plot(smoothed_rewards)
        axs[i].set_title(f'Lambda = {values_to_test[i]:.1f}')
        axs[i].set_xlabel("Episode")
        axs[i].set_ylabel("Reward")
        axs[i].axhline(0, color='black', lw=1)
    plt.show()

    

    


if False:

    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    agent = SarsaLambdaAgent(env.action_space)

    episodes = 20000
    rewards = []

    log_epsilon = []
    action_counts = {a: 0 for a in range(2)}

    for episode in tqdm(range(episodes)):
        state = env.reset()[0]
        action = agent.choose_action(state)
        done = False

        total_reward = 0

        while not done:
            next_state, reward, done, _ , _ = env.step(action)


            y_diff = next_state[1]
            # x_diff = next_state[0]

            reward = -abs(y_diff) / 15


            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_reward += reward

            action_counts[action] += 1

        rewards.append(total_reward)
        log_epsilon.append(agent.epsilon)

    if True:
        env.close()

        # Plot the rewards
        smoothed_rewards = pd.Series(rewards).rolling(window=50).mean()
        plt.plot(rewards)
        plt.plot(smoothed_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        # plt.ylim(-1, 30)
        plt.axhline(0, color='black', lw=1)
        plt.show()

        # Plotting epsilon decay
        plt.plot(log_epsilon)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay")
        plt.show()

        # Plotting action distribution
        plt.bar(range(len(action_counts)), action_counts.values())
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.title('Action Distribution')
        plt.show()
        
        # Plot the policy
        plot_policy(agent, (15, 20))

    else:
        # play the game
        obs = env.reset()
        state = env.reset()[0]

        while True:
            action = agent.choose_action(state)
            next_state , _ , done , _ , _ = env.step(action)
            state = next_state

            os.system("clear")
            sys.stdout.write(env.render())
            time.sleep(0.2)

            if done:
                break

        env.close()
        

