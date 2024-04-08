import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import os, sys
import time
import text_flappy_bird_gym

np.random.seed(11)


class MonteCarloAgent:
    def __init__(self, action_space , epsilon_decay = 0.8 , alpha = 0.05):
        self.action_value_estimates = {}  # {(state, action): [returns_sum, visit_count]}
        self.policy = {}  # {state: action}
        self.action_space = action_space

        self.epsilon = 0.1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.alpha = alpha
    
    def choose_action(self, state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if np.random.rand() < self.epsilon or state not in self.policy:
            return self.action_space.sample()
        return self.policy[state]
    
    def update(self, episode, gamma=0.989):
        # gamma: discount factor: 0 is short-sighted, 1 is far-sighted
        total_return = 0
        for state, action, reward in reversed(episode):
            # Calculate the total return
            total_return = gamma * total_return + reward
            # Check if this state-action pair has been visited before
            if (state, action) not in self.action_value_estimates:
                self.action_value_estimates[(state, action)] = [0, 0]
            # Get the previous sum of returns and the visit count for this state-action pair
            returns_sum, visit_count = self.action_value_estimates[(state, action)]
            # Calculate the new value
            new_value = returns_sum + self.alpha * (total_return - returns_sum)
            # Update the action-value estimate with the new value and incremented visit count
            self.action_value_estimates[(state, action)] = [new_value, visit_count + 1]
            
            # Now, we perform policy improvement:
            # Find the best action by looking at the updated Q-values
            best_action = None
            best_value = float('-inf')
            for a in range(self.action_space.n):
                q_value = self.action_value_estimates.get((state, a), [0, 0])[0]
                if q_value > best_value:
                    best_value = q_value
                    best_action = a
            # Update the policy to choose the best action in this state
            self.policy[state] = best_action






if __name__ == "__main__":

    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    agent = MonteCarloAgent(env.action_space)

    episodes = 10000
    rewards = []

    for episode in tqdm(range(episodes)):
        state = env.reset()[0]
        
        episode_memory = []
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state , reward , done , _ , _ = env.step(action) # ((1, 12), 1, True, False, {'score': 0, 'player': [6, 15], 'distance': 12.041594578792296})
            episode_memory.append((state , action , reward))
            state = next_state
            total_reward += reward

        

        agent.update(episode_memory)
        rewards.append(total_reward)

    if True:
        # plot the rewards
        env.close()
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        plt.show()

    else:
        # play the game
        obs = env.reset()
        state = env.reset()[0]

        while True:
            action = agent.choose_action(state)
            next_state , reward , done , _ , _ = env.step(action)
            state = next_state

            os.system("clear")
            sys.stdout.write(env.render())
            time.sleep(0.2)

            if done:
                break

        env.close()