from MonteCarloAgent import MonteCarloAgent , train_monte_carlo
from SarsaLambdaAgent import SarsaLambdaAgent , train_sarsa

import gymnasium as gym
import text_flappy_bird_gym

import matplotlib.pyplot as plt
import os, sys
import time



def play_game(env, agent , fast=False):
    state = env.reset()[0]

    while True:
        action = agent.choose_action(state)
        next_state , _ , done , _ , _ = env.step(action)
        state = next_state

        os.system("clear")
        sys.stdout.write(env.render())

        if not fast:
            time.sleep(0.2)

        if done:
            break

    env.close()




def run_sarsa(play , fast):
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)

    SarsaSettings = {
        "alpha": 0.12,
        "gamma": 0.9999,
        "lam": 0.9,
        "epsilon": 0.1,
        "epsilon_decay": 0.99999,
        "epsilon_min": 0.0001
    }

    agent = SarsaLambdaAgent(env.action_space , SarsaSettings)

    episodes = 10000
    agent , rewards , log_epsilons = train_sarsa(env, agent, episodes)

    if play:
        play_game(env, agent , fast)
    else:
        fig, axs = plt.subplots(2)
        fig.suptitle("Sarsa Lambda Agent")

        axs[0].plot(rewards)
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")
        axs[0].set_title("Reward vs Episode")

        axs[1].plot(log_epsilons)
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Epsilon")
        axs[1].set_title("Epsilon vs Episode")

        plt.show()




def run_monte_carlo(play , fast):
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)

    MonteCarloSettings = {
        "epsilon": 0.1,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
        "gamma": 0.99,
        "alpha": 0.1
    }

    agent = MonteCarloAgent(env.action_space , MonteCarloSettings)

    episodes = 50000
    agent , rewards , l500_mean = train_monte_carlo(env, agent, episodes)

    if play:
        play_game(env, agent , fast)
    else:
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        plt.show()




if __name__ == "__main__":
    run_sarsa(play=True , fast=False)
    # run_monte_carlo(play=True , fast=False)