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




def run_sarsa(play = False):
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)

    SarsaSettings = {
        "alpha": 0.12,
        "gamma": 0.99,
        "lam": 0.9,
        "epsilon": 0.1,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01
    }

    agent = SarsaLambdaAgent(env.action_space , SarsaSettings)

    episodes = 10000
    agent , rewards , l500_mean = train_sarsa(env, agent, episodes)

    if play:
        play_game(env, agent)
    else:
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        plt.show()




def run_monte_carlo(play = False):
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
        play_game(env, agent)
    else:
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward vs Episode")
        plt.show()




if __name__ == "__main__":
    #run_sarsa(play=False)
    run_monte_carlo(play=True)