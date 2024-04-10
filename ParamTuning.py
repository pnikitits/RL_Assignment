import wandb
import yaml
import gymnasium as gym
import text_flappy_bird_gym
import numpy as np
from SarsaLambdaAgent import SarsaLambdaAgent , train_sarsa



def run_sweeping():
    a = wandb.init()

    # with open("param_sweep/bayes_sweep.yaml") as file:
    #     config = yaml.load(file, Loader=yaml.FullLoader)

    settings = {
        "alpha": wandb.config.alpha,
        "gamma": wandb.config.gamma,
        "lam": wandb.config.lam,
        "epsilon": wandb.config.epsilon,
        "epsilon_decay": 0.999,
        "epsilon_min": wandb.config.epsilon_min
    }


    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    agent = SarsaLambdaAgent(env.action_space , settings)

    episodes = 10000
    agent , rewards , log_epsilons = train_sarsa(env, agent, episodes)

    


if __name__ == "__main__":
    with open("param_sweep/bayes_sweep.yaml") as file:
        sweep_configuration = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="flappy bird")
    wandb.agent(sweep_id, function=run_sweeping, count = 100)