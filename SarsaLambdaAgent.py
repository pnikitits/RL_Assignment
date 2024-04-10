from tqdm import tqdm
import numpy as np
import wandb



class SarsaLambdaAgent:
    def __init__(self, action_space, agent_settings):
        self.action_space = action_space

        self.alpha = agent_settings["alpha"]
        self.gamma = agent_settings["gamma"]
        self.lam = agent_settings["lam"]
        self.epsilon = agent_settings["epsilon"]
        self.epsilon_decay = agent_settings["epsilon_decay"]
        self.epsilon_min = agent_settings["epsilon_min"]

        self.Q = {}
        self.E = {}


    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            q_values = [self.get_Q(state, a) for a in range(self.action_space.n)]
            return np.argmax(q_values)


    def get_Q(self, state, action):
        return self.Q.get((state, action), 0.0)


    def update(self, state, action, reward, next_state, next_action, done):
        sa_pair = (state, action)
        
        q_current = self.get_Q(state, action)
        q_next = self.get_Q(next_state, next_action) if not done else 0
        
        delta = reward + self.gamma * q_next - q_current
        
        self.E[sa_pair] = self.E.get(sa_pair, 0.0) + 1

        for sa in self.E:
            self.Q[sa] = self.Q.get(sa, 0.0) + self.alpha * delta * self.E[sa]
            self.E[sa] *= self.gamma * self.lam

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)





def train_sarsa(env, agent, episodes):
    np.random.seed(11)

    rewards = []
    log_epsilons = []

    for _ in tqdm(range(episodes)):
        state = env.reset()[0]
        action = agent.choose_action(state)
        done = False

        total_reward = 0

        while not done:
            next_state , reward , done , _ , _ = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_reward += reward

        wandb.log({"episode reward": np.mean(total_reward)})
        rewards.append(total_reward)
        log_epsilons.append(agent.epsilon)

        if total_reward > 200:
            break

    wandb.log({"avg_reward": np.mean(rewards)})
    env.close()
    return agent , rewards , log_epsilons