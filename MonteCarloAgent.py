from tqdm import tqdm
import numpy as np
np.random.seed(11)



class MonteCarloAgent:
    def __init__(self, action_space , agent_settings):
        self.action_space = action_space

        self.epsilon = agent_settings["epsilon"]
        self.epsilon_decay = agent_settings["epsilon_decay"]
        self.epsilon_min = agent_settings["epsilon_min"]
        self.gamma = agent_settings["gamma"]
        self.alpha = agent_settings["alpha"]

        self.Q = {}
        self.policy = {}
    

    def choose_action(self, state):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if np.random.rand() < self.epsilon or state not in self.policy:
            return self.action_space.sample()
        return self.policy[state]
    

    def update(self, episode):
        total_return = 0

        for state, action, reward in reversed(episode):
            total_return = self.gamma * total_return + reward

            if (state, action) not in self.Q:
                self.Q[(state, action)] = [0, 0]

            returns_sum, visit_count = self.Q[(state, action)]
            new_value = returns_sum + self.alpha * (total_return - returns_sum)
            self.Q[(state, action)] = [new_value, visit_count + 1]
            
            best_action = None
            best_value = float('-inf')

            for a in range(self.action_space.n):
                q_value = self.Q.get((state, a), [0, 0])[0]

                if q_value > best_value:
                    best_value = q_value
                    best_action = a
                    
            self.policy[state] = best_action





def train_monte_carlo(env , agent , episodes):
    rewards = []

    for _ in tqdm(range(episodes)):
        state = env.reset()[0]
        
        episode_memory = []
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state , reward , done , _ , _ = env.step(action)
            episode_memory.append((state , action , reward))
            state = next_state
            total_reward += reward

        agent.update(episode_memory)
        rewards.append(total_reward)

    env.close()
    l500_mean = np.mean(rewards[-500:])
    return agent , rewards , l500_mean