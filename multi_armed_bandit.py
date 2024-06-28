import numpy as np

class MultiArmedBandit:
    def __init__(self, num_bandits):
        self.num_bandits = num_bandits
        self.q_true = np.random.normal(0, 1, self.num_bandits)  # True mean values (unknown to agent)
        self.q_estimate = np.zeros(self.num_bandits)  # Estimated values
        self.action_count = np.zeros(self.num_bandits)  # Number of times each action was chosen

    def select_action(self, epsilon):
        if np.random.rand() < epsilon:
            # Explore: choose a random bandit
            action = np.random.randint(self.num_bandits)
        else:
            # Exploit: choose the bandit with the highest estimated value
            action = np.argmax(self.q_estimate)
        
        return action

    def take_action(self, action):
        # Generate reward from a normal distribution with mean q_true[action] and variance 1
        reward = np.random.normal(self.q_true[action], 1)
        self.action_count[action] += 1
        self.q_estimate[action] += (reward - self.q_estimate[action]) / self.action_count[action]
        return reward

def main():
    num_bandits = 5
    num_steps = 1000
    epsilon = 0.1

    bandit = MultiArmedBandit(num_bandits)

    rewards = []

    for _ in range(num_steps):
        action = bandit.select_action(epsilon)
        reward = bandit.take_action(action)
        rewards.append(reward)

    print("Estimated values:", bandit.q_estimate)
    print("Total reward:", np.sum(rewards))

if __name__ == "__main__":
    main()
