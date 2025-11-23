import numpy as np

class PolicyGradientAgent:
    """
    A simple REINFORCE (Monte Carlo Policy Gradient) Agent.
    """
    def __init__(self, action_space, input_dim, learning_rate=0.01):
        self.action_space = action_space
        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Simple linear policy: W * state -> softmax -> probs
        # Weights shape: (input_dim, num_actions)
        self.W = np.random.randn(input_dim, action_space.n) * 0.01

        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_obs = []
        self.episode_actions = []

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, observation):
        # Observation is expected to be flattened
        # If it's zero or None, use zeros
        if observation is None:
            observation = np.zeros(self.input_dim)

        logits = np.dot(observation, self.W)
        probs = self.softmax(logits)

        action = np.random.choice(range(self.action_space.n), p=probs)

        # Store for update
        self.episode_obs.append(observation)
        self.episode_actions.append(action)
        self.episode_log_probs.append(probs[action]) # simplified, not actual log prob yet

        return action

    def store_reward(self, reward):
        self.episode_rewards.append(reward)

    def update(self):
        # Monte Carlo Policy Gradient Update
        # Return G_t
        discounted_rewards = np.zeros_like(self.episode_rewards, dtype=np.float32)
        cumulative = 0.0
        gamma = 0.99

        for t in reversed(range(len(self.episode_rewards))):
            cumulative = cumulative * gamma + self.episode_rewards[t]
            discounted_rewards[t] = cumulative

        # Normalize rewards for stability
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)

        # Update weights
        for t in range(len(self.episode_rewards)):
            obs = self.episode_obs[t]
            action = self.episode_actions[t]
            G_t = discounted_rewards[t]

            # Gradient of log_prob with respect to logits: (1 - p) if action chosen, -p otherwise
            logits = np.dot(obs, self.W)
            probs = self.softmax(logits)

            d_softmax = probs.copy()
            d_softmax[action] -= 1
            d_softmax = -d_softmax # This is effectively gradient of -log(p) ?
            # Actually grad J = sum G_t * grad log pi
            # grad log pi = (1[a] - pi) * x

            grad_log_pi = np.outer(obs, (np.eye(self.action_space.n)[action] - probs))

            self.W += self.learning_rate * G_t * grad_log_pi

        # Reset episode buffers
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_obs = []
        self.episode_actions = []

# Alias for compatibility if needed, but we will update usages
SimpleRLAgent = PolicyGradientAgent
