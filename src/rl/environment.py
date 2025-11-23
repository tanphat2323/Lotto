import gymnasium as gym
from gymnasium import spaces
import numpy as np

class LottoEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    def __init__(self, historical_data, lookback=50, prize_table=None):
        super(LottoEnv, self).__init__()

        self.data = historical_data
        self.lookback = lookback
        self.current_step = lookback

        # Action space:
        # 0: Skip
        # 1: Buy 1 ticket (Random/Best)
        # 2: Buy Small Wheel (Budget 5)
        # 3: Buy Large Wheel (Budget 10)
        self.action_space = spaces.Discrete(4)

        # Observation space:
        # For now, just a dummy vector of shape (lookback, features)
        # We will flatten it for the simple agent
        # Or just use the last draw features + wallet info
        self.observation_space = spaces.Box(low=0, high=1, shape=(lookback * 35,), dtype=np.float32)

        self.balance = 0
        self.cost_per_ticket = 10000 # 10k VND

        # Default prize table (from PDF)
        if prize_table is None:
            self.prize_table = {
                5: 12000000000, # Jackpot min (ignoring rolldown for now)
                4: 500000,      # Match 4
                3: 50000,       # Match 3
                2: 5000         # Match 2 (assuming)
                # Note: Correct values should be checked from PDF.
                # PDF: 3 numbers => 30,000; 4 numbers => 3,000,000 ???
                # I will use placeholders and user can update.
            }
            # Actually let's try to be accurate if possible.
            # Match 3: 30,000
            # Match 4: 3,000,000
            # Match 5: Jackpot
            self.prize_table = {
                5: 12000000000,
                4: 3000000,
                3: 30000,
                2: 0 # Usually match 2 is nothing in 5/35? Need to verify.
            }
        else:
            self.prize_table = prize_table

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        self.balance = 0
        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        # Return flattened history of last 'lookback' draws
        # Just return zeros for this skeleton if data not fully linked
        return np.zeros(self.lookback * 35, dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0

        # Simulate results based on next draw in data
        if self.current_step >= len(self.data) - 1:
            done = True
            return self._get_obs(), 0, done, False, {}

        # Real result
        # For RL training, we need the actual next result to calculate reward
        # But in live mode we don't know it.
        # This Env is for *Training* the RL agent on historical data.

        # For skeleton training, we assume we are at 'current_step'.
        # The result of this step is at index 'current_step'.
        # We need to compute if we won based on a simulated action.
        # Since the action is just "Buy 1" without specifying numbers (RL Agent limitation in this skeleton),
        # we have to assume a heuristic for winnings: e.g. "Buy 1" means "Buy the top 1 predicted by Model".
        # But the Env doesn't have the model.
        # So for RL training purposes, we will assign a random small probability of winning or use a placeholder.
        # BETTER: The RL agent learns *when* to play. The outcome depends on luck (random).
        # We can simulate the outcome based on empirical win rates.
        # Probability of matching 3 is ~1/100, 4 is ~1/3000, 5 is ~1/300k.

        cost = 0
        winnings = 0

        if action == 0: # Skip
            pass
        elif action == 1: # Buy 1
            cost = self.cost_per_ticket
            if np.random.random() < 0.01: winnings = self.prize_table[3] # Simulate win
        elif action == 2: # Wheel 5
            cost = 5 * self.cost_per_ticket
            if np.random.random() < 0.05: winnings = self.prize_table[3]
        elif action == 3: # Wheel 10
            cost = 10 * self.cost_per_ticket
            if np.random.random() < 0.10: winnings = self.prize_table[3]

        reward = winnings - cost
        self.balance += reward

        self.current_step += 1

        observation = self._get_obs()

        return observation, reward, done, False, {"balance": self.balance}

if __name__ == "__main__":
    env = LottoEnv(historical_data=[1]*100) # Dummy data
    obs, _ = env.reset()
    print("Env Reset. Obs shape:", obs.shape)

    action = env.action_space.sample()
    obs, reward, done, _, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Balance: {info['balance']}")
