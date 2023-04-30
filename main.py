import gymnasium as gym
from gymnasium.wrappers import RescaleAction, NormalizeObservation
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.distributions import Normal

import torch

NUM_EPS = 1000
LR = 1e-3
GAMMA = 0.5
# EPSILON = 0.1

# initialize model
# per documentatio, humanoid task is
# scene: (376,) float32
# action: (17,) float64

# so our network should take (376,) as input
# and have ((17,), (1,) as output --- first half
# used to seed the mean, second half used to seed
# the standard distribution

class RLModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(376, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, 17)
        self.std = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        net = self.relu(self.l1(x))
        net = self.relu(self.l2(net))
        net = self.relu(self.l3(net))

        mean = self.mean(net)
        std = self.std(net) # we treat std, per usual, as unactivated log

        distribution = Normal(mean, torch.exp(std)) # so actual std is e^that value

        return distribution

model = RLModel()
optim = AdamW(model.parameters(), lr=LR)

env = gym.make("HumanoidStandup-v4", render_mode="human")
env = NormalizeObservation(RescaleAction(env, min_action=0, max_action=1))

for _ in range(NUM_EPS):
    terminated = False
    observation, info = env.reset()

    # store our actions + corresponding rewards
    action_probabilities = []
    rewards = [] 

    # play the video game
    while not terminated:
        # create and sample action from distribution
        dist = model(torch.tensor(observation).float())
        action = dist.sample()

        # take action
        observation, reward, terminated, truncated, info = env.step(F.sigmoid(action).cpu().numpy())

        # if we are truncated, we still consider our episode done
        terminated = terminated or truncated

        # store!
        action_probabilities.append(dist.log_prob(action))
        rewards.append(reward)

    print(f"Mean reward this episode: {torch.mean(torch.tensor(rewards)).item()}")

    # now, actually perform optimization
    # calculate discounted future reward per timestamp
    # we do this backwards, coming from the back and
    # having a running talley of the future distcount

    discounted_future_reward = 0
    cumulative_rewards = []

    for i in reversed(rewards):
        # get the current reward
        cumulative_reward = (i + discounted_future_reward)
        cumulative_rewards.append(cumulative_reward)

        # add the current reward to the future 
        discounted_future_reward = GAMMA*cumulative_reward

    # reverse the rewards list to be in the right order
    cumulative_rewards = list(reversed(cumulative_rewards))

    # and now, train the model
    for logA, R in zip(action_probabilities, cumulative_rewards):
        (-torch.mean(logA*R)).backward()

    optim.step()
    optim.zero_grad()

env.close()
