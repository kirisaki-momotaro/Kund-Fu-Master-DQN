import torch
import gym
import numpy as np
from PIL import Image
import os

from breakout import *

print(torch.cuda.is_available())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

environment = DQNBreakout(device=device , render_mode='human')

state = environment.reset()
num_actions = environment.action_space.n
print(f"Number of actions: {num_actions}") #print number of availabel actions
for _ in range(100):
    action = environment.action_space.sample()
    state, reward,done ,info = environment.step(action)
    print(state.shape)