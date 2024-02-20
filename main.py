import torch
import gym
import numpy as np
from PIL import Image
from agent import Agent
import os

from model import AtariNet

from kung_fu_master import *

print(torch.cuda.is_available())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#environment = DQNKungFuMaster(device=device , render_mode='human')
environment = DQNKungFuMaster(device=device)

model = AtariNet(nb_actions= 14)

model.to(device)

model.load_the_model()

agent = Agent(model=model,device=device,epsilon=1.0,
              nb_warmup=500, nb_actions=14, learning_rate=0.001,memory_capacity=50000,
                batch_size=256)

agent.train(env=environment,epochs=200000)