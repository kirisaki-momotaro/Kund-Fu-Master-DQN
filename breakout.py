import collections
import cv2
import gym
import numpy as np
import torch
from PIL import Image

class DQNBreakout(gym.Wrapper):
    def __init__(self,render_mode='rgb_array',repeat=4,device='cpu'):
        env = gym.make("ALE/KungFuMaster-v5",render_mode=render_mode)

        super(DQNBreakout, self).__init__(env)
        self.repeat = repeat
        self.lives=env.ale.lives() #get number of lives
        self.frame_buffer=[]

    def step(self,action): #take step and get observation
        total_reward=0
        done = False

        for i in range(self.repeat): #you dont really want to react on every frame, goup frames 4 at a time
            observation,reward,done,trucated,info=self.env.step(action)
            total_reward+=reward

            print(info)
            self.frame_buffer.append(observation)
            if done == True:
                break

        max_frame = np.max(self.frame_buffer[-2:],axis=0)
        #max_frame = max_frame.to(self.device)
        return max_frame, total_reward,done,info








