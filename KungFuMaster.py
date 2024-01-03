import collections
import cv2
import gym
import numpy as np
import torch
from PIL import Image

class DQNKungFuMaster(gym.Wrapper):
    def __init__(self,render_mode='rgb_array',repeat=4,device='cpu'):
        env = gym.make("ALE/KungFuMaster-v5",render_mode=render_mode)

        super(DQNKungFuMaster, self).__init__(env)


        self.repeat = repeat
        #self.lives=env.ale.lives() #get number of lives
        self.frame_buffer=[]
        self.device=device

    def step(self,action): #take step and get observation
        total_reward=0
        done = False

        for i in range(self.repeat): #you dont really want to react on every frame, goup frames 4 at a time
            observation,reward,done,trucated,info=self.env.step(action)
            total_reward+=reward

            #print(info,total_reward)

            #current_lives = info['lives']

            #if current_lives<self.lives:
                #total_reward = total_reward-1
                #self.lives = current_lives

            #print(f"lives:{self.lives} , Total Reward: {total_reward}")
            #print(observation)
            self.frame_buffer.append(observation)
            if done == True:
                break

        last_4_frames = self.frame_buffer[-4:]



        stacked_frames = self.process_observation(last_4_frames)
        #stacked_frames = torch.tensor(stacked_frames, dtype=torch.float32)
        stacked_frames = stacked_frames.to(self.device)

        total_reward = torch.tensor(total_reward).view(1,-1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1,-1)
        done = done.to(self.device)

        #stacked_frames = stacked_frames.to(self.device)
        return stacked_frames, total_reward,done,info

    def reset(self):
        self.frame_buffer = []
        observation , _ = self.env.reset()
        #self.lives=self.env.ale.lives()
        observation = self.process_observation(observation)
        return observation

    def process_observation(self, last_4_frames):
        #print("Stack Size:", observation_stack.shape)
        processed_frames = []

        for i in range(4):
            img = Image.fromarray(last_4_frames[i])
            img = img.convert("L")

            original_width, original_height = img.size
            box = (0, 95, original_width, original_height - 55)
            img = img.crop(box)

            new_size = (100, 40)
            img = img.resize(new_size)

            img = np.array(img)
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)
            #img = img.unsqueeze(0)
            #img = img.squeeze(0)
            img = img / 255.0
            processed_frames.append(img)
            #print("Processed Stack Size:", img.size())

        # Stack the processed frames along the first dimension
        processed_stack = np.stack(processed_frames, axis=1)
        #print("Processed Stack Size before torch:", processed_stack.shape)
        torch_processed_stack = torch.from_numpy(processed_stack)
        torch_processed_stack = torch_processed_stack.to(self.device)

        #print("Processed Stack Size:", torch_processed_stack.size())




        return torch_processed_stack




