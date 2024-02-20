import collections
import cv2
import gym
import numpy as np
import torch
from PIL import Image

class DQNKungFuMaster(gym.Wrapper):
    def __init__(self,render_mode='rgb_array',device='cpu'):
        env = gym.make("ALE/KungFuMaster-v5",render_mode=render_mode)

        super(DQNKungFuMaster, self).__init__(env)   
        self.image_shape = (84,84)        
        self.device=device

    def step(self,action): #take step and get observation
        total_reward=0
        done = False
        buffer=[]
        for i in range(4): #you dont really want to react on every frame, goup frames 4 at a time
            observation,reward,done,trucated,info=self.env.step(action)
            total_reward+=reward

            proccessed_img = Image.fromarray(observation)
            proccessed_img = proccessed_img.resize(self.image_shape)
            proccessed_img = proccessed_img.convert("L")  
            proccessed_img = self.crop_image(proccessed_img,5,12,20)        

            buffer.append(proccessed_img)      
            if done == True:
                break
        
        frame=np.concatenate(buffer[-1:],axis=1)
        buffer.clear()

        frame = np.array(frame)        
        frame = torch.from_numpy(frame)
        frame = frame.unsqueeze(0)
        frame = frame.unsqueeze(0)
        frame = frame/255.0
        frame = frame.to(self.device)

        
        total_reward = torch.tensor(total_reward).view(1,-1).float()
        total_reward = total_reward.to(self.device)

        done = torch.tensor(done).view(1,-1)
        done = done.to(self.device)

        #max_frame = max_frame.to(self.device)
        return frame, total_reward,done,info


    def crop_image(self,image,left_px,top_px,bottom_px):
        width,height = image.size
        cropped_image = image.crop((left_px, top_px, width, height - bottom_px))
        width,height = cropped_image.size
        strip_upper_line=41
        strip_width = 15
        extra_cropped_image_up = cropped_image.crop((0, 0, width, height - strip_upper_line))
        extra_cropped_image_down = cropped_image.crop((0, strip_upper_line-strip_width, width, height))
        new_height=extra_cropped_image_up.height+extra_cropped_image_down.height
        extra_cropped_image = Image.new('L', (width, new_height))
        extra_cropped_image.paste(extra_cropped_image_up, (0, 0))
        extra_cropped_image.paste(extra_cropped_image_down, (0, extra_cropped_image_up.height))
        #print(extra_cropped_image.size) (79, 37) size 
        return extra_cropped_image

    def reset(self):        
        observation , _ = self.env.reset()        
        #observation = self.process_observation(observation)
        proccessed_img = Image.fromarray(observation)
        proccessed_img = proccessed_img.resize(self.image_shape)
        proccessed_img = proccessed_img.convert("L")  
        images=np.concatenate((proccessed_img,proccessed_img,proccessed_img,proccessed_img),axis=1)
        images = np.array(images)        
        images = torch.from_numpy(images)
        images = images.unsqueeze(0)
        images = images.unsqueeze(0)
        images = images/255.0
        images = images.to(self.device)
        
  
        return observation

    