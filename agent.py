
import torch
import random
class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.memory = []
        self.position = 0
        self.memory_max_report

    def insert(self,transition): #switch to RAM if GPU memory is full
        transition = [item.to('cpu') for item in transition]

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)

    def sample(self,batch_size=32):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]

    def can_sample(self,batch_size):
        return len(self.memory) >= batch_size*10

    def __len__(self):
        return len(self.memory)
class Agent:
    pass