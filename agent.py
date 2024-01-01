import torch
import random
import copy
import torch.optim as optim
import torch.nn.functional as F
from plot import Liveplot
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.memory = []
        self.position = 0
        self.memory_max_report

    def insert(self, transition):  # switch to RAM if GPU memory is full
        transition = [item.to('cpu') for item in transition]

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory.remove(self.memory[0])
            self.memory.append(transition)

    def sample(self, batch_size=32):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items).to(self.device) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, model, device='cpu', epsilon=1.0, min_epsilon=0.1, nb_warmup=10000, nb_actions=None,
                 memory_capacity=10000,
                 batch_size=32, learning_rate=0.00025):
        self.memory = ReplayMemory(device=device, capacity=memory_capacity)
        self.model = model
        self.target_model = copy.deepcopy(model).eval()
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = 1 - (((epsilon - min_epsilon) / nb_warmup) * 2)
        self.batch_size = batch_size
        self.model.to(device)
        self.target_model.to(device)
        self.gamma = 0.99
        self.nb_actions = nb_actions

        self.optimizer = optim.AdanW(model.parameters, lr=learning_rate)

        print(f"starting epsilon is {self.epsilon}")
        print(f"epsilon decay is {self.epsilon_decay}")

    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.nb_actions, (1, 1))
        else:
            av = self.model(state).detach()
            return torch.argmax(av, dim=1, keepdim=True)

    def train(self, env, epochs):
        stats = {'Returns': [], 'AvgReturns': [], 'EpsilonCheckpoint': []}
        plotter = Liveplot()

        for epoch in range(1, epochs + 1):
            state = env.reset()
            done = False
            ep_return = 0

            while not done:
                action = self.get_action(state)

                next_state, reward, done, info = env.step(action)

                self.memory.insert([state, action, reward, done, next_state])

                if self.memory.can_sample(self.batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = self.memory.sample(self.batch_size)
                    qsa_b = self.model(state_b).gather(1, action_b)
                    next_qsa_b = self.target_model(next_state_b)
                    next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]
                    target_b = reward + ~done_b * self.gamma * next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state
                ep_return += reward.item()
            stats['Returns'].append(ep_return)

            if self.epsilon > self.min_epsilon:
                self.epsilon = self.epsilon * self.epsilon_decay

            if epoch % 10 == 0:
                self.model.save_the_model()
                print(" ")

                average_returns = np.mean(stats['Returns'][-100:])
                stats['AvgReturns'].append(average_returns)
                stats['EpsilonCheckpoint'].append(self.epsilon)
                if (len(stats['Returns'])) > 100:
                    print(f"Epoch {epoch} - Avg return: {np.mean(stats['Returns'][-100:])} - Epsilon {self.epsilon}")
                else:
                    print(f"Epoch {epoch} - Avg return: {np.mean(stats['Returns'][-1:])} - Epsilon {self.epsilon}")

            if epoch % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                plotter.update_plot(stats)

            if epoch % 1000 == 0:
                self.model.save_the_model(f"models/model_iter_{epoch}.pt")
        return stats
