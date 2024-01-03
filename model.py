import torch
import torch.nn as nn
import os

class AtariNet(nn.Module):
    def __init__(self, nb_actions=14):
        super(AtariNet, self).__init__()

        self.relu = nn.ReLU()
        #TODO check CNN theory
        self.conv1 = nn.Conv2d(4,8, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p=0.2) #drops some random weights to add some randomness

        self.action_value1 = nn.Linear(144, 256)
        self.action_value2 = nn.Linear(256, 256)
        self.action_value3 = nn.Linear(256, nb_actions)

        self.state_value1 = nn.Linear(144, 256)
        self.state_value2 = nn.Linear(256, 256)
        self.state_value3 = nn.Linear(256, 1)


    def forward(self, x):
        #print("Input Size:", x.size())

        x = torch.Tensor(x)
        #print("After torch.Tensor Conversion Size:", x.size())

        x = self.relu(self.conv1(x))
        #print("After conv1 and ReLU Size:", x.size())

        x = self.relu(self.conv2(x))
        #print("After conv2 and ReLU Size:", x.size())

        x = self.relu(self.conv3(x))
        #print("After conv3 and ReLU Size:", x.size())

        x = self.flatten(x)
        print("After Flatten Size:", x.size())



        state_value = self.relu(self.state_value1(x))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value2(state_value))
        state_value = self.dropout(state_value)
        state_value = self.relu(self.state_value3(state_value))



        action_value = self.relu(self.action_value1(x))
        action_value = self.dropout(action_value)
        action_value = self.relu(self.action_value2(action_value))
        action_value = self.dropout(action_value)
        action_value = self.action_value3(action_value)


        output = state_value + (action_value - action_value.mean())

        return output

    def save_the_model(self, weights_filename='models/latest.pt'):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(self.state_dict(), weights_filename)

    def load_the_model(self, weights_filename='models/latest.pt'):
        try:
            self.load_state_dict(torch.load(weights_filename))
            print("Model Load Sucessfully")
        except:
            print("failed to load the model")
