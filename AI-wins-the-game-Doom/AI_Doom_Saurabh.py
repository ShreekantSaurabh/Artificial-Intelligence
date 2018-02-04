#AI for Doom

#Imorting the libraries
import numpy as np
import torch
import torch.nn as nn                 #neural network using pytorch
import torch.nn.functional as F       #activation/loss functions are part of this module
import torch.optim as optim           #optimizer eg.ADAM
from torch.autograd import Variable   #converts torch tensor to the variable that contains tensor and gradient


# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper     #imports all the tools in the gym environment
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete  #package for doom game, specifically for the no. of actions (7) for the doom move left/right/straight, turn left/right, run & shoot  

# Importing the other Python files
import experience_replay, image_preprocessing      #py files are present in the same folder  


# Part 1 - Building the AI
#Making the brain
class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()       #activate the inheritence to use all the tools of nn.Module
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        #in_channels is for input channel for convolution, in_channels=1 is for black&white images, in_channels=3 is for colored images.
        #out_channels= 32 is no. of features detectors ie. no. of features we want to detect from 1 input image. So, 32 processed images will be the output
        #kernel_size = 5 is dimension of feature detector ie. 5*5 . For next convolution layer, we will make it of smaller dimensions
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 80, 80)), out_features = 40)   #dimension of image coming from doom is 80 * 80
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
        
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))     #creates an input image with random pixel
        #* allows pass the arguements as a list or tuple
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))   
        #applies convolution to input images and then apply max pooling to convoluted images with kernel size / feature detector = 3, 
        #and stride = 2 ie by how many pixels the feature detector will slide over the image and then all neurons are activated using ReLu activation function  
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))    #apply the same with 2nd & 3rd convolutional layer
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)       #to get no. of nuerons.
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))      #applied the o/p of 3rd convolution layer to the 1st fully connected layer
        x = self.fc2(x)              #applied the output of fc1 to fc2
        return x

#Making the body
class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T                       #Temperature parameter T increases the probability or certainity of action to take

    def forward(self, outputs):
        #forwards the o/p signal of brain to the body of ai to play the 7 actions
        probs = F.softmax(outputs * self.T)    #probs is probability. Temperature parameter T increases the probability or certainity of action to take
        actions = probs.multinomial()
        return actions

#Making the AI
class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        #converting the image into torch format ie. convert input images into numpy array, then convert them into torch tensor and then put it inside torch Variable which contains both tensor and gradient.
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy() 


# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
#gym.make("ppaquette/DoomCorridor-v0") : DoomCorridor-v0 is the environment name of the game which we are playing is imported using gym.make()
#We used PreprocessImage class from image_preprocessing.py to pre-process the input images in square format with dimension 80*80 with gray scale that will come into neural network
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
#whole game's video is imported with above line of code into "videos" folder. After end of the game, we can see the videos of it.
number_actions = doom_env.action_space.n      #no. of actions (7) for the doom move left/right/straight, turn left/right, run & shoot

# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)        #Temperature = 1.0
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay using Eligibility trace (step size = 10)
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)
#memory capacity is 10000 ie. memory is dependent on the last 10000 steps performed by the AI. It gonna learn every 10 steps.


# Implementing Eligibility Trace (n-step Q-learning)
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        #converting states into torch variable
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        #cumulative reward = 0 if last transition of series ie. series[-1] is done, otherwise cumulative reward = output[1].data.max() ie. maximum of q-values  
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state     #state of the 1st transition
        target = output[0].data     #q-value of input state of 1st transition
        target[series[0].action] = cumul_reward      #action of the 1st step of the series = cumulative reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size     #size of the list of reward on which we will compute the moving average
    def add(self, rewards):
        #adds cumulative rewards to list of rewards
        if isinstance(rewards, list):
            #if rewards are into the list then add cumulative rewards to list of rewards
            self.list_of_rewards += rewards
        else:
            #otherwise append cumulative rewards into list of rewards
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            #whenever the length of list of rewards > 100 then delete the 1st element of list of rewards
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)      #average of list of rewards
ma = MA(100)

# Training the AI
loss = nn.MSELoss()         #Mean squared error loss function
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)      #learning rate = 0.001
nb_epochs = 100      #no. of epochs
for epoch in range(1, nb_epochs + 1):
    memory.run_steps(200)       #each epoch is 200 runs of 10 steps
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        optimizer.zero_grad()
        loss_error.backward()   #Back propagation
        optimizer.step()
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))
    if avg_reward >= 1500:
        print("Congratulations, your AI wins")
        break

# Closing the Doom environment
doom_env.close()

