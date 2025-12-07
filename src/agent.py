#!/usr/bin/env python3

import random
import time
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from carla_navigation.src.config import *
from carla_navigation.src.logging import ModifiedTensorBoard


class MobileNetV2DQN(nn.Module):
    """DQN model using MobileNetV2 backbone for feature extraction"""
    
    def __init__(self, num_actions=3):
        super(MobileNetV2DQN, self).__init__()
        
        # Load pre-trained MobileNetV2 model
        mobilenet = models.mobilenet_v2(pretrained=False)
        
        # Use all layers except the final classification layer
        self.features = mobilenet.features
        
        # Adaptive pooling to get fixed output size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Q-value head
        self.fc = nn.Linear(1280, num_actions)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        
        self.model = MobileNetV2DQN(num_actions=3).to(self.device)
        self.target_model = MobileNetV2DQN(num_actions=3).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Extract states and convert to tensors
        current_states = np.array([transition[0] for transition in minibatch]) / 255.0
        current_states = torch.FloatTensor(current_states).permute(0, 3, 1, 2).to(self.device)
        
        new_states = np.array([transition[3] for transition in minibatch]) / 255.0
        new_states = torch.FloatTensor(new_states).permute(0, 3, 1, 2).to(self.device)

        # Get Q values for current states
        with torch.no_grad():
            current_qs_list = self.model(current_states)
            future_qs_list = self.target_model(new_states)

        X = []
        y = []

        for index, transition in enumerate(minibatch):
            (current_state, action, reward, new_state, done) = transition

            if not done:
                max_future_q = torch.max(future_qs_list[index]).item()
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index].clone().detach()
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs.cpu().numpy())

        # Training step
        X = np.array(X) / 255.0
        X = torch.FloatTensor(X).permute(0, 3, 1, 2).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        
        predictions = self.model(X)
        loss = self.criterion(predictions, y)
        
        loss.backward()
        self.optimizer.step()

        # Update target model
        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

    def get_qs(self, state):
        state = np.array(state) / 255.0
        state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            qs = self.model(state)
        
        return qs[0].cpu().numpy()
    
    def train_in_loop(self):
        # Initialize with dummy data
        X = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(self.device)
        y = torch.randn(1, 3).to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            
            self.train()
            time.sleep(0.01)