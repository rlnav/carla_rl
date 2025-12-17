#!/usr/bin/env python3

import random
import time
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from carla_rl.src.config import *
from carla_rl.src.critic import MobileNetV2DQN


class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_model = MobileNetV2DQN(num_actions=3).to(self.device)
        self.target_model = MobileNetV2DQN(num_actions=3).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        # self.optimizer = optim.Adam(self.policy_model.parameters(), lr=0.001)
        # self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.policy_model.parameters(), lr=3e-4, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        self.tensorboard = SummaryWriter(log_dir=f"logs/{datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}")
        self.target_update_counter = 0
        
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

        self.step = 0

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Extract states and convert to tensors
        current_states = np.array([transition[0] for transition in minibatch]) / 255.0
        current_states = torch.FloatTensor(current_states).permute(0, 3, 1, 2).to(self.device)  # (B, C, H, W) images
        
        new_states = np.array([transition[3] for transition in minibatch]) / 255.0
        new_states = torch.FloatTensor(new_states).permute(0, 3, 1, 2).to(self.device)  # (B, C, H, W) images

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

        self.policy_model.train()
        self.optimizer.zero_grad()
        
        predictions = self.policy_model(X)
        loss = self.criterion(predictions, y)

        self.step += 1
        # Pass a Python scalar to TensorBoard (loss.item()) and flush so the
        # background training thread's logs appear promptly.
        try:
            self.tensorboard.add_scalar("loss", loss.item(), self.step)
            self.tensorboard.flush()
        except Exception:
            # If writer fails for any reason, skip logging to avoid crashing training thread
            pass
        
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 100)
        self.optimizer.step()

        # # Update target model
        # self.target_update_counter += 1
        # if self.target_update_counter > UPDATE_TARGET_EVERY:
        #     self.target_model.load_state_dict(self.policy_model.state_dict())
        #     self.target_update_counter = 0
        target_state = self.target_model.state_dict()
        policy_state = self.policy_model.state_dict()
        for key in policy_state:
            target_state[key] = policy_state[key] * TAU + target_state[key] * (1.0 - TAU)
        self.target_model.load_state_dict(target_state)

    def get_qs(self, state):
        state = np.array(state) / 255.0
        state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            qs = self.policy_model(state)
        
        return qs[0].cpu().numpy()
    
    def train_in_loop(self):
        # Initialize with dummy data
        X = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(self.device)
        y = torch.randn(1, 3).to(self.device)
        
        self.policy_model.train()
        self.optimizer.zero_grad()
        outputs = self.policy_model(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            
            self.train()
            time.sleep(0.01)