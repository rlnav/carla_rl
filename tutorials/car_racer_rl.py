#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import torch
import time, datetime

from pathlib import Path
from tqdm import tqdm
from torch import nn
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    """TensorBoard-based metric logger that preserves the on-disk text log.

    Public API kept the same as the previous logger:
      - log_step(reward, loss, q)
      - log_episode()
      - record(episode, epsilon, step)

    This writes scalars to TensorBoard and also appends a human-readable
    line to `save_dir/log` for convenience.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.save_log = save_dir / "log"
        # SummaryWriter will create a timestamped run dir under save_dir
        self.writer = SummaryWriter(log_dir=str(save_dir))

        # History metrics (kept for backward compatibility)
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode accumulator
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += float(reward)
        self.curr_ep_length += 1
        if loss is not None:
            try:
                self.curr_ep_loss += float(loss)
                self.curr_ep_q += float(q) if q is not None else 0.0
                self.curr_ep_loss_length += 1
            except Exception:
                # best-effort: ignore conversion errors
                pass

    def log_episode(self):
        """Finalize current episode and append averages to history."""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0.0
            ep_avg_q = 0.0
        else:
            ep_avg_loss = float(self.curr_ep_loss / self.curr_ep_loss_length)
            ep_avg_q = float(self.curr_ep_q / self.curr_ep_loss_length)

        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        # Reset accumulators for next episode
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        """Record aggregated metrics to TensorBoard and to text log.

        `step` is used as the global_step for TensorBoard scalars so graphs
        align with optimizer/interaction steps.
        """
        mean_ep_reward = float(np.round(np.mean(self.ep_rewards[-100:]) if self.ep_rewards else 0.0, 3))
        mean_ep_length = float(np.round(np.mean(self.ep_lengths[-100:]) if self.ep_lengths else 0.0, 3))
        mean_ep_loss = float(np.round(np.mean(self.ep_avg_losses[-100:]) if self.ep_avg_losses else 0.0, 3))
        mean_ep_q = float(np.round(np.mean(self.ep_avg_qs[-100:]) if self.ep_avg_qs else 0.0, 3))

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = float(np.round(self.record_time - last_record_time, 3))

        # Print to terminal (compact)
        print(
            f"Episode {episode} - Step {step} - Epsilon {epsilon:.3f} - "
            f"Mean Reward {mean_ep_reward} - Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        # Log to TensorBoard using `step` as global step
        self.writer.add_scalar("train/mean_reward", mean_ep_reward, global_step=step)
        self.writer.add_scalar("train/mean_length", mean_ep_length, global_step=step)
        self.writer.add_scalar("train/mean_loss", mean_ep_loss, global_step=step)
        self.writer.add_scalar("train/mean_q", mean_ep_q, global_step=step)
        self.writer.add_scalar("train/epsilon", epsilon, global_step=step)
        self.writer.flush()

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass

    def __del__(self):
        # best-effort close
        try:
            self.writer.close()
        except Exception:
            pass


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            state, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, trunk, info


class ImageNormalization(gym.Wrapper):
    """Normalize image observations to [0,1] float32 - CNN standard"""
    
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.env.observation_space
        # Update space: uint8(0-255) → float32(0-1)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=obs_space.shape, 
            dtype=np.float32
        )
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        state = self._normalize(state)
        return state, reward, done, truncated, info
    
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        state = self._normalize(state)
        return state, info
    
    def _normalize(self, state):
        """(H,W,C) uint8 → (H,W,C) float32 [0,1]"""
        return state.astype(np.float32) / 255.0


class CarNet(nn.Module):
    """mini CNN structure
        input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 96:
            raise ValueError(f"Expecting input height: 96, got: {h}")
        if w != 96:
            raise ValueError(f"Expecting input width: 96, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class Car:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Car's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = CarNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Car Net

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
        Outputs:
        ``action_idx`` (``int``): An integer representing which action Car will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (``LazyFrame``),
        next_state (``LazyFrame``),
        action (``int``),
        reward (``float``),
        done(``bool``))
        """
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x
        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        # self.memory.append((state, next_state, action, reward, done,))
        self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get Q Estimate
        q_est = self.q_estimate(state, action)

        # Get Q Target
        q_tgt = self.q_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(q_est, q_tgt)

        return (q_est.mean().item(), loss)

    def q_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def q_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, q_estimate, q_target):
        loss = self.loss_fn(q_estimate, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"car_net_{int(self.curr_step // self.save_every)}.ckpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"CarNet saved to {save_path} at step {self.curr_step}")


def main():
    # Initialise the environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = gym.wrappers.GrayscaleObservation(env)
    env = ImageNormalization(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    # Reset the environment to generate the first observation
    state, info = env.reset(seed=42)
    action = env.action_space.sample()
    # receiving the next observation, reward and if the episode has done or truncated
    state, reward, done, truncated, info = env.step(action)
    # print(f"{action}, {state.shape}, {reward}, {done}, {info}")

    save_dir = Path(f"logs/{datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}/")
    save_dir.mkdir(parents=True)

    car = Car(state_dim=(4, 96, 96), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 10_000
    for e in tqdm(range(episodes)):

        state = env.reset()

        # Play the game!
        while True:

            # Run agent on the state
            action = car.act(state)

            # Agent performs action
            next_state, reward, done, trunc, info = env.step(action)

            # Remember
            car.cache(state, next_state, action, reward, done)

            # Learn
            q, loss = car.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Update state
            state = next_state

            # Check if end of game
            if done or trunc:
                break

        logger.log_episode()

        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=car.exploration_rate, step=car.curr_step)


if __name__ == "__main__":
    main()