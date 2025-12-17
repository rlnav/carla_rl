"""Class-based rewrite of the PyTorch DQN tutorial example.

This keeps the original training behaviour but organizes code into a
readable `DQNAgent` class and logs scalars to TensorBoard:
- episode durations
- loss per optimization step
- epsilon decay over time

Usage:
    python3 carla_rl/tutorials/dqn_tutorial.py
    tensorboard --logdir runs
"""

import math
import random
import logging
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os


logging.basicConfig(level="DEBUG")
logger = logging.getLogger(name="dqn_cartpole")


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """Simple replay buffer using a deque."""

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition to the replay buffer.

        The arguments should match the Transition namedtuple fields
        (state, action, next_state, reward).
        """
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """Return a list of ``batch_size`` random transitions.

        This uses Python's :pyfunc:`random.sample` to pick unique items.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current number of stored transitions."""
        return len(self.memory)


class DQN(nn.Module):
    """Simple 3-layer MLP for Q-value approximation."""

    def __init__(self, n_observations: int, n_actions: int):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for input observation(s).

        Args:
            x: a tensor of shape (batch, n_observations) or (1, n_observations)

        Returns:
            Tensor of shape (batch, n_actions) with unnormalized Q-values.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    """Model wrapper that holds the policy and target networks.

    DQNAgent is intentionally lightweight: it contains the policy and
    target networks and small helpers for selecting greedy or random
    actions. Training state (replay buffer, optimizer, loss, epsilon
    scheduling) is owned by :class:`Trainer` to separate concerns.
    """

    def __init__(self, n_observations: int, n_actions: int, device: torch.device):
        # agent holds only the networks and exposes inference helpers.
        self.device = device

        # networks
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # action space size (for random actions)
        self.n_actions = n_actions

    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return the greedy action tensor for a given state.

        The returned tensor has shape (1, 1) and holds the selected action
        index (dtype: long) ready to be passed to ``env.step(action.item())``.
        """
        with torch.no_grad():
            return self.policy_net(state).max(1).indices.view(1, 1)

    def random_action(self) -> torch.Tensor:
        """Return a random action tensor (environment-agnostic).

        Uses :pyfunc:`random.randrange` with the agent's ``n_actions`` to
        pick a valid action index.
        """
        return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


class Trainer:
    """Owns the environment, seeding, writer and coordinates training.

    Trainer creates the env, the agent (passing the writer), and runs
    episodes by calling into the agent for action selection and optimization.
    """

    def __init__(self, env_name: str = "CartPole-v1", seed: int = 42, device: torch.device | None = None):
        self.seed = seed
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        logger.debug(f"Instantiated {env_name} environment.")

        # reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        logger.debug(f"Set random seed: {seed}.")

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = device

        # create writer and agent
        self.writer = SummaryWriter()

        # training hyperparameters (moved here)
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.01
        self.eps_decay = 2500
        self.tau = 0.005
        self.lr = 3e-4

        # training components (moved here from agent)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.training_steps = 0
        self.episode_durations = []
        self.episode_rewards = []
        self.best_reward = float('-inf')

        # determine shapes
        state, _ = self.env.reset()
        n_observations = len(state)
        n_actions = self.env.action_space.n
        logger.debug("N observations: %s. N actions: %s.", n_observations, n_actions)

        # instantiate agent (model-only) and training artifacts
        self.agent = DQNAgent(n_observations=n_observations, n_actions=n_actions, device=self.device)
        self.optimizer = optim.AdamW(self.agent.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()
        logger.debug("Instantiated DQN agent and training components.")

        # checkpoint directory inside the TensorBoard run directory
        try:
            ckpt_dir = os.path.join(self.writer.log_dir, 'checkpoints')
        except Exception:
            ckpt_dir = os.path.join('runs', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        self.checkpoint_dir = ckpt_dir

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select an action using an epsilon-greedy policy.

        This method computes the current epsilon using the trainer's
        schedule, logs it to TensorBoard, and returns either the greedy
        action from the agent or a random action.

        Args:
            state: single observation tensor shaped (1, n_observations).

        Returns:
            A tensor shaped (1, 1) containing the chosen action index.
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if self.writer is not None:
            try:
                self.writer.add_scalar('epsilon', eps_threshold, self.steps_done)
            except Exception:
                pass

        if sample > eps_threshold:
            return self.agent.greedy_action(state)
        else:
            return self.agent.random_action()

    def optimize_model(self):
        """Run one optimization step on a sampled replay batch.

        Steps performed:
        1. Sample a batch from the trainer-owned replay memory.
        2. Compute current Q-values and expected Q-values using the
           target network for non-final next states.
        3. Compute Huber loss, backpropagate and apply an optimizer step.
        4. Log the scalar loss value to TensorBoard.
        """
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        if any(non_final_mask):
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        else:
            non_final_next_states = torch.empty((0,), device=self.device)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a)
        state_action_values = self.agent.policy_net(state_batch).gather(1, action_batch)

        # V(s_{t+1}) for non-final next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.numel() > 0:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.agent.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.agent.policy_net.parameters(), 100)
        self.optimizer.step()

        # log loss
        self.training_steps += 1
        if self.writer is not None:
            try:
                self.writer.add_scalar('loss', float(loss.item()), self.training_steps)
            except Exception:
                pass

    def soft_update_target(self):
        """Soft-update the target network parameters.

        Performs the standard soft-update:
            θ_target = τ * θ_policy + (1 - τ) * θ_target

        This keeps the target network as a slowly-moving average of the
        policy network which stabilizes learning.
        """
        target_state = self.agent.target_net.state_dict()
        policy_state = self.agent.policy_net.state_dict()
        for key in policy_state:
            target_state[key] = policy_state[key] * self.tau + target_state[key] * (1.0 - self.tau)
        self.agent.target_net.load_state_dict(target_state)

    def train(self, num_episodes: int | None = None):
        """Run the main training loop.

        The Trainer runs ``num_episodes`` episodes (chosen automatically
        if ``None``), interacts with the environment, stores transitions
        in the replay buffer, performs optimization steps, and logs
        episode-level metrics to TensorBoard.

        Args:
            num_episodes: number of episodes to train for; if ``None``, a
                default is chosen depending on device availability.
        """
        if num_episodes is None:
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                num_episodes = 600
            else:
                num_episodes = 50

        for i_episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            ep_reward = 0.0
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                # convert reward to tensor for storage/learning and accumulate scalar
                reward = torch.tensor([reward], device=self.device)
                ep_reward += float(reward.item())
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                self.optimize_model()
                self.soft_update_target()

                if done:
                    self.episode_durations.append(t + 1)
                    # log episode duration and running mean(100)
                    try:
                        self.writer.add_scalar('episode_duration', t + 1, i_episode)
                        if len(self.episode_durations) >= 100:
                            mean100 = sum(self.episode_durations[-100:]) / 100.0
                            self.writer.add_scalar('episode_duration_mean_100', mean100, i_episode)
                    except Exception:
                        pass
                    # log episode reward and running mean of rewards
                    try:
                        self.writer.add_scalar('episode_reward', ep_reward, i_episode)
                        self.episode_rewards.append(ep_reward)
                        if len(self.episode_rewards) >= 100:
                            mean_r100 = sum(self.episode_rewards[-100:]) / 100.0
                            self.writer.add_scalar('episode_reward_mean_100', mean_r100, i_episode)
                    except Exception:
                        pass

                    # save checkpoint if this episode is the best so far
                    if ep_reward > self.best_reward:
                        self.best_reward = ep_reward
                        ckpt_path = os.path.join(self.checkpoint_dir, f"best_policy_ep{i_episode}_r{ep_reward}.pth")
                        try:
                            torch.save({
                                'episode': i_episode,
                                'model_state_dict': self.agent.policy_net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'reward': ep_reward,
                            }, ckpt_path)
                            logger.info("Saved new best model to %s (reward=%.2f)", ckpt_path, ep_reward)
                        except Exception as e:
                            logger.warning("Failed to save checkpoint: %s", e)
                    break

        logger.info("Complete.")
        try:
            self.writer.close()
        except Exception:
            pass


def main():
    """Create a Trainer and run training.

    This entry point keeps the example simple. For real experiments
    consider adding an argument parser to control hyperparameters,
    checkpointing and logging directories.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    trainer = Trainer(env_name="CartPole-v1", device=device)
    trainer.train()


if __name__ == '__main__':
    main()
