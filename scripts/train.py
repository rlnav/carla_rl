#!/usr/bin/env python3

import cv2
import os
import logging
import random
import time
import numpy as np
from tqdm import tqdm
from threading import Thread

import torch

from carla_rl.src.config import *
from carla_rl.src.environment import CarEnv
from carla_rl.src.agent import DQNAgent


logging.basicConfig(level="DEBUG")
logger = logging.getLogger(name="carla_rl")


if __name__ == "__main__":
    ep_rewards = [MIN_REWARD]

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    os.makedirs("models", exist_ok=True)

    agent = DQNAgent()
    env = CarEnv()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    logger.debug("Started training process")

    while not agent.training_initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((env.img_height, env.img_width, 3)))
    logger.debug("Initialized agent")

    for episode in tqdm(range(1, EPISODES + 1), unit="episodes"):
        env.collision_hist = []
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            if np.random.random() > epsilon:
                q_values = agent.get_qs(current_state)
                action = np.argmax(q_values)
                logger.debug(f"Q values: {q_values}, action: {action}")
            else:
                action = np.random.randint(0, 3)
                logger.debug(f"Random action: {action}")
                time.sleep(1. / FPS)

            new_state, reward, done, _  = env.step(action)
            logger.debug(f"Step reward: {reward}")

            episode_reward += reward

            transition = (current_state, action, reward, new_state, done)
            agent.update_replay_memory(transition)

            current_state = new_state.copy()

            step += 1

            if SHOW_PREVIEW and current_state is not None:
                cv2.imshow("Car camera", current_state)
                cv2.waitKey(1)

            if done:
                break

        for actor in env.actor_list:
            actor.destroy()
        logger.debug(f"Episode reward: {episode_reward}")

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            avg_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.add_scalar("metrics/reward_avg", avg_reward, episode) # reward_min=min_reward, reward_max=max_reward, epsilon=epsilon
            agent.tensorboard.add_scalar("metrics/reward_min", min_reward, episode)
            agent.tensorboard.add_scalar("metrics/reward_max", max_reward, episode)
            agent.tensorboard.add_scalar("metrics/epsilon", epsilon, episode)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                model_path = f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pth'
                torch.save(agent.policy_model.state_dict(), model_path)

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
    
    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    model_path = f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.pth'
    torch.save(agent.policy_model.state_dict(), model_path)