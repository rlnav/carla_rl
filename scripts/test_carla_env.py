#!/usr/bin/env python3

import cv2
import logging
import time
import torch
import numpy as np

from carla_rl.src.config import *
from carla_rl.src.environment import CarEnv
from carla_rl.src.critic import MobileNetV2DQN


logging.basicConfig(level="DEBUG")
logger = logging.getLogger()


def load_model(checkpoint: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = MobileNetV2DQN(num_actions=3).to(device)
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()  # Critical for DQN inference
    return model

def main():
    env = CarEnv()

    state = env.reset()
    done = False

    # device = "cpu"
    # checkpoint_path = "models/MobileNetV2__-102.00max_-151.00avg_-200.00min__1765534399.pth"
    # model = load_model(checkpoint_path, device=device)

    # # warm-up model
    # X = torch.tensor(state / 255.).permute(2, 1, 0).to(device=device, dtype=torch.float32)
    # q_values = model(X.unsqueeze(0))

    for _ in range(100):
        # random action
        action = np.random.randint(3)

        # # trained model greedy action
        # with torch.no_grad():
        #     X = torch.tensor(state / 255.).permute(2, 1, 0).to(device=device, dtype=torch.float32)
        #     q_values = model(X.unsqueeze(0)).squeeze(0)
        #     action = torch.argmax(q_values).item()
        #     logger.debug(f"Q values: {q_values}, action: {action}")

        state, reward, done, _  = env.step(action)
        logger.debug(f"Step reward: {reward}")

        if state is not None:
            cv2.imshow(f"Car camera", state)
            cv2.waitKey(1)

        if done:
            logger.debug("Episode is finished")
            break
        time.sleep(1. / FPS)

    for actor in env.actor_list:
        actor.destroy()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print(ex)