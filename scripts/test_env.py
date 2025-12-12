#!/usr/bin/env python3

import cv2
import logging
import time
import numpy as np

from carla_rl.src.config import *
from carla_rl.src.environment import CarEnv


logging.basicConfig(level="DEBUG")
logger = logging.getLogger()


def main():
    FPS = 10

    env = CarEnv()

    state = env.reset()
    done = False
    episode_start = time.time()

    for _ in range(100):
        action = np.random.randint(3)

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