# CARLA-RL

Q-learning implementation in Pytorch for a car agent to drive based on RGB image input in [CARLA](https://carla.org/) simulator.

Based on the original Tensorflow implementation: [https://github.com/Sentdex/Carla-RL/](https://github.com/Sentdex/Carla-RL/).

## Training

Start CARLA simulator (tested with the [0.10.0](https://github.com/carla-simulator/carla/releases/tag/0.10.0) release version)

```bash
bash carla_rl/script/start_carla.sh
```

Run the Q-learning agent training:
```bash
python carla_rl/scripts/train.py
```

## Reference

- [Self-driving cars with CARLA and Python](https://pythonprogramming.net/introduction-self-driving-autonomous-cars-carla-python/)
- [RL lecture](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/self-driving-cars/)
