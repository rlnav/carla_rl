import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human",
               lap_complete_percent=0.95, domain_randomize=False, continuous=False)
obs, _ = env.reset()
print(f"Image shape: {obs.shape}")  # (96, 96, 3)

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
