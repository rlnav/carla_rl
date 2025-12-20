import gymnasium as gym
import torch
from car_racer_rl import SkipFrame, ImageNormalization, CarNet


def load_model(checkpoint: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = CarNet(input_dim=(4, 96, 96), output_dim=5).to(device)
    # print(model.state_dict().keys())
    # print("\n-------------------\n")
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)["model"]
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    model.eval()  # Critical for DQN inference
    return model

def main():
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    env = gym.make("CarRacing-v3", render_mode="human",
                lap_complete_percent=0.95, domain_randomize=False, continuous=False)
    
    # # Apply Wrappers to environment
    # env = SkipFrame(env, skip=4)
    # env = gym.wrappers.GrayscaleObservation(env)
    # env = ImageNormalization(env)
    # env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    # checkpoint = "/home/ruslan/Carla-0.10.0-Linux-Shipping/logs/2025-12-19/15-15-08/car_net_4.ckpt"
    # model  = load_model(checkpoint)

    state, _ = env.reset()
    print(f"Image shape: {state.shape}")  # (96, 96, 4)

    # # war-up model
    # state0 = state[0].__array__() if isinstance(state, tuple) else state.__array__()
    # state0 = torch.tensor(state0, device=device).unsqueeze(0)
    # action_values = model(state0, model="online")

    while True:
        # random action
        action = env.action_space.sample()

        # learned model action
        # state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
        # state = torch.tensor(state, device=device).unsqueeze(0)
        # action_values = model(state, model="online")
        # action = torch.argmax(action_values, axis=1).item()

        state, reward, terminated, truncated, _ = env.step(action)

        env.render()

        if terminated or truncated:
            state, _ = env.reset()


if __name__ == "__main__":
    main()