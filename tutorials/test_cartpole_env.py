import gymnasium as gym
import torch
from dqn_tutorial import DQN


def load_model(checkpoint: str, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = DQN(n_observations=4, n_actions=2).to(device)
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)["model_state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()  # Critical for DQN inference
    return model


# Simple manual exploration of CartPole-v1 with visualization
def explore_cartpole():
    env = gym.make('CartPole-v1', render_mode='human')

    device = "cpu"
    checkpoint_path = "runs/Dec15_12-45-56_ruslan-HP-Pavilion-Gaming-Laptop-15-ec1xxx/checkpoints/best_policy_ep147_r500.0.pth"
    model = load_model(checkpoint_path, device=device)
    
    state, _ = env.reset()
    total_reward = 0
    step_count = 0

    # warm-up model
    q_values = model(torch.tensor(state).unsqueeze(0).to(device))
    
    print("CartPole exploration started! Close window or Ctrl+C to stop.")
    print("Controls: Arrow keys (if enabled) or automatic random actions.")
    
    try:
        while True:
            # Random actions for exploration (0=left, 1=right)
            # action = env.action_space.sample()

            # trained model greedy action
            with torch.no_grad():
                q_values = model(torch.tensor(state).unsqueeze(0).to(device)).squeeze(0)
                action = torch.argmax(q_values).item()
            # print(f"Q values: {q_values}, action: {action}.")
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
            
            state = next_state
            
            # Print episode stats every 50 steps
            if step_count % 50 == 0:
                print(f"Step {step_count}: Reward so far = {total_reward:.0f}")
            
            # Reset on episode end
            if terminated or truncated:
                print(f"Episode ended! Total reward: {total_reward:.0f}, Steps: {step_count}")
                state, _ = env.reset()
                total_reward = 0
                step_count = 0
                
    except KeyboardInterrupt:
        print("\nExploration stopped.")
    
    env.close()


if __name__ == "__main__":
    explore_cartpole()
