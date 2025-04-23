# -*- coding: utf-8 -*-
"""
Training script for Mamba-Based Actor-Critic Scheduler for Vehicle Model Updates
"""

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from vehicle_env import VehicleModelUpdateEnv
from mamba_scheduler import MambaActor, MambaCritic, StreamingMambaScheduler, Vehicle, GlobalState

# Set random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create directories for saving models and logs
log_dir = "mamba_scheduler_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

model_dir = "mamba_scheduler_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def convert_env_vehicle_to_model_vehicle(env_vehicle):
    """Convert environment vehicle format to model vehicle format"""
    return Vehicle(
        vehicle_id=env_vehicle['vehicle_id'],
        model_version=env_vehicle['model_version'],
        sojourn_time=env_vehicle['sojourn_time'],
        compute_capacity=env_vehicle['compute_capacity'],
        data_quality=env_vehicle['data_quality'],
        connectivity=env_vehicle['connectivity'],
        vehicle_type=env_vehicle['vehicle_type']
    )

def convert_env_state_to_global_state(env_state):
    """Convert environment state to global state format"""
    global_state = GlobalState()
    global_state.current_model_performance = env_state['current_model_performance']
    global_state.round_number = env_state['current_round']
    global_state.elapsed_time = env_state['elapsed_time']
    global_state.scheduled_count = env_state['scheduled_count']
    global_state.target_vehicle_count = 10  # Default target
    global_state.performance_gap = env_state['performance_gap']
    return global_state

def train(num_episodes=1000, gamma=0.99, lr_actor=3e-4, lr_critic=1e-3, print_interval=10, save_interval=100):
    """Train the Mamba scheduler on the vehicle environment"""
    # Initialize environment
    env = VehicleModelUpdateEnv(
        max_vehicles=100,
        target_performance=0.95,
        max_rounds=100,
        arrival_rate=0.7
    )

    # Initialize models
    vehicle_feature_dim = 6  # model_version, sojourn_time, compute_capacity, data_quality, connectivity, type
    global_state_dim = 6     # current_model_performance, round_number, elapsed_time, scheduled_count, target_count, performance_gap

    actor = MambaActor(vehicle_feature_dim, global_state_dim, d_model=256, n_layers=3, d_state=16).to(device)
    critic = MambaCritic(vehicle_feature_dim, global_state_dim, d_model=256, n_layers=2, d_state=16).to(device)

    # Initialize optimizers
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    # Training metrics
    episode_rewards = []
    episode_performances = []
    episode_lengths = []

    # Start training
    start_time = datetime.now()
    print(f"Starting training at {start_time}")

    for episode in range(1, num_episodes + 1):
        # Reset environment
        env_state = env.reset()
        global_state = convert_env_state_to_global_state(env_state)

        # Initialize scheduler
        scheduler = StreamingMambaScheduler(actor, critic)
        scheduler.global_state = global_state

        # Episode variables
        total_reward = 0
        done = False

        # Run episode
        while not done:
            # Get new vehicles from environment
            new_env_vehicles = env.get_new_vehicles()
            new_vehicles = [convert_env_vehicle_to_model_vehicle(v) for v in new_env_vehicles]

            # Process new vehicles
            for vehicle in new_vehicles:
                scheduler.process_new_vehicle(vehicle)

            # Update scheduler's global state
            scheduler.global_state = convert_env_state_to_global_state(env_state)

            # Get action from scheduler
            with torch.no_grad():
                selected_vehicles = scheduler.make_scheduling_decision(target_count=10)
                selected_ids = [v.vehicle_id for v in selected_vehicles]

            # Take step in environment
            next_env_state, reward, done, info = env.step(selected_ids)

            # Store reward
            total_reward += reward

            # Update state
            env_state = next_env_state

            # Update critic
            if len(selected_vehicles) > 0:
                # Convert vehicles to tensors
                vehicle_tensors = [v.to_tensor() for v in scheduler.vehicle_cache if not v.scheduled]
                if len(vehicle_tensors) > 0:
                    # Update critic
                    current_value = critic(vehicle_tensors, scheduler.global_state.to_tensor())

                    # Calculate target value (simple TD learning)
                    next_global_state = convert_env_state_to_global_state(next_env_state)
                    next_vehicles = [v for v in scheduler.vehicle_cache if not v.scheduled]
                    next_vehicle_tensors = [v.to_tensor() for v in next_vehicles]

                    if len(next_vehicle_tensors) > 0 and not done:
                        with torch.no_grad():
                            next_value = critic(next_vehicle_tensors, next_global_state.to_tensor())
                            target_value = reward + gamma * next_value
                    else:
                        target_value = torch.tensor([[reward]], device=device)

                    # Compute critic loss
                    critic_loss = torch.nn.functional.mse_loss(current_value, target_value)

                    # Update critic
                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    optimizer_critic.step()

                    # Update actor
                    probs = actor(vehicle_tensors, scheduler.global_state.to_tensor())

                    # Compute advantage
                    with torch.no_grad():
                        value = critic(vehicle_tensors, scheduler.global_state.to_tensor())
                        advantage = target_value - value

                    # Compute actor loss
                    action_log_probs = torch.log(probs + 1e-10)
                    actor_loss = -torch.mean(action_log_probs * advantage)

                    # Add entropy regularization
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    actor_loss -= 0.01 * entropy  # Entropy coefficient

                    # Update actor
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

        # Record episode metrics
        episode_rewards.append(total_reward)
        episode_performances.append(env_state['current_model_performance'])
        episode_lengths.append(env_state['current_round'])

        # Print progress
        if episode % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_performance = np.mean(episode_performances[-print_interval:])
            avg_length = np.mean(episode_lengths[-print_interval:])

            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Avg Performance: {avg_performance:.4f} | "
                  f"Avg Length: {avg_length:.1f}")

        # Save models
        if episode % save_interval == 0:
            # Save model checkpoint
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                'episode': episode,
                'reward': total_reward
            }, os.path.join(model_dir, f'mamba_scheduler_checkpoint_{episode}.pt'))

            # Plot and save learning curves
            plot_learning_curves(episode_rewards, episode_performances, episode_lengths, episode)

    # Save final model
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_actor_state_dict': optimizer_actor.state_dict(),
        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
        'episode': num_episodes,
        'reward': total_reward
    }, os.path.join(model_dir, 'mamba_scheduler_final.pt'))

    # Plot final learning curves
    plot_learning_curves(episode_rewards, episode_performances, episode_lengths, num_episodes)

    end_time = datetime.now()
    training_time = end_time - start_time
    print(f"Training completed in {training_time}")

    return actor, critic, episode_rewards, episode_performances, episode_lengths

def plot_learning_curves(rewards, performances, lengths, episode):
    """Plot and save learning curves"""
    plt.figure(figsize=(15, 5))

    # Plot rewards
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # Plot performances
    plt.subplot(1, 3, 2)
    plt.plot(performances)
    plt.title('Model Performance')
    plt.xlabel('Episode')
    plt.ylabel('Final Performance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='Target')
    plt.legend()

    # Plot episode lengths
    plt.subplot(1, 3, 3)
    plt.plot(lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Number of Rounds')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'learning_curves_{episode}.png'))
    plt.close()

def evaluate(model_path, num_episodes=10):
    """Evaluate a trained model"""
    # Load model with weights_only=True for security
    checkpoint = torch.load(model_path, weights_only=True)

    # Initialize models
    vehicle_feature_dim = 6
    global_state_dim = 6

    # Use the same architecture as in training
    actor = MambaActor(vehicle_feature_dim, global_state_dim, d_model=256, n_layers=3, d_state=16).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    # Initialize environment
    env = VehicleModelUpdateEnv(
        max_vehicles=100,
        target_performance=0.95,
        max_rounds=100,
        arrival_rate=0.7
    )

    # Evaluation metrics
    eval_rewards = []
    eval_performances = []
    eval_lengths = []

    for episode in range(num_episodes):
        # Reset environment
        env_state = env.reset()
        global_state = convert_env_state_to_global_state(env_state)

        # Initialize scheduler
        scheduler = StreamingMambaScheduler(actor, None)
        scheduler.global_state = global_state

        # Episode variables
        total_reward = 0
        done = False

        # Run episode
        while not done:
            # Get new vehicles from environment
            new_env_vehicles = env.get_new_vehicles()
            new_vehicles = [convert_env_vehicle_to_model_vehicle(v) for v in new_env_vehicles]

            # Process new vehicles
            for vehicle in new_vehicles:
                scheduler.process_new_vehicle(vehicle)

            # Update scheduler's global state
            scheduler.global_state = convert_env_state_to_global_state(env_state)

            # Get action from scheduler
            with torch.no_grad():
                selected_vehicles = scheduler.make_scheduling_decision(target_count=10)
                selected_ids = [v.vehicle_id for v in selected_vehicles]

            # Take step in environment
            next_env_state, reward, done, info = env.step(selected_ids)

            # Store reward
            total_reward += reward

            # Update state
            env_state = next_env_state

        # Record episode metrics
        eval_rewards.append(total_reward)
        eval_performances.append(env_state['current_model_performance'])
        eval_lengths.append(env_state['current_round'])

        print(f"Evaluation Episode {episode+1}/{num_episodes} | "
              f"Reward: {total_reward:.4f} | "
              f"Performance: {env_state['current_model_performance']:.4f} | "
              f"Length: {env_state['current_round']}")

    # Print average metrics
    avg_reward = np.mean(eval_rewards)
    avg_performance = np.mean(eval_performances)
    avg_length = np.mean(eval_lengths)

    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average Performance: {avg_performance:.4f}")
    print(f"Average Episode Length: {avg_length:.1f}")

    return eval_rewards, eval_performances, eval_lengths

if __name__ == "__main__":
    # Train the model
    print("Starting training...")
    actor, critic, rewards, performances, lengths = train(
        num_episodes=100,  # Reduced for demonstration
        print_interval=5,
        save_interval=20
    )

    # Evaluate the final model
    print("\nEvaluating final model...")
    eval_rewards, eval_performances, eval_lengths = evaluate('mamba_scheduler_models/mamba_scheduler_final.pt')
