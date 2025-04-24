# -*- coding: utf-8 -*-
"""
Training script for MR-VFL Scheduler
This script trains the MR-VFL scheduler on the vehicular federated learning environment.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from vehicular_fl_env import VehicularFLEnv
from mr_vfl_scheduler import MRVFLScheduler

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def train_scheduler(args):
    """Train the MR-VFL scheduler"""
    # Create output directories
    log_dir = "mr_vfl_logs"
    model_dir = "mr_vfl_models"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Creating environment with {args.vehicle_count} vehicles, {args.max_rounds} max rounds...")
    # Create environment
    try:
        env = VehicularFLEnv(
            vehicle_count=args.vehicle_count,
            max_round=args.max_rounds,
            sync_limit=args.sync_limit
        )
        print(f"Environment created successfully")
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise

    print(f"Creating scheduler with hidden_dim={args.hidden_dim}, n_layers={args.n_layers}...")
    # Create scheduler
    try:
        scheduler = MRVFLScheduler(
            input_dim=6,  # Fixed input dimension based on vehicle state
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            lr=args.learning_rate,
            gamma=args.gamma
        )
        print(f"Scheduler created successfully")
    except Exception as e:
        print(f"Error creating scheduler: {e}")
        raise

    # Load pretrained model if specified
    if args.load_model and os.path.exists(args.load_model):
        try:
            scheduler.load_model(args.load_model)
            print(f"Loaded pretrained model from {args.load_model}")
        except Exception as e:
            print(f"Error loading model: {e}")

    # Train scheduler
    print(f"Starting training for {args.num_episodes} episodes...")
    try:
        # Manual training loop for better debugging
        rewards = []
        performances = []

        for episode in range(args.num_episodes):
            print(f"Episode {episode+1}/{args.num_episodes} starting...")
            # Reset environment
            state = env.reset()
            scheduler.selection_history = np.zeros(env.vehicle_count)

            episode_reward = 0
            done = False
            step = 0

            while not done:
                # Create mask for available vehicles
                available_mask = np.zeros(env.vehicle_count)
                for i, vehicle in enumerate(env.vehicles):
                    if env._is_vehicle_available(vehicle):
                        available_mask[i] = 1

                # Select action
                try:
                    action = scheduler.select_action(state, available_mask)
                    print(f"  Step {step+1}: Selected vehicle {action[0]} with params: {action[1:]}")
                except Exception as e:
                    print(f"  Error selecting action: {e}")
                    raise

                # Take step in environment
                try:
                    next_state, reward, done, info = env.step(action)
                    print(f"  Step {step+1}: Reward = {reward:.4f}, Done = {done}")
                except Exception as e:
                    print(f"  Error taking step in environment: {e}")
                    raise

                # Update selection history
                scheduler.selection_history[action[0]] += 1

                # Store experience in buffer
                scheduler.buffer.push(state, action, reward, next_state, done)

                # Update networks
                scheduler.steps += 1
                if scheduler.steps % scheduler.update_every == 0:
                    try:
                        scheduler.update()
                        print(f"  Updated networks at step {scheduler.steps}")
                    except Exception as e:
                        print(f"  Error updating networks: {e}")
                        raise

                # Update state and metrics
                state = next_state
                episode_reward += reward
                step += 1

            # Record episode metrics
            rewards.append(episode_reward)
            performances.append(env.current_model_performance)

            print(f"Episode {episode+1}/{args.num_episodes} completed: "
                  f"Reward = {episode_reward:.4f}, "
                  f"Performance = {env.current_model_performance:.4f}")

            # Save model periodically
            if (episode + 1) % 10 == 0 or episode == args.num_episodes - 1:
                model_path = os.path.join(model_dir, f"mr_vfl_scheduler_ep{episode+1}.pt")
                scheduler.save_model(model_path)
                print(f"Saved model to {model_path}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        # Save partial results if available
        if rewards:
            print(f"Saving partial results from {len(rewards)} episodes")
            return scheduler, rewards, performances
        else:
            raise

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(model_dir, f"mr_vfl_scheduler_{timestamp}.pt")
    scheduler.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot and save results
    plt.figure(figsize=(15, 10))

    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    # Plot smoothed rewards
    plt.subplot(2, 2, 2)
    window_size = 10
    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(smoothed_rewards)
    plt.title(f'Smoothed Rewards (Window={window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)

    # Plot model performance
    plt.subplot(2, 2, 3)
    plt.plot(performances)
    plt.title('Model Performance')
    plt.xlabel('Episode')
    plt.ylabel('Performance')
    plt.grid(True)

    # Plot performance improvement rate
    plt.subplot(2, 2, 4)
    performance_diffs = np.diff(performances, prepend=0)
    plt.plot(performance_diffs)
    plt.title('Performance Improvement Rate')
    plt.xlabel('Episode')
    plt.ylabel('Improvement')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"training_results_{timestamp}.png"))

    # Save metrics to CSV
    import pandas as pd
    metrics_df = pd.DataFrame({
        'Episode': range(1, len(rewards) + 1),
        'Reward': rewards,
        'Performance': performances
    })
    metrics_df.to_csv(os.path.join(log_dir, f"training_metrics_{timestamp}.csv"), index=False)

    print(f"Training results saved to {log_dir}")

    return scheduler, rewards, performances

def evaluate_scheduler(scheduler, args):
    """Evaluate the trained scheduler"""
    # Create environment
    env = VehicularFLEnv(
        vehicle_count=args.vehicle_count,
        max_round=args.max_rounds,
        sync_limit=args.sync_limit
    )

    # Evaluation metrics
    episode_rewards = []
    episode_performances = []
    fairness_violations = []
    decision_times = []

    print("\nEvaluating scheduler...")

    # Run evaluation episodes
    for episode in range(args.eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Create mask for available vehicles
            available_mask = np.zeros(env.vehicle_count)
            for i, vehicle in enumerate(env.vehicles):
                if env._is_vehicle_available(vehicle):
                    available_mask[i] = 1

            # Select action
            import time
            start_time = time.time()
            action = scheduler.select_action(state, available_mask)
            decision_time = (time.time() - start_time) * 1000  # ms
            decision_times.append(decision_time)

            # Take step in environment
            next_state, reward, done, info = env.step(action)

            # Update state and metrics
            state = next_state
            episode_reward += reward

            # Check fairness violation
            if info['fairness_violation']:
                fairness_violations.append(1)
            else:
                fairness_violations.append(0)

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_performances.append(env.current_model_performance)

        print(f"Episode {episode+1}/{args.eval_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Performance={env.current_model_performance:.4f}")

    # Calculate average metrics
    avg_reward = np.mean(episode_rewards)
    avg_performance = np.mean(episode_performances)
    avg_decision_time = np.mean(decision_times)
    fairness_violation_rate = np.mean(fairness_violations)

    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Performance: {avg_performance:.4f}")
    print(f"Average Decision Time: {avg_decision_time:.2f} ms")
    print(f"Fairness Violation Rate: {fairness_violation_rate:.2f}")

    return {
        'avg_reward': avg_reward,
        'avg_performance': avg_performance,
        'avg_decision_time': avg_decision_time,
        'fairness_violation_rate': fairness_violation_rate,
        'episode_rewards': episode_rewards,
        'episode_performances': episode_performances
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train MR-VFL Scheduler')

    # Environment parameters
    parser.add_argument('--vehicle_count', type=int, default=30, help='Number of vehicles in environment')
    parser.add_argument('--max_rounds', type=int, default=100, help='Maximum number of rounds')
    parser.add_argument('--sync_limit', type=int, default=1000, help='Synchronization time limit')

    # Scheduler parameters
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--load_model', type=str, default=None, help='Path to pretrained model')

    # Evaluation parameters
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')

    args = parser.parse_args()

    # Train scheduler
    scheduler, rewards, performances = train_scheduler(args)

    # Evaluate scheduler if specified
    if args.eval:
        eval_results = evaluate_scheduler(scheduler, args)

    print("Done!")

if __name__ == "__main__":
    main()
