# -*- coding: utf-8 -*-
"""
Comparison script for evaluating Mamba scheduler against SOTA methods
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import schedulers
from mamba_scheduler import MambaActor, OptimizedMambaScheduler, Vehicle, GlobalState
from scheduler_comparison_methods import (
    TransformerScheduler,
    GNNScheduler,
    LSTMScheduler,
    MARLScheduler,
    HierarchicalScheduler,
    AttentionScheduler,
    HeuristicScheduler,
    HybridScheduler,
    compare_schedulers,
    plot_comparison_results
)
from vehicle_env import VehicleModelUpdateEnv

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = "scheduler_comparison_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Wrapper class for Mamba scheduler to match comparison interface
class MambaSchedulerWrapper:
    def __init__(self, model_path):
        self.scheduler = OptimizedMambaScheduler(model_path)
        self.global_state = GlobalState()
        
    def select_vehicles(self, vehicles, target_count=10):
        # Convert environment vehicles to scheduler vehicles
        scheduler_vehicles = []
        for v in vehicles:
            if not v['scheduled']:
                vehicle = Vehicle(
                    vehicle_id=v['vehicle_id'],
                    model_version=v['model_version'],
                    sojourn_time=v['sojourn_time'],
                    compute_capacity=v['compute_capacity'],
                    data_quality=v['data_quality'],
                    connectivity=v['connectivity'],
                    vehicle_type=v['vehicle_type']
                )
                scheduler_vehicles.append(vehicle)
        
        # Update global state based on environment
        # (In a real comparison, this would be updated from the environment)
        
        # Make scheduling decision
        selected_vehicles = self.scheduler.inference_mode_scheduling(
            scheduler_vehicles, self.global_state, target_count
        )
        
        # Convert back to environment vehicle format
        return [v for v in vehicles if v['vehicle_id'] in [sv.vehicle_id for sv in selected_vehicles]]

# Wrapper class for ML-based schedulers to match comparison interface
class MLSchedulerWrapper:
    def __init__(self, model):
        self.model = model
        self.global_state = GlobalState()
        
    def select_vehicles(self, vehicles, target_count=10):
        # Convert environment vehicles to tensors
        eligible_vehicles = [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
        
        if not eligible_vehicles:
            return []
        
        # Create vehicle tensors
        vehicle_tensors = []
        for v in eligible_vehicles:
            tensor = torch.tensor([
                v['model_version'],
                v['sojourn_time'],
                v['compute_capacity'],
                v['data_quality'],
                v['connectivity'],
                v['vehicle_type']
            ], dtype=torch.float32).to(device)
            vehicle_tensors.append(tensor)
        
        # Create mask
        mask = torch.ones(len(eligible_vehicles)).to(device)
        
        # Update global state
        global_state_tensor = self.global_state.to_tensor()
        
        # Get selection probabilities
        with torch.no_grad():
            if isinstance(self.model, LSTMScheduler):
                probs, _ = self.model(vehicle_tensors, global_state_tensor, mask)
            else:
                probs = self.model(vehicle_tensors, global_state_tensor, mask)
        
        # Select vehicles based on probabilities
        selected_indices = []
        remaining_indices = list(range(len(eligible_vehicles)))
        
        # Select up to target_count vehicles
        for _ in range(min(target_count, len(remaining_indices))):
            if not remaining_indices:
                break
                
            # Normalize probabilities for remaining vehicles
            remaining_probs = probs[remaining_indices]
            remaining_probs = remaining_probs / remaining_probs.sum()
            
            # Sample based on probabilities
            idx = np.random.choice(len(remaining_indices), p=remaining_probs.cpu().numpy())
            selected_idx = remaining_indices[idx]
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        
        # Return selected vehicles
        return [eligible_vehicles[i] for i in selected_indices]

def initialize_schedulers():
    """Initialize all schedulers for comparison"""
    schedulers = {}
    
    # Vehicle and state dimensions
    input_dim = 6  # model_version, sojourn_time, compute_capacity, data_quality, connectivity, type
    state_dim = 6  # current_model_performance, round_number, elapsed_time, scheduled_count, target_count, performance_gap
    
    # 1. Mamba Scheduler
    mamba_model_path = "mamba_scheduler_models/mamba_scheduler_final.pt"
    if os.path.exists(mamba_model_path):
        try:
            schedulers["Mamba"] = MambaSchedulerWrapper(mamba_model_path)
            print("Loaded Mamba scheduler from checkpoint")
        except Exception as e:
            print(f"Error loading Mamba model: {e}")
            print("Creating a new Mamba model for comparison")
            # Create a dummy model for demonstration
            actor = MambaActor(input_dim, state_dim, d_model=256, n_layers=3, d_state=16).to(device)
            dummy_model = {'actor_state_dict': actor.state_dict(), 'critic_state_dict': {}}
            os.makedirs(os.path.dirname(mamba_model_path), exist_ok=True)
            torch.save(dummy_model, mamba_model_path)
            schedulers["Mamba"] = MambaSchedulerWrapper(mamba_model_path)
    else:
        print("Mamba model checkpoint not found")
        print("Creating a new Mamba model for comparison")
        # Create a dummy model for demonstration
        actor = MambaActor(input_dim, state_dim, d_model=256, n_layers=3, d_state=16).to(device)
        dummy_model = {'actor_state_dict': actor.state_dict(), 'critic_state_dict': {}}
        os.makedirs(os.path.dirname(mamba_model_path), exist_ok=True)
        torch.save(dummy_model, mamba_model_path)
        schedulers["Mamba"] = MambaSchedulerWrapper(mamba_model_path)
    
    # 2. Transformer Scheduler
    transformer_model = TransformerScheduler(input_dim, state_dim).to(device)
    schedulers["Transformer"] = MLSchedulerWrapper(transformer_model)
    
    # 3. GNN Scheduler
    gnn_model = GNNScheduler(input_dim, state_dim).to(device)
    schedulers["GNN"] = MLSchedulerWrapper(gnn_model)
    
    # 4. LSTM Scheduler
    lstm_model = LSTMScheduler(input_dim, state_dim).to(device)
    schedulers["LSTM"] = MLSchedulerWrapper(lstm_model)
    
    # 5. Attention Scheduler
    attention_model = AttentionScheduler(input_dim, state_dim).to(device)
    schedulers["Attention"] = MLSchedulerWrapper(attention_model)
    
    # 6. Heuristic Scheduler
    heuristic_scheduler = HeuristicScheduler()
    
    # Create a wrapper for the heuristic scheduler
    class HeuristicSchedulerWrapper:
        def __init__(self, scheduler):
            self.scheduler = scheduler
            
        def select_vehicles(self, vehicles, target_count=10):
            # Convert environment vehicles to scheduler format
            scheduler_vehicles = []
            for v in vehicles:
                vehicle = Vehicle(
                    vehicle_id=v['vehicle_id'],
                    model_version=v['model_version'],
                    sojourn_time=v['sojourn_time'],
                    compute_capacity=v['compute_capacity'],
                    data_quality=v['data_quality'],
                    connectivity=v['connectivity'],
                    vehicle_type=v['vehicle_type']
                )
                vehicle.scheduled = v['scheduled']
                scheduler_vehicles.append(vehicle)
                
            # Select vehicles
            selected = self.scheduler.select_vehicles(scheduler_vehicles, target_count)
            
            # Convert back to environment format
            return [v for v in vehicles if v['vehicle_id'] in [sv.vehicle_id for sv in selected]]
    
    schedulers["Heuristic"] = HeuristicSchedulerWrapper(heuristic_scheduler)
    
    # 7. Random Scheduler (baseline)
    class RandomScheduler:
        def select_vehicles(self, vehicles, target_count=10):
            eligible_vehicles = [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
            if not eligible_vehicles:
                return []
            selected_count = min(target_count, len(eligible_vehicles))
            return random.sample(eligible_vehicles, selected_count)
    
    schedulers["Random"] = RandomScheduler()
    
    return schedulers

def run_comparison(num_episodes=5, max_rounds=100):
    """Run comparison between different schedulers"""
    # Initialize environment
    env = VehicleModelUpdateEnv(
        max_vehicles=1000,
        target_performance=0.95,
        max_rounds=max_rounds,
        arrival_rate=0.7
    )
    
    # Initialize schedulers
    schedulers = initialize_schedulers()
    
    # Run comparison
    results = {}
    
    for name, scheduler in schedulers.items():
        print(f"Evaluating {name} scheduler...")
        
        episode_rewards = []
        episode_performances = []
        episode_lengths = []
        decision_times = []
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            
            for round_num in range(max_rounds):
                # Get new vehicles
                new_vehicles = env.get_new_vehicles()
                
                # Make scheduling decision
                start_time = time.time()
                selected_vehicles = scheduler.select_vehicles(env.active_vehicles)
                decision_time = (time.time() - start_time) * 1000  # ms
                
                # Take step in environment
                selected_ids = [v['vehicle_id'] for v in selected_vehicles]
                next_state, reward, done, info = env.step(selected_ids)
                
                # Record metrics
                total_reward += reward
                decision_times.append(decision_time)
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            # Record episode metrics
            episode_rewards.append(total_reward)
            episode_performances.append(state['current_model_performance'])
            episode_lengths.append(state['current_round'])
            
            print(f"  Episode {episode+1}/{num_episodes}: Reward={total_reward:.2f}, "
                  f"Performance={state['current_model_performance']:.4f}, "
                  f"Length={state['current_round']}")
        
        # Compute average metrics
        results[name] = {
            'avg_reward': np.mean(episode_rewards),
            'avg_performance': np.mean(episode_performances),
            'avg_length': np.mean(episode_lengths),
            'avg_decision_time': np.mean(decision_times),
            'rewards': episode_rewards,
            'performances': episode_performances,
            'lengths': episode_lengths,
            'decision_times': decision_times
        }
        
        print(f"  Average Reward: {results[name]['avg_reward']:.2f}")
        print(f"  Average Performance: {results[name]['avg_performance']:.4f}")
        print(f"  Average Length: {results[name]['avg_length']:.1f}")
        print(f"  Average Decision Time: {results[name]['avg_decision_time']:.2f} ms")
        print()
    
    return results

def plot_results(results):
    """Plot comparison results"""
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot metrics
    metrics = ['avg_reward', 'avg_performance', 'avg_length', 'avg_decision_time']
    titles = ['Average Reward', 'Average Performance', 'Average Episode Length', 'Average Decision Time (ms)']
    ylabels = ['Reward', 'Performance', 'Rounds', 'Time (ms)']
    
    plt.figure(figsize=(15, 10))
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        plt.subplot(2, 2, i+1)
        
        values = [results[name][metric] for name in results]
        plt.bar(results.keys(), values)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        
        # Add values on top of bars
        for j, v in enumerate(values):
            plt.text(j, v, f"{v:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scheduler_comparison_{timestamp}.png"))
    
    # Plot performance over time for each scheduler
    plt.figure(figsize=(10, 6))
    for name in results:
        plt.plot(results[name]['performances'], label=name)
    
    plt.title('Model Performance by Episode')
    plt.xlabel('Episode')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"performance_comparison_{timestamp}.png"))
    
    # Plot decision time distribution
    plt.figure(figsize=(10, 6))
    plt.boxplot([results[name]['decision_times'] for name in results], labels=list(results.keys()))
    plt.title('Decision Time Distribution')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, f"decision_time_comparison_{timestamp}.png"))
    
    # Save results to CSV
    import pandas as pd
    summary = {name: {
        'Avg Reward': results[name]['avg_reward'],
        'Avg Performance': results[name]['avg_performance'],
        'Avg Episode Length': results[name]['avg_length'],
        'Avg Decision Time (ms)': results[name]['avg_decision_time']
    } for name in results}
    
    df = pd.DataFrame(summary).T
    df.to_csv(os.path.join(output_dir, f"scheduler_comparison_{timestamp}.csv"))
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    print("Starting scheduler comparison...")
    
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run comparison
    results = run_comparison(num_episodes=50, max_rounds=100)
    
    # Plot results
    plot_results(results)
    
    print("Comparison complete!")
