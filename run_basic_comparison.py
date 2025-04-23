# -*- coding: utf-8 -*-
"""
Basic comparison script for evaluating scheduler methods
This version doesn't require the Mamba model to be installed
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from datetime import datetime
from vehicle_env import VehicleModelUpdateEnv

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = "scheduler_comparison_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a simple vehicle class for the schedulers
class SimpleVehicle:
    def __init__(self, vehicle_id, model_version, sojourn_time, compute_capacity, 
                 data_quality, connectivity, vehicle_type, scheduled=False):
        self.vehicle_id = vehicle_id
        self.model_version = model_version
        self.sojourn_time = sojourn_time
        self.compute_capacity = compute_capacity
        self.data_quality = data_quality
        self.connectivity = connectivity
        self.vehicle_type = vehicle_type
        self.scheduled = scheduled
        
    @classmethod
    def from_env_vehicle(cls, env_vehicle):
        return cls(
            vehicle_id=env_vehicle['vehicle_id'],
            model_version=env_vehicle['model_version'],
            sojourn_time=env_vehicle['sojourn_time'],
            compute_capacity=env_vehicle['compute_capacity'],
            data_quality=env_vehicle['data_quality'],
            connectivity=env_vehicle['connectivity'],
            vehicle_type=env_vehicle['vehicle_type'],
            scheduled=env_vehicle['scheduled']
        )

# Define scheduler implementations
class RandomScheduler:
    """Random scheduler (baseline)"""
    def select_vehicles(self, vehicles, target_count=10):
        eligible_vehicles = [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
        if not eligible_vehicles:
            return []
        selected_count = min(target_count, len(eligible_vehicles))
        return random.sample(eligible_vehicles, selected_count)

class QualityScheduler:
    """Scheduler that prioritizes data quality"""
    def select_vehicles(self, vehicles, target_count=10):
        eligible_vehicles = [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
        if not eligible_vehicles:
            return []
        
        # Sort by data quality
        sorted_vehicles = sorted(eligible_vehicles, key=lambda v: v['data_quality'], reverse=True)
        
        # Select top vehicles
        selected_count = min(target_count, len(eligible_vehicles))
        return sorted_vehicles[:selected_count]

class ComputeScheduler:
    """Scheduler that prioritizes compute capacity"""
    def select_vehicles(self, vehicles, target_count=10):
        eligible_vehicles = [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
        if not eligible_vehicles:
            return []
        
        # Sort by compute capacity
        sorted_vehicles = sorted(eligible_vehicles, key=lambda v: v['compute_capacity'], reverse=True)
        
        # Select top vehicles
        selected_count = min(target_count, len(eligible_vehicles))
        return sorted_vehicles[:selected_count]

class SojournScheduler:
    """Scheduler that prioritizes sojourn time"""
    def select_vehicles(self, vehicles, target_count=10):
        eligible_vehicles = [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
        if not eligible_vehicles:
            return []
        
        # Sort by sojourn time
        sorted_vehicles = sorted(eligible_vehicles, key=lambda v: v['sojourn_time'], reverse=True)
        
        # Select top vehicles
        selected_count = min(target_count, len(eligible_vehicles))
        return sorted_vehicles[:selected_count]

class WeightedScheduler:
    """Scheduler that uses weighted combination of features"""
    def __init__(self, weights=None):
        # Default weights if not provided
        if weights is None:
            self.weights = {
                'data_quality': 0.3,
                'compute_capacity': 0.3,
                'connectivity': 0.2,
                'sojourn_time': 0.1,
                'model_version': 0.1
            }
        else:
            self.weights = weights
    
    def select_vehicles(self, vehicles, target_count=10):
        eligible_vehicles = [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
        if not eligible_vehicles:
            return []
        
        # Calculate scores
        scores = []
        for v in eligible_vehicles:
            score = (
                self.weights['data_quality'] * v['data_quality'] +
                self.weights['compute_capacity'] * v['compute_capacity'] +
                self.weights['connectivity'] * v['connectivity'] +
                self.weights['sojourn_time'] * min(1.0, v['sojourn_time'] / 10.0) +
                self.weights['model_version'] * (1.0 - v['model_version'])  # Lower version = higher priority
            )
            scores.append((v, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top vehicles
        selected_count = min(target_count, len(eligible_vehicles))
        return [item[0] for item in scores[:selected_count]]

class DiversityScheduler:
    """Scheduler that prioritizes diversity in vehicle types"""
    def select_vehicles(self, vehicles, target_count=10):
        eligible_vehicles = [v for v in vehicles if not v['scheduled'] and v['sojourn_time'] >= 1.0]
        if not eligible_vehicles:
            return []
        
        # Group by vehicle type
        type_groups = {}
        for v in eligible_vehicles:
            vtype = v['vehicle_type']
            if vtype not in type_groups:
                type_groups[vtype] = []
            type_groups[vtype].append(v)
        
        # Sort each group by data quality
        for vtype in type_groups:
            type_groups[vtype].sort(key=lambda v: v['data_quality'], reverse=True)
        
        # Select vehicles in a round-robin fashion from each type
        selected_vehicles = []
        while len(selected_vehicles) < target_count and any(len(g) > 0 for g in type_groups.values()):
            for vtype in sorted(type_groups.keys()):
                if len(type_groups[vtype]) > 0:
                    selected_vehicles.append(type_groups[vtype].pop(0))
                    if len(selected_vehicles) >= target_count:
                        break
        
        return selected_vehicles

def run_comparison(num_episodes=5, max_rounds=100):
    """Run comparison between different schedulers"""
    # Initialize environment
    env = VehicleModelUpdateEnv(
        max_vehicles=100,
        target_performance=0.95,
        max_rounds=max_rounds,
        arrival_rate=0.7
    )
    
    # Initialize schedulers
    schedulers = {
        "Random": RandomScheduler(),
        "Quality": QualityScheduler(),
        "Compute": ComputeScheduler(),
        "Sojourn": SojournScheduler(),
        "Weighted": WeightedScheduler(),
        "Diversity": DiversityScheduler()
    }
    
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
                env.get_new_vehicles()
                
                # Make scheduling decision
                start_time = time.time()
                selected_vehicles = scheduler.select_vehicles(env.active_vehicles)
                decision_time = (time.time() - start_time) * 1000  # ms
                
                # Take step in environment
                selected_ids = [v['vehicle_id'] for v in selected_vehicles]
                next_state, reward, done, _ = env.step(selected_ids)
                
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
    try:
        import pandas as pd
        summary = {name: {
            'Avg Reward': results[name]['avg_reward'],
            'Avg Performance': results[name]['avg_performance'],
            'Avg Episode Length': results[name]['avg_length'],
            'Avg Decision Time (ms)': results[name]['avg_decision_time']
        } for name in results}
        
        df = pd.DataFrame(summary).T
        df.to_csv(os.path.join(output_dir, f"scheduler_comparison_{timestamp}.csv"))
    except ImportError:
        print("pandas not installed, skipping CSV export")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    print("Starting basic scheduler comparison...")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run comparison
    results = run_comparison(num_episodes=5, max_rounds=100)
    
    # Plot results
    plot_results(results)
    
    print("Comparison complete!")
