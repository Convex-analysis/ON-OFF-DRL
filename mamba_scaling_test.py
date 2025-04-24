# -*- coding: utf-8 -*-
"""
Scaling test to demonstrate Mamba scheduler's advantages over Transformer
This script specifically tests how decision time scales with increasing vehicle counts
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from datetime import datetime
import gc  # Garbage collection for clean measurements

# Import schedulers
from mamba_scheduler import MambaActor, OptimizedMambaScheduler, Vehicle, GlobalState
from scheduler_comparison_methods import TransformerScheduler, GNNScheduler, LSTMScheduler

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Enable cuDNN benchmarking for better performance
torch.backends.cudnn.benchmark = True

# Create output directory
output_dir = "scaling_test_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_random_vehicle(vehicle_id):
    """Generate a random vehicle for testing"""
    return {
        'vehicle_id': vehicle_id,
        'model_version': random.uniform(0.1, 1.0),
        'sojourn_time': random.uniform(1.0, 10.0),
        'compute_capacity': random.uniform(0.1, 1.0),
        'data_quality': random.uniform(0.3, 1.0),
        'connectivity': random.uniform(0.5, 1.0),
        'vehicle_type': random.randint(0, 3),
        'scheduled': False
    }

def convert_to_scheduler_vehicle(env_vehicle):
    """Convert environment vehicle to scheduler vehicle"""
    return Vehicle(
        vehicle_id=env_vehicle['vehicle_id'],
        model_version=env_vehicle['model_version'],
        sojourn_time=env_vehicle['sojourn_time'],
        compute_capacity=env_vehicle['compute_capacity'],
        data_quality=env_vehicle['data_quality'],
        connectivity=env_vehicle['connectivity'],
        vehicle_type=env_vehicle['vehicle_type']
    )

class MLSchedulerWrapper:
    """Wrapper for ML-based schedulers to match comparison interface"""
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

class MambaSchedulerWrapper:
    """Wrapper class for Mamba scheduler to match comparison interface"""
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

        # Make scheduling decision
        selected_vehicles = self.scheduler.inference_mode_scheduling(
            scheduler_vehicles, self.global_state, target_count
        )

        # Convert back to environment vehicle format
        return [v for v in vehicles if v['vehicle_id'] in [sv.vehicle_id for sv in selected_vehicles]]

def initialize_schedulers():
    """Initialize schedulers for comparison"""
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
            print("Skipping Mamba scheduler")
    else:
        print("Mamba model checkpoint not found")
        print("Skipping Mamba scheduler")

    # 2. Transformer Scheduler
    transformer_model = TransformerScheduler(input_dim, state_dim).to(device)
    schedulers["Transformer"] = MLSchedulerWrapper(transformer_model)

    # 3. LSTM Scheduler (for comparison with another O(n) method)
    lstm_model = LSTMScheduler(input_dim, state_dim).to(device)
    schedulers["LSTM"] = MLSchedulerWrapper(lstm_model)

    return schedulers

def run_scaling_test():
    """Run scaling test to compare decision time with increasing vehicle counts"""
    print("\n=== Running Scaling Test ===")

    # Initialize schedulers
    schedulers = initialize_schedulers()

    # Define vehicle counts to test
    vehicle_counts = [10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]

    # Store results
    results = {name: [] for name in schedulers}

    # Run test for each scheduler and vehicle count
    for count in vehicle_counts:
        print(f"\nTesting with {count} vehicles:")

        # Generate test vehicles
        vehicles = [generate_random_vehicle(i) for i in range(count)]

        for name, scheduler in schedulers.items():
            # Clear GPU cache before each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Warm-up run (not measured)
            _ = scheduler.select_vehicles(vehicles, target_count=10)

            # Actual test runs
            times = []
            for _ in range(5):  # Run 5 times and take average
                # Force garbage collection
                gc.collect()

                # Measure decision time
                import time as time_module  # Avoid variable shadowing
                start_time = time_module.time()
                _ = scheduler.select_vehicles(vehicles, target_count=10)
                decision_time = (time_module.time() - start_time) * 1000  # ms
                times.append(decision_time)

            # Record average time
            avg_time = np.mean(times)
            results[name].append(avg_time)
            print(f"  {name}: {avg_time:.2f} ms")

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot 1: Decision Time vs Vehicle Count (linear scale)
    plt.subplot(2, 1, 1)
    for name, times in results.items():
        if name == "Mamba":
            plt.plot(vehicle_counts, times, 'o-', linewidth=2, label=name, color='green')
        elif name == "Transformer":
            plt.plot(vehicle_counts, times, 's--', linewidth=2, label=name, color='red')
        else:
            plt.plot(vehicle_counts, times, 'x-.', linewidth=1.5, label=name, color='blue')

    plt.title('Decision Time Scaling with Vehicle Count (Linear Scale)')
    plt.xlabel('Number of Vehicles')
    plt.ylabel('Decision Time (ms)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Decision Time vs Vehicle Count (log-log scale to show complexity)
    plt.subplot(2, 1, 2)
    for name, times in results.items():
        if name == "Mamba":
            plt.loglog(vehicle_counts, times, 'o-', linewidth=2, label=name, color='green')
        elif name == "Transformer":
            plt.loglog(vehicle_counts, times, 's--', linewidth=2, label=name, color='red')
        else:
            plt.loglog(vehicle_counts, times, 'x-.', linewidth=1.5, label=name, color='blue')

    # Add reference lines for O(n) and O(n²) complexity
    x_ref = np.array(vehicle_counts)

    # Find scaling factors to match the curves
    if "Mamba" in results and "Transformer" in results:
        # Use middle point to calibrate
        mid_idx = len(vehicle_counts) // 2
        mamba_scale = results["Mamba"][mid_idx] / vehicle_counts[mid_idx]
        transformer_scale = results["Transformer"][mid_idx] / (vehicle_counts[mid_idx]**2)

        # Plot reference lines
        plt.loglog(x_ref, mamba_scale * x_ref, 'k--', alpha=0.5, label='O(n) reference')
        plt.loglog(x_ref, transformer_scale * x_ref**2, 'k:', alpha=0.5, label='O(n²) reference')

    plt.title('Decision Time Scaling (Log-Log Scale)')
    plt.xlabel('Number of Vehicles (log scale)')
    plt.ylabel('Decision Time (ms) (log scale)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f"scaling_test_{timestamp}.png"))

    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame({
        'Vehicle Count': vehicle_counts,
        **{name: results[name] for name in schedulers}
    })
    df.to_csv(os.path.join(output_dir, f"scaling_test_{timestamp}.csv"), index=False)

    print(f"\nScaling test results saved to {output_dir}")

    # Calculate and display complexity analysis
    print("\n=== Complexity Analysis ===")
    for name in schedulers:
        # Use log-log regression to estimate complexity
        if len(results[name]) > 2:  # Need at least 3 points for meaningful regression
            x = np.log(vehicle_counts)
            y = np.log(results[name])

            # Linear regression on log-log data
            coeffs = np.polyfit(x, y, 1)
            complexity = coeffs[0]  # Slope in log-log is the power in the complexity

            print(f"{name} estimated complexity: O(n^{complexity:.2f})")

            # Theoretical vs. measured
            if name == "Mamba":
                print(f"  Theoretical: O(n), Measured: O(n^{complexity:.2f})")
            elif name == "Transformer":
                print(f"  Theoretical: O(n²), Measured: O(n^{complexity:.2f})")
            else:
                print(f"  Measured: O(n^{complexity:.2f})")

    return results

def run_memory_test():
    """Test memory usage of different schedulers with increasing vehicle counts"""
    print("\n=== Running Memory Usage Test ===")

    # Skip if torch.cuda.memory_stats is not available
    if not hasattr(torch.cuda, 'memory_stats'):
        print("Memory usage test requires PyTorch with CUDA memory stats support")
        return {}

    # Initialize schedulers
    schedulers = initialize_schedulers()

    # Define vehicle counts to test
    vehicle_counts = [100, 500, 1000, 2000, 5000]

    # Store results
    memory_results = {name: [] for name in schedulers}

    # Run test for each scheduler and vehicle count
    for count in vehicle_counts:
        print(f"\nTesting memory usage with {count} vehicles:")

        # Generate test vehicles
        vehicles = [generate_random_vehicle(i) for i in range(count)]

        for name, scheduler in schedulers.items():
            # Clear GPU cache before each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Run scheduler
            _ = scheduler.select_vehicles(vehicles, target_count=10)

            # Measure peak memory usage
            if torch.cuda.is_available():
                memory_stats = torch.cuda.memory_stats()
                peak_memory = memory_stats["allocated_bytes.all.peak"] / (1024 * 1024)  # Convert to MB
                memory_results[name].append(peak_memory)
                print(f"  {name}: {peak_memory:.2f} MB")

    # Plot results if we have data
    if any(memory_results.values()):
        plt.figure(figsize=(10, 6))

        for name, memory in memory_results.items():
            if memory:  # Check if we have data
                if name == "Mamba":
                    plt.plot(vehicle_counts[:len(memory)], memory, 'o-', linewidth=2, label=name, color='green')
                elif name == "Transformer":
                    plt.plot(vehicle_counts[:len(memory)], memory, 's--', linewidth=2, label=name, color='red')
                else:
                    plt.plot(vehicle_counts[:len(memory)], memory, 'x-.', linewidth=1.5, label=name, color='blue')

        plt.title('Memory Usage Scaling with Vehicle Count')
        plt.xlabel('Number of Vehicles')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.legend()
        plt.grid(True)

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(output_dir, f"memory_test_{timestamp}.png"))

        print(f"\nMemory test results saved to {output_dir}")

    return memory_results

if __name__ == "__main__":
    print("Starting Mamba scaling advantage tests...")

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Run scaling test
    scaling_results = run_scaling_test()

    # Run memory test if on CUDA
    if torch.cuda.is_available():
        memory_results = run_memory_test()

    print("\nTests complete!")
