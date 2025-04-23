# -*- coding: utf-8 -*-
"""
Production deployment script for Mamba-Based Actor-Critic Scheduler
This script demonstrates how to use the trained scheduler in a production environment.
"""

import os
import torch
import numpy as np
import time
import json
from datetime import datetime
from mamba_scheduler import OptimizedMambaScheduler, Vehicle, GlobalState

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class VehicleSchedulerService:
    """Service for scheduling vehicles for federated learning updates"""
    def __init__(self, model_path, log_dir="scheduler_logs"):
        self.scheduler = OptimizedMambaScheduler(model_path)
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Initialize state
        self.global_state = GlobalState()
        self.vehicle_cache = {}  # vehicle_id -> Vehicle
        self.scheduled_vehicles = set()

        # Initialize metrics
        self.metrics = {
            "rounds": 0,
            "total_vehicles_processed": 0,
            "total_vehicles_scheduled": 0,
            "avg_decision_time_ms": 0,
            "current_model_performance": 0.0,
            "start_time": datetime.now().isoformat()
        }

        print(f"Scheduler service initialized with model: {model_path}")

    def process_new_vehicle(self, vehicle_data):
        """Process a new vehicle arrival"""
        # Create Vehicle object
        vehicle = Vehicle(
            vehicle_id=vehicle_data["vehicle_id"],
            model_version=vehicle_data["model_version"],
            sojourn_time=vehicle_data["sojourn_time"],
            compute_capacity=vehicle_data["compute_capacity"],
            data_quality=vehicle_data["data_quality"],
            connectivity=vehicle_data["connectivity"],
            vehicle_type=vehicle_data["vehicle_type"]
        )

        # Add to cache
        self.vehicle_cache[vehicle.vehicle_id] = vehicle

        # Update metrics
        self.metrics["total_vehicles_processed"] += 1

        return {"status": "success", "vehicle_id": vehicle.vehicle_id}

    def make_scheduling_decision(self, target_count=10):
        """Make a scheduling decision for the current round"""
        # Get available vehicles (not already scheduled)
        available_vehicles = [v for v in self.vehicle_cache.values()
                             if v.vehicle_id not in self.scheduled_vehicles]

        # Start timing
        start_time = time.time()

        if not available_vehicles:
            # Calculate decision time even when no vehicles are available
            decision_time_ms = (time.time() - start_time) * 1000

            # Return a response with all expected keys
            return {
                "status": "no_vehicles_available",
                "selected_vehicles": [],
                "decision_time_ms": decision_time_ms,
                "current_performance": self.global_state.current_model_performance
            }

        # Make scheduling decision
        with torch.no_grad():
            selected_vehicles = self.scheduler.inference_mode_scheduling(
                available_vehicles, self.global_state, target_count
            )

        # Update scheduled set
        for vehicle in selected_vehicles:
            self.scheduled_vehicles.add(vehicle.vehicle_id)

        # Update global state
        self.global_state.scheduled_count += len(selected_vehicles)
        self.global_state.round_number += 1
        self.metrics["rounds"] = self.global_state.round_number

        # Update model performance (simplified simulation)
        if selected_vehicles:
            # Simple model: performance increases based on data quality and compute capacity
            avg_quality = sum(v.data_quality for v in selected_vehicles) / len(selected_vehicles)
            avg_compute = sum(v.compute_capacity for v in selected_vehicles) / len(selected_vehicles)

            # Performance increase with diminishing returns
            diminishing_factor = 1.0 - self.global_state.current_model_performance
            performance_increase = 0.01 * avg_quality * avg_compute * diminishing_factor * len(selected_vehicles) / 10

            self.global_state.current_model_performance += performance_increase
            self.global_state.current_model_performance = min(1.0, self.global_state.current_model_performance)
            self.global_state.performance_gap = max(0, 1.0 - self.global_state.current_model_performance)

            self.metrics["current_model_performance"] = self.global_state.current_model_performance

        # Calculate decision time if not already calculated
        if 'decision_time_ms' not in locals():
            decision_time_ms = (time.time() - start_time) * 1000

        # Update metrics
        self.metrics["total_vehicles_scheduled"] += len(selected_vehicles)

        # Update average decision time with exponential moving average
        if self.metrics["avg_decision_time_ms"] == 0:
            self.metrics["avg_decision_time_ms"] = decision_time_ms
        else:
            self.metrics["avg_decision_time_ms"] = 0.9 * self.metrics["avg_decision_time_ms"] + 0.1 * decision_time_ms

        # Log decision
        self._log_decision(selected_vehicles, decision_time_ms)

        # Return selected vehicle IDs
        return {
            "status": "success",
            "selected_vehicles": [v.vehicle_id for v in selected_vehicles],
            "decision_time_ms": decision_time_ms,
            "current_performance": self.global_state.current_model_performance
        }

    def remove_vehicle(self, vehicle_id):
        """Remove a vehicle from the system (e.g., when it departs)"""
        if vehicle_id in self.vehicle_cache:
            del self.vehicle_cache[vehicle_id]
            if vehicle_id in self.scheduled_vehicles:
                self.scheduled_vehicles.remove(vehicle_id)
            return {"status": "success", "vehicle_id": vehicle_id}
        else:
            return {"status": "error", "message": f"Vehicle {vehicle_id} not found"}

    def get_metrics(self):
        """Get current metrics"""
        # Add real-time metrics
        self.metrics["active_vehicles"] = len(self.vehicle_cache)
        self.metrics["scheduled_vehicles"] = len(self.scheduled_vehicles)
        self.metrics["available_vehicles"] = len(self.vehicle_cache) - len(self.scheduled_vehicles)

        # Calculate uptime
        start_time = datetime.fromisoformat(self.metrics["start_time"])
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        self.metrics["uptime_seconds"] = uptime_seconds

        return self.metrics

    def _log_decision(self, selected_vehicles, decision_time_ms):
        """Log scheduling decision"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "round": self.global_state.round_number,
            "selected_count": len(selected_vehicles),
            "decision_time_ms": decision_time_ms,
            "model_performance": self.global_state.current_model_performance,
            "selected_vehicles": [v.vehicle_id for v in selected_vehicles]
        }

        # Write to log file
        log_file = os.path.join(self.log_dir, f"scheduler_log_{datetime.now().strftime('%Y%m%d')}.jsonl")
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

def simulate_vehicle_arrivals(count=100):
    """Simulate vehicle arrivals for testing"""
    vehicles = []
    for i in range(count):
        vehicle = {
            "vehicle_id": i,
            "model_version": np.random.uniform(0.1, 1.0),
            "sojourn_time": np.random.uniform(1.0, 10.0),
            "compute_capacity": np.random.uniform(0.1, 1.0),
            "data_quality": np.random.uniform(0.3, 1.0),
            "connectivity": np.random.uniform(0.5, 1.0),
            "vehicle_type": np.random.randint(0, 4)
        }
        vehicles.append(vehicle)
    return vehicles

def run_simulation(model_path, num_rounds=10):
    """Run a simulation of the scheduler service"""
    # Initialize service
    service = VehicleSchedulerService(model_path)

    # Simulate initial vehicle arrivals
    initial_vehicles = simulate_vehicle_arrivals(30)
    for vehicle in initial_vehicles:
        service.process_new_vehicle(vehicle)

    print(f"Initialized with {len(initial_vehicles)} vehicles")

    # Run scheduling rounds
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        # Simulate new vehicle arrivals
        if np.random.random() < 0.7:  # 70% chance of new arrivals
            new_count = np.random.randint(1, 6)
            new_vehicles = simulate_vehicle_arrivals(new_count)
            for vehicle in new_vehicles:
                service.process_new_vehicle(vehicle)
            print(f"Processed {new_count} new vehicle arrivals")

        # Make scheduling decision
        result = service.make_scheduling_decision(target_count=10)

        print(f"Selected {len(result['selected_vehicles'])} vehicles")
        print(f"Decision time: {result['decision_time_ms']:.2f} ms")
        print(f"Current model performance: {result['current_performance']:.4f}")

        # Simulate vehicle departures
        if np.random.random() < 0.3:  # 30% chance of departures
            metrics = service.get_metrics()
            if metrics['active_vehicles'] > 0:
                depart_count = np.random.randint(1, max(2, metrics['active_vehicles'] // 5))
                vehicle_ids = list(service.vehicle_cache.keys())
                depart_ids = np.random.choice(vehicle_ids, size=min(depart_count, len(vehicle_ids)), replace=False)

                for vid in depart_ids:
                    service.remove_vehicle(vid)

                print(f"Processed {len(depart_ids)} vehicle departures")

        # Print current metrics
        if round_num % 5 == 0:
            metrics = service.get_metrics()
            print("\nCurrent Metrics:")
            print(f"Active Vehicles: {metrics['active_vehicles']}")
            print(f"Scheduled Vehicles: {metrics['scheduled_vehicles']}")
            print(f"Available Vehicles: {metrics['available_vehicles']}")
            print(f"Avg Decision Time: {metrics['avg_decision_time_ms']:.2f} ms")
            print(f"Model Performance: {metrics['current_model_performance']:.4f}")

        # Small delay between rounds
        time.sleep(0.5)

    # Final metrics
    final_metrics = service.get_metrics()
    print("\n=== Final Metrics ===")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Path to trained model
    model_path = "mamba_scheduler_models/mamba_scheduler_final.pt"

    # Check if model exists and is valid, if not, create a dummy model
    create_dummy = False
    if not os.path.exists(model_path):
        create_dummy = True
        print(f"Model file {model_path} not found.")
    else:
        # Try to load the model to verify it's valid
        try:
            # Just check if we can load it
            test_load = torch.load(model_path, weights_only=True)
            if 'actor_state_dict' not in test_load:
                create_dummy = True
                print(f"Model file {model_path} exists but doesn't contain expected data.")
        except Exception as e:
            create_dummy = True
            print(f"Error loading model file: {e}")

    if create_dummy:
        print("Please train the model first using train_mamba_scheduler.py")
        print("For demonstration purposes, we'll simulate with a mock model.")

        # Create a directory for the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Create a dummy model with proper structure
        try:
            # Initialize a temporary model to get the state dict structure
            from mamba_scheduler import MambaActor
            temp_actor = MambaActor(6, 6, d_model=256, n_layers=3, d_state=16)
            dummy_model = {
                'actor_state_dict': temp_actor.state_dict(),
                'critic_state_dict': {}
            }
        except Exception as e:
            print(f"Could not create proper dummy model: {e}")
            print("Creating a simple placeholder model instead.")
            dummy_model = {
                'actor_state_dict': {},
                'critic_state_dict': {}
            }
        # Create a dummy model file for demonstration
        torch.save(dummy_model, model_path)
        print(f"Created dummy model at {model_path} for demonstration")

    # Run simulation
    print("Starting scheduler service simulation...")
    run_simulation(model_path, num_rounds=20)
