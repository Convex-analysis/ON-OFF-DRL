# -*- coding: utf-8 -*-
"""
Mobility Residient Vehicular Federated Learning (MR-VFL) Environment
This environment simulates vehicles participating in federated learning with fairness constraints.
"""

import numpy as np
import random
from collections import deque

class VehicularFLEnv:
    """
    Environment for MR-VFL scheduler with fairness constraints.
    Simulates vehicle arrivals, departures, and federated learning rounds.
    """
    def __init__(self, vehicle_count=30, max_round=100, sync_limit=1000):
        self.vehicle_count = vehicle_count
        self.max_round = max_round
        self.sync_limit = sync_limit
        self.current_round = 0
        self.selection_history = np.zeros(vehicle_count)  # γ_v^t tracking
        self.fairness_threshold = 50  # Γ_t

        # State variables
        self.vehicles = []
        self.current_model_performance = 0.0
        self.elapsed_time = 0.0
        self.scheduled_count = 0

        # Performance tracking
        self.performance_history = []
        self.reward_history = []

    def reset(self):
        """Reset the environment to initial state"""
        self.current_round = 0
        self.selection_history = np.zeros(self.vehicle_count)
        self.current_model_performance = 0.0
        self.elapsed_time = 0.0
        self.scheduled_count = 0
        self.performance_history = []
        self.reward_history = []

        # Initialize vehicle states
        self.vehicles = self._generate_vehicles()

        return self._get_state()

    def _generate_vehicles(self):
        """Generate vehicles with random attributes"""
        vehicles = []
        for i in range(self.vehicle_count):
            vehicle = {
                'id': i,
                'computation_capacity': np.random.uniform(0.5, 2.0),
                'data_quality': np.random.uniform(0.1, 1.0),
                'data_size': np.random.randint(100, 1000),
                'sojourn_time': np.random.uniform(100, 2000),
                'arrival_time': np.random.uniform(0, self.sync_limit/2),
                'departure_time': 0,  # Will be computed based on arrival and sojourn
                'channel_gain': np.random.uniform(0.1, 1.0),
                'scheduled': False
            }
            vehicle['departure_time'] = vehicle['arrival_time'] + vehicle['sojourn_time']
            vehicles.append(vehicle)
        return vehicles

    def step(self, action):
        """
        Take a step in the environment by scheduling selected vehicles with parameters

        Args:
            action: [vehicle_idx, amplification_factor, scheduled_time, bandwidth]

        Returns:
            state: Current environment state
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Parse action
        vehicle_idx, alpha, scheduled_time, bandwidth = action

        # Apply action and calculate reward
        reward = self._calculate_reward(vehicle_idx, alpha, scheduled_time, bandwidth)

        # Update selection history
        self.selection_history[vehicle_idx] += 1

        # Mark vehicle as scheduled
        self.vehicles[vehicle_idx]['scheduled'] = True
        self.scheduled_count += 1

        # Update time
        self.elapsed_time = max(self.elapsed_time, scheduled_time)

        # Check if fairness constraint is violated
        # Use clipping to prevent overflow
        clipped_history = np.clip(self.selection_history, 0, 10)  # Limit exponential growth
        fairness_score = np.sum(np.exp(clipped_history))
        fairness_violation = fairness_score > self.fairness_threshold

        # Adjust reward if fairness constraint is violated
        if fairness_violation:
            fairness_penalty = 0.5 * min(10, (fairness_score - self.fairness_threshold))  # Limit penalty more aggressively
            reward -= fairness_penalty

        # Limit minimum reward to prevent extremely negative values
        reward = max(-100, reward)

        # Update model performance based on scheduled vehicle
        accuracy_improvement = 0.05 * self.vehicles[vehicle_idx]['data_quality']
        self.current_model_performance += accuracy_improvement
        self.current_model_performance = min(1.0, self.current_model_performance)

        self.performance_history.append(self.current_model_performance)
        self.reward_history.append(reward)

        # Check if round is complete
        done = self._is_round_complete()
        if done:
            self.current_round += 1

        # Get next state
        next_state = self._get_state()

        # Additional info
        info = {
            'fairness_score': fairness_score,
            'fairness_violation': fairness_violation,
            'accuracy_improvement': accuracy_improvement,
            'performance_history': self.performance_history,
            'reward_history': self.reward_history,
            'scheduled_count': self.scheduled_count
        }

        return next_state, reward, done, info

    def _calculate_reward(self, vehicle_idx, alpha, scheduled_time, bandwidth):
        """Calculate reward based on scheduled vehicle and parameters"""
        vehicle = self.vehicles[vehicle_idx]

        # Time efficiency component
        time_efficiency = self.sync_limit - scheduled_time

        # Accuracy improvement component (based on data quality)
        accuracy_gain = 0.05 * vehicle['data_quality']

        # Communication efficiency (based on bandwidth allocation)
        comm_efficiency = bandwidth * vehicle['channel_gain']

        # Computation efficiency (based on amplification factor and computation capacity)
        comp_efficiency = alpha * vehicle['computation_capacity']

        # Fairness penalty based on selection frequency
        # Use clipping to prevent overflow and limit maximum penalty
        clipped_count = min(5, self.selection_history[vehicle_idx])  # Limit exponential growth more aggressively
        fairness_penalty = min(100, np.exp(clipped_count) - 1)  # Cap maximum penalty

        # Combined reward with dual-timescale optimization
        # Short-term efficiency (immediate training efficiency)
        short_term = 0.3 * time_efficiency/self.sync_limit + 0.2 * comm_efficiency

        # Long-term performance (model accuracy improvement)
        long_term = 0.4 * accuracy_gain + 0.2 * comp_efficiency

        # Combined reward with fairness penalty
        reward = short_term + long_term - 0.1 * fairness_penalty

        return reward

    def _get_state(self):
        """Get current environment state"""
        # Create state representation including available vehicles and their attributes
        state = []
        for i, vehicle in enumerate(self.vehicles):
            if self._is_vehicle_available(vehicle):
                vehicle_state = [
                    vehicle['computation_capacity'],
                    vehicle['data_quality'],
                    vehicle['data_size'],
                    vehicle['sojourn_time'],
                    vehicle['channel_gain'],
                    self.selection_history[i]  # Include selection history in state
                ]
                state.append(vehicle_state)

        # Padding to fixed length for batch processing
        while len(state) < self.vehicle_count:
            state.append([0, 0, 0, 0, 0, 0])  # Zero padding for unavailable vehicles

        return np.array(state)

    def _is_vehicle_available(self, vehicle):
        """Check if vehicle is currently available"""
        current_time = self.elapsed_time
        return (vehicle['arrival_time'] <= current_time <= vehicle['departure_time']) and not vehicle['scheduled']

    def _is_round_complete(self):
        """Check if enough vehicles have been selected or time limit reached"""
        selected_count = np.sum(self.selection_history > 0)
        step_count = np.sum(self.selection_history)  # Total steps taken

        # Add a maximum step limit to prevent infinite loops
        max_steps_per_round = 100

        return (selected_count >= 20 or  # Required number of vehicles
                self.elapsed_time >= self.sync_limit or  # Time limit reached
                self.current_round >= self.max_round or  # Max rounds reached
                step_count >= max_steps_per_round)  # Max steps per round reached

# Example usage
if __name__ == "__main__":
    env = VehicularFLEnv()
    state = env.reset()

    # Simple random policy for testing
    done = False
    total_reward = 0

    while not done:
        # Get available vehicles
        available_indices = []
        for i, vehicle in enumerate(env.vehicles):
            if env._is_vehicle_available(vehicle):
                available_indices.append(i)

        if available_indices:
            # Randomly select a vehicle and parameters
            vehicle_idx = random.choice(available_indices)
            alpha = random.uniform(0.5, 2.0)  # Amplification factor
            scheduled_time = env.elapsed_time + random.uniform(10, 50)  # Scheduled time
            bandwidth = random.uniform(0.1, 1.0)  # Bandwidth allocation

            action = [vehicle_idx, alpha, scheduled_time, bandwidth]
        else:
            # No available vehicles, use dummy action
            action = [0, 1.0, env.elapsed_time + 10, 0.5]

        # Take step in environment
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(f"Round {env.current_round}, Vehicle: {action[0]}, "
              f"Performance: {env.current_model_performance:.4f}, Reward: {reward:.4f}")

        # Update state
        state = next_state

    print(f"Episode finished. Total reward: {total_reward:.4f}, "
          f"Final performance: {env.current_model_performance:.4f}")
