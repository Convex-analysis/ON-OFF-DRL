# -*- coding: utf-8 -*-
"""
Vehicle Model Update Environment for Federated Learning
This environment simulates vehicles arriving and participating in federated learning rounds.
"""

import numpy as np
import random
from collections import deque
import torch

class VehicleModelUpdateEnv:
    """
    Environment for vehicle model update scheduling in federated learning.
    Simulates vehicle arrivals, departures, and federated learning rounds.
    """
    def __init__(self,
                 max_vehicles=100,
                 target_performance=0.95,
                 max_rounds=100,
                 arrival_rate=0.7,
                 min_vehicles_per_round=5,
                 max_vehicles_per_round=20):

        self.max_vehicles = max_vehicles
        self.target_performance = target_performance
        self.max_rounds = max_rounds
        self.arrival_rate = arrival_rate
        # Calculate reasonable min and max vehicles per round based on environment parameters
        avg_vehicles_per_round = max_vehicles / max_rounds
        self.min_vehicles_per_round = min(min_vehicles_per_round, int(avg_vehicles_per_round))
        self.max_vehicles_per_round = max(max_vehicles_per_round, int(2 * avg_vehicles_per_round))

        # Ensure min is always less than max
        if self.min_vehicles_per_round >= self.max_vehicles_per_round:
            self.min_vehicles_per_round = max(1, int(self.max_vehicles_per_round / 2))

        # State variables
        self.vehicles = []
        self.active_vehicles = []
        self.current_round = 0
        self.current_model_performance = 0.0
        self.elapsed_time = 0.0
        self.scheduled_count = 0

        # Performance tracking
        self.performance_history = []
        self.reward_history = []

    def reset(self):
        """Reset the environment to initial state"""
        self.vehicles = []
        self.active_vehicles = []
        self.current_round = 0
        self.current_model_performance = 0.0
        self.elapsed_time = 0.0
        self.scheduled_count = 0
        self.performance_history = []
        self.reward_history = []

        # Generate initial set of vehicles
        self._generate_vehicles(random.randint(10, 30))

        return self._get_state()

    def _generate_vehicles(self, count):
        """Generate new vehicles with random attributes"""
        new_vehicles = []
        for _ in range(count):
            vehicle_id = len(self.vehicles) + len(new_vehicles)

            # Generate vehicle with random attributes
            vehicle = {
                'vehicle_id': vehicle_id,
                'model_version': random.uniform(0.1, 1.0),
                'sojourn_time': random.uniform(1.0, 10.0),
                'compute_capacity': random.uniform(0.1, 1.0),
                'data_quality': random.uniform(0.3, 1.0),
                'connectivity': random.uniform(0.5, 1.0),
                'vehicle_type': random.randint(0, 3),
                'arrival_time': self.elapsed_time,
                'departure_time': self.elapsed_time + random.uniform(1.0, 10.0),
                'scheduled': False
            }

            new_vehicles.append(vehicle)

        self.vehicles.extend(new_vehicles)
        self.active_vehicles.extend(new_vehicles)
        return new_vehicles

    def get_new_vehicles(self):
        """Get newly arrived vehicles in the current time step"""
        # Simulate new vehicle arrivals based on arrival rate
        if random.random() < self.arrival_rate:
            count = random.randint(self.min_vehicles_per_round, self.max_vehicles_per_round)  # Random number of new vehicles
            return self._generate_vehicles(count)
        return []

    def _remove_departed_vehicles(self):
        """Remove vehicles that have departed"""
        current_active = []
        for vehicle in self.active_vehicles:
            if vehicle['departure_time'] > self.elapsed_time:
                current_active.append(vehicle)

        self.active_vehicles = current_active

    def _get_state(self):
        """Get current environment state"""
        return {
            'active_vehicles': self.active_vehicles,
            'current_round': self.current_round,
            'current_model_performance': self.current_model_performance,
            'elapsed_time': self.elapsed_time,
            'scheduled_count': self.scheduled_count,
            'target_performance': self.target_performance,
            'performance_gap': max(0, self.target_performance - self.current_model_performance)
        }

    def step(self, selected_vehicle_ids):
        """
        Take a step in the environment by scheduling selected vehicles

        Args:
            selected_vehicle_ids: List of vehicle IDs to schedule for the current round

        Returns:
            state: Current environment state
            reward: Reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Update time
        self.elapsed_time += 1.0
        self.current_round += 1

        # Remove departed vehicles
        self._remove_departed_vehicles()

        # Get selected vehicles
        selected_vehicles = []
        for vehicle in self.active_vehicles:
            if vehicle['vehicle_id'] in selected_vehicle_ids and not vehicle['scheduled']:
                vehicle['scheduled'] = True
                selected_vehicles.append(vehicle)

        self.scheduled_count += len(selected_vehicles)

        # Calculate reward and update model performance
        reward = self._calculate_reward(selected_vehicles)
        self.reward_history.append(reward)

        # Update model performance based on selected vehicles
        if selected_vehicles:
            # Simple model: performance increases based on data quality and compute capacity
            avg_quality = sum(v['data_quality'] for v in selected_vehicles) / len(selected_vehicles)
            avg_compute = sum(v['compute_capacity'] for v in selected_vehicles) / len(selected_vehicles)

            # Performance increase depends on quality, compute capacity, and diminishing returns
            diminishing_factor = 1.0 - self.current_model_performance  # Harder to improve as we get better
            performance_increase = 0.01 * avg_quality * avg_compute * diminishing_factor * len(selected_vehicles) / 10

            self.current_model_performance += performance_increase
            self.current_model_performance = min(1.0, self.current_model_performance)

        self.performance_history.append(self.current_model_performance)

        # Check if episode is done
        done = (self.current_round >= self.max_rounds or
                self.current_model_performance >= self.target_performance)

        # Get next state
        state = self._get_state()

        # Additional info
        info = {
            'performance_history': self.performance_history,
            'reward_history': self.reward_history,
            'selected_count': len(selected_vehicles)
        }

        return state, reward, done, info

    def _calculate_reward(self, selected_vehicles):
        """Calculate reward based on selected vehicles and current state"""
        if not selected_vehicles:
            return -0.1  # Penalty for not selecting any vehicles

        # Base reward components
        count_reward = min(1.0, len(selected_vehicles) / self.max_vehicles_per_round)

        # Quality reward based on vehicle attributes
        avg_quality = sum(v['data_quality'] for v in selected_vehicles) / len(selected_vehicles)
        avg_compute = sum(v['compute_capacity'] for v in selected_vehicles) / len(selected_vehicles)
        quality_reward = avg_quality * avg_compute

        # Diversity reward (based on vehicle types)
        vehicle_types = set(v['vehicle_type'] for v in selected_vehicles)
        diversity_reward = len(vehicle_types) / 4.0  # Assuming 4 vehicle types

        # Performance improvement reward
        performance_gap = max(0, self.target_performance - self.current_model_performance)
        improvement_potential = 5.0 * performance_gap  # Higher reward potential when far from target

        # Combine rewards
        reward = (
            0.2 * count_reward +
            0.4 * quality_reward +
            0.2 * diversity_reward +
            0.2 * improvement_potential
        )

        return reward

# Example usage
if __name__ == "__main__":
    env = VehicleModelUpdateEnv()
    state = env.reset()

    # Simple random policy for testing
    done = False
    total_reward = 0

    while not done:
        # Get available vehicles
        available_vehicles = [v for v in state['active_vehicles'] if not v['scheduled']]

        # Randomly select vehicles
        if available_vehicles:
            num_to_select = min(10, len(available_vehicles))
            selected = random.sample(available_vehicles, num_to_select)
            selected_ids = [v['vehicle_id'] for v in selected]
        else:
            selected_ids = []

        # Take step in environment
        state, reward, done, info = env.step(selected_ids)
        total_reward += reward

        print(f"Round {state['current_round']}, Selected: {len(selected_ids)}, "
              f"Performance: {state['current_model_performance']:.4f}, Reward: {reward:.4f}")

    print(f"Episode finished. Total reward: {total_reward:.4f}, "
          f"Final performance: {state['current_model_performance']:.4f}")
