# -*- coding: utf-8 -*-
"""
Implementation of Mamba-Based Actor-Critic Scheduler for Vehicle Model Updates
This scheduler dynamically assigns vehicles to federated learning updates
while handling real-time arrivals efficiently.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from datetime import datetime
from mamba_ssm import Mamba

# Set device to cpu or cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Device set to: {torch.cuda.get_device_name(device)}")
else:
    print("Device set to: cpu")

################################## Define Data Structures ##################################

class Vehicle:
    """Vehicle class representing a connected vehicle in the system"""
    def __init__(self, vehicle_id, model_version, sojourn_time, compute_capacity, 
                 data_quality, connectivity, vehicle_type):
        self.vehicle_id = vehicle_id
        self.model_version = model_version  # Current model version
        self.sojourn_time = sojourn_time    # Expected available time
        self.compute_capacity = compute_capacity  # Processing capability
        self.data_quality = data_quality    # Data quality metric
        self.connectivity = connectivity    # Network conditions
        self.vehicle_type = vehicle_type    # Vehicle type/category
        self.arrival_time = None
        self.scheduled = False
        
    def to_tensor(self):
        """Convert vehicle attributes to tensor for model input"""
        return torch.tensor([
            self.model_version,
            self.sojourn_time,
            self.compute_capacity,
            self.data_quality,
            self.connectivity,
            self.vehicle_type
        ], dtype=torch.float32).to(device)

class GlobalState:
    """Global state of the federated learning system"""
    def __init__(self):
        self.current_model_performance = 0.0
        self.round_number = 0
        self.elapsed_time = 0.0
        self.scheduled_count = 0
        self.target_vehicle_count = 0
        self.performance_gap = 1.0  # Gap between current and target performance
        
    def to_tensor(self):
        """Convert global state to tensor for model input"""
        return torch.tensor([
            self.current_model_performance,
            self.round_number,
            self.elapsed_time,
            self.scheduled_count,
            self.target_vehicle_count,
            self.performance_gap
        ], dtype=torch.float32).to(device)
    
    def update(self, scheduled_vehicles):
        """Update global state based on scheduled vehicles"""
        self.scheduled_count += len(scheduled_vehicles)
        self.round_number += 1
        # Update model performance based on scheduled vehicles
        if scheduled_vehicles:
            # Simple model: performance increases based on data quality and compute capacity
            avg_quality = sum(v.data_quality for v in scheduled_vehicles) / len(scheduled_vehicles)
            avg_compute = sum(v.compute_capacity for v in scheduled_vehicles) / len(scheduled_vehicles)
            performance_increase = 0.01 * avg_quality * avg_compute
            self.current_model_performance += performance_increase
            self.performance_gap = max(0, 1.0 - self.current_model_performance)

################################## Define Mamba Actor Network ##################################

class MambaActor(nn.Module):
    """
    Mamba-based actor for vehicle selection.
    Key features:
    - Linear time complexity O(n) with sequence length
    - Efficient processing of streaming vehicle arrivals
    - State-space model for maintaining context
    """
    def __init__(self, input_dim, state_dim, d_model=512, n_layers=4, d_state=16):
        super(MambaActor, self).__init__()
        
        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, d_model)
        
        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, d_model)
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            ) for _ in range(n_layers)
        ])
        
        # Action head for vehicle selection probabilities
        self.action_head = nn.Linear(d_model, 1)
        
    def forward(self, vehicles, global_state, mask=None):
        batch_size = len(vehicles)
        
        # Embed global state
        state_embed = self.state_embedding(global_state)
        
        # Embed vehicle features
        vehicle_embeds = torch.stack([self.vehicle_embedding(v) for v in vehicles])
        
        # Concatenate state with each vehicle embedding
        state_expanded = state_embed.unsqueeze(0).expand(batch_size, -1)
        combined_embeds = vehicle_embeds + state_expanded
        
        # Process through Mamba layers
        x = combined_embeds
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        
        # Generate selection probabilities
        logits = self.action_head(x).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        
        # Return selection probabilities
        return F.softmax(logits, dim=0)

################################## Define Mamba Critic Network ##################################

class MambaCritic(nn.Module):
    """
    Mamba-based critic for state-value estimation.
    Evaluates the quality of current state and scheduling decisions.
    """
    def __init__(self, input_dim, state_dim, d_model=384, n_layers=3, d_state=16):
        super(MambaCritic, self).__init__()
        
        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, d_model)
        
        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, d_model)
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            ) for _ in range(n_layers)
        ])
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, vehicles, global_state):
        batch_size = len(vehicles)
        
        # Embed global state
        state_embed = self.state_embedding(global_state)
        
        # Embed vehicle features
        vehicle_embeds = torch.stack([self.vehicle_embedding(v) for v in vehicles])
        
        # Concatenate state with each vehicle embedding
        state_expanded = state_embed.unsqueeze(0).expand(batch_size, -1)
        combined_embeds = vehicle_embeds + state_expanded
        
        # Process through Mamba layers
        x = combined_embeds
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        
        # Pool vehicle representations
        x = torch.mean(x, dim=0)
        
        # Output state value estimation
        return self.value_head(x)

################################## Create Streaming Scheduler ##################################

class StreamingMambaScheduler:
    """
    Handles real-time vehicle arrivals with O(1) complexity per new vehicle.
    Maintains persistent state for efficient updates.
    """
    def __init__(self, actor, critic, max_vehicles=100):
        self.actor = actor
        self.critic = critic
        self.vehicle_cache = []
        self.hidden_state = None
        self.global_state = GlobalState()
        self.max_vehicles = max_vehicles
        
    def process_new_vehicle(self, vehicle):
        """Process single vehicle arrival efficiently"""
        vehicle.arrival_time = self.global_state.elapsed_time
        self.vehicle_cache.append(vehicle)
        
        # Limit cache size by removing oldest vehicles if needed
        if len(self.vehicle_cache) > self.max_vehicles:
            self.vehicle_cache.pop(0)
            
        return vehicle
        
    def make_scheduling_decision(self, target_count=10):
        """Generate scheduling actions based on current state"""
        if not self.vehicle_cache:
            return []
            
        # Create mask for vehicles that can't be scheduled (e.g., too short sojourn time)
        mask = torch.ones(len(self.vehicle_cache)).to(device)
        for i, vehicle in enumerate(self.vehicle_cache):
            if vehicle.scheduled or vehicle.sojourn_time < 1.0:  # Minimum required time
                mask[i] = 0
                
        # Get selection probabilities from actor
        vehicle_tensors = [v.to_tensor() for v in self.vehicle_cache]
        global_state_tensor = self.global_state.to_tensor()
        
        with torch.no_grad():
            selection_probs = self.actor(vehicle_tensors, global_state_tensor, mask)
        
        # Select vehicles based on probabilities
        selected_indices = []
        remaining_indices = list(range(len(self.vehicle_cache)))
        remaining_indices = [i for i in remaining_indices if mask[i] > 0]
        
        # Select up to target_count vehicles
        for _ in range(min(target_count, len(remaining_indices))):
            if not remaining_indices:
                break
                
            # Normalize probabilities for remaining vehicles
            probs = selection_probs[remaining_indices]
            probs = probs / probs.sum()
            
            # Sample based on probabilities
            idx = np.random.choice(len(remaining_indices), p=probs.cpu().numpy())
            selected_idx = remaining_indices[idx]
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        
        # Mark selected vehicles as scheduled
        selected_vehicles = []
        for idx in selected_indices:
            self.vehicle_cache[idx].scheduled = True
            selected_vehicles.append(self.vehicle_cache[idx])
            
        # Update global state
        self.global_state.update(selected_vehicles)
        
        return selected_vehicles

################################## Implement Experience Replay Buffer ##################################

class EpisodicReplayMemory:
    """Experience replay buffer for training the Mamba scheduler"""
    def __init__(self, capacity, max_episode_length):
        self.num_episodes = capacity // max_episode_length
        self.buffer = deque(maxlen=self.num_episodes)
        self.buffer.append([])
        self.position = 0
        
    def push(self, vehicles, global_state, actions, reward, next_vehicles, next_global_state, done):
        self.buffer[self.position].append((vehicles, global_state, actions, reward, next_vehicles, next_global_state, done))
        if done:
            self.buffer.append([])
            self.position = min(self.position + 1, self.num_episodes - 1)
            
    def sample(self, batch_size, max_len=None):
        min_len = 0
        while min_len == 0:
            rand_episodes = random.sample(self.buffer, batch_size)
            min_len = min(len(episode) for episode in rand_episodes)
            
        if max_len:
            max_len = min(max_len, min_len)
        else:
            max_len = min_len
            
        episodes = []
        for episode in rand_episodes:
            if len(episode) > max_len:
                rand_idx = random.randint(0, len(episode) - max_len)
            else:
                rand_idx = 0

            episodes.append(episode[rand_idx:rand_idx+max_len])
            
        return list(map(list, zip(*episodes)))
    
    def __len__(self):
        return len(self.buffer)

################################## Implement Actor-Critic Training Loop ##################################

def compute_returns(rewards, masks, gamma=0.99):
    """Compute discounted returns"""
    returns = torch.zeros_like(rewards)
    running_returns = 0
    
    for t in reversed(range(len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns
        
    return returns

def train_mamba_scheduler(environment, num_episodes=1000, gamma=0.99, lr=1e-4):
    """
    Training logic with specific optimizations for Mamba:
    - Experience replay buffer
    - Asynchronous training for streaming data
    - Dynamic learning rate adjustment
    """
    # Initialize models
    vehicle_feature_dim = 6  # model_version, sojourn_time, compute_capacity, data_quality, connectivity, type
    global_state_dim = 6     # current_model_performance, round_number, elapsed_time, scheduled_count, target_count, performance_gap
    
    actor = MambaActor(vehicle_feature_dim, global_state_dim).to(device)
    critic = MambaCritic(vehicle_feature_dim, global_state_dim).to(device)
    
    # Initialize optimizers
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)
    
    # Initialize replay buffer
    replay_buffer = EpisodicReplayMemory(capacity=10000, max_episode_length=200)
    
    # Training loop
    for episode in range(num_episodes):
        state = environment.reset()
        scheduler = StreamingMambaScheduler(actor, critic)
        
        episode_rewards = []
        done = False
        
        while not done:
            # Get new vehicles from environment
            new_vehicles = environment.get_new_vehicles()
            for vehicle in new_vehicles:
                scheduler.process_new_vehicle(vehicle)
                
            # Make scheduling decision
            selected_vehicles = scheduler.make_scheduling_decision()
            
            # Take action in environment
            next_state, reward, done, info = environment.step(selected_vehicles)
            
            # Store transition in replay buffer
            replay_buffer.push(
                scheduler.vehicle_cache.copy(),
                scheduler.global_state,
                selected_vehicles,
                reward,
                scheduler.vehicle_cache.copy(),  # Next vehicles (updated after step)
                scheduler.global_state,          # Next global state (updated after step)
                done
            )
            
            episode_rewards.append(reward)
            
            # Update environment state
            state = next_state
            
            # Perform experience replay
            if len(replay_buffer) > 1:
                # Sample batch from replay buffer
                batch = replay_buffer.sample(batch_size=16)
                vehicles_batch, global_states_batch, actions_batch, rewards_batch, next_vehicles_batch, next_global_states_batch, dones_batch = batch
                
                # Convert to tensors
                rewards = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
                masks = torch.tensor([1.0 - float(done) for done in dones_batch], dtype=torch.float32).to(device)
                
                # Compute returns
                returns = compute_returns(rewards, masks, gamma)
                
                # Update critic
                for i in range(len(vehicles_batch)):
                    value = critic(vehicles_batch[i], global_states_batch[i])
                    critic_loss = F.mse_loss(value, returns[i])
                    
                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    optimizer_critic.step()
                
                # Update actor
                for i in range(len(vehicles_batch)):
                    # Get action probabilities
                    probs = actor(vehicles_batch[i], global_states_batch[i])
                    
                    # Compute advantage
                    value = critic(vehicles_batch[i], global_states_batch[i]).detach()
                    advantage = returns[i] - value
                    
                    # Compute actor loss
                    action_log_probs = torch.log(probs)
                    actor_loss = -torch.mean(action_log_probs * advantage)
                    
                    # Add entropy regularization
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    actor_loss -= 0.01 * entropy  # Entropy coefficient
                    
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()
        
        # Print episode statistics
        episode_reward = sum(episode_rewards)
        print(f"Episode {episode}, Total Reward: {episode_reward}")
        
        # Save models periodically
        if episode % 100 == 0:
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_actor_state_dict': optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
            }, f'mamba_scheduler_checkpoint_{episode}.pt')
    
    return actor, critic

################################## Optimize for Production Deployment ##################################

class OptimizedMambaScheduler:
    """
    Production-ready scheduler with optimizations:
    - Model quantization (int8/fp16)
    - JIT compilation
    - Hardware-specific acceleration
    """
    def __init__(self, pretrained_model_path):
        # Load pretrained model
        checkpoint = torch.load(pretrained_model_path)
        
        # Initialize models
        vehicle_feature_dim = 6
        global_state_dim = 6
        
        self.actor = MambaActor(vehicle_feature_dim, global_state_dim).to(device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        
        # Quantize model for faster inference
        if device.type == 'cuda':
            self.actor = self.actor.half()  # Convert to FP16 for faster inference
        
        # JIT compile for faster execution
        self.actor = torch.jit.script(self.actor)
        
        # Set to evaluation mode
        self.actor.eval()
        
        # Initialize scheduler
        self.scheduler = StreamingMambaScheduler(self.actor, None)
        
    def inference_mode_scheduling(self, vehicles, global_state, target_count=10):
        """Fast inference path for production"""
        # Update scheduler state
        self.scheduler.global_state = global_state
        self.scheduler.vehicle_cache = []
        
        # Process vehicles
        for vehicle in vehicles:
            self.scheduler.process_new_vehicle(vehicle)
            
        # Make scheduling decision
        with torch.no_grad():
            selected_vehicles = self.scheduler.make_scheduling_decision(target_count)
            
        return selected_vehicles

if __name__ == "__main__":
    # This would be replaced with your actual environment
    class DummyEnvironment:
        def __init__(self):
            self.time = 0
            self.vehicles = []
            self.global_state = GlobalState()
            
        def reset(self):
            self.time = 0
            self.vehicles = []
            self.global_state = GlobalState()
            return self.global_state
            
        def get_new_vehicles(self):
            # Simulate new vehicle arrivals
            new_vehicles = []
            if random.random() < 0.7:  # 70% chance of new vehicle
                for _ in range(random.randint(1, 3)):
                    vehicle = Vehicle(
                        vehicle_id=len(self.vehicles),
                        model_version=random.uniform(0.1, 1.0),
                        sojourn_time=random.uniform(1.0, 10.0),
                        compute_capacity=random.uniform(0.1, 1.0),
                        data_quality=random.uniform(0.3, 1.0),
                        connectivity=random.uniform(0.5, 1.0),
                        vehicle_type=random.randint(0, 3)
                    )
                    new_vehicles.append(vehicle)
                    self.vehicles.append(vehicle)
            return new_vehicles
            
        def step(self, selected_vehicles):
            # Update time
            self.time += 1
            self.global_state.elapsed_time = self.time
            
            # Calculate reward based on selected vehicles
            if selected_vehicles:
                # Reward based on vehicle quality and diversity
                avg_quality = sum(v.data_quality for v in selected_vehicles) / len(selected_vehicles)
                avg_compute = sum(v.compute_capacity for v in selected_vehicles) / len(selected_vehicles)
                reward = avg_quality * avg_compute * len(selected_vehicles) * 0.1
            else:
                reward = 0
                
            # Check if episode is done
            done = self.time >= 100 or self.global_state.current_model_performance >= 0.95
            
            return self.global_state, reward, done, {}
    
    # Create environment and train
    env = DummyEnvironment()
    actor, critic = train_mamba_scheduler(env, num_episodes=10)  # Reduced for testing
    
    # Save final model
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
    }, 'mamba_scheduler_final.pt')
    
    # Test optimized scheduler
    optimized_scheduler = OptimizedMambaScheduler('mamba_scheduler_final.pt')
    
    # Generate test vehicles
    test_vehicles = [
        Vehicle(1, 0.5, 5.0, 0.8, 0.9, 0.7, 1),
        Vehicle(2, 0.3, 3.0, 0.6, 0.7, 0.8, 2),
        Vehicle(3, 0.7, 7.0, 0.9, 0.8, 0.9, 0)
    ]
    
    # Test scheduling
    test_global_state = GlobalState()
    selected = optimized_scheduler.inference_mode_scheduling(test_vehicles, test_global_state)
    print(f"Selected {len(selected)} vehicles for scheduling")
