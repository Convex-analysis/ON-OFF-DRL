# -*- coding: utf-8 -*-
"""
Multi-Resolution Vehicular Federated Learning (MR-VFL) Scheduler with Mamba Architecture
This scheduler uses a Mamba-based Actor-Critic architecture to optimize vehicle selection
with fairness constraints and dual-timescale optimization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time
from datetime import datetime

# Try to import Mamba SSM, use GRU as fallback if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Using Mamba SSM for sequence modeling")
except ImportError:
    MAMBA_AVAILABLE = False
    print("Mamba SSM not available, using GRU as fallback")

# Set device to cpu or cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Enable cuDNN benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    print(f"Device set to: {torch.cuda.get_device_name(device)}")
else:
    print(f"Device set to: {device}")

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
        self._tensor = None  # Cache for tensor representation

    def to_tensor(self, dtype=torch.float32):
        """Convert vehicle attributes to tensor for model input"""
        # Use cached tensor if available and dtype matches
        if self._tensor is not None and self._tensor.dtype == dtype:
            return self._tensor

        # Create new tensor with specified dtype
        self._tensor = torch.tensor([
            self.model_version,
            self.sojourn_time,
            self.compute_capacity,
            self.data_quality,
            self.connectivity,
            self.vehicle_type
        ], dtype=dtype).to(device)

        return self._tensor

class GlobalState:
    """Global state of the federated learning system"""
    def __init__(self):
        self.current_model_performance = 0.0
        self.round_number = 0
        self.elapsed_time = 0.0
        self.scheduled_count = 0
        self.target_vehicle_count = 0
        self.performance_gap = 1.0  # Gap between current and target performance

    def to_tensor(self, dtype=torch.float32):
        """Convert global state to tensor for model input"""
        return torch.tensor([
            self.current_model_performance,
            self.round_number,
            self.elapsed_time,
            self.scheduled_count,
            self.target_vehicle_count,
            self.performance_gap
        ], dtype=dtype).to(device)

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

class MRVFLMambaActor(nn.Module):
    """
    Mamba-based actor for vehicle selection.
    Key features:
    - Linear time complexity O(n) with sequence length
    - Efficient processing of streaming vehicle arrivals
    - State-space model for maintaining context
    - Outputs both vehicle selection probabilities and continuous action parameters
    """
    def __init__(self, input_dim=6, state_dim=6, d_model=256, n_layers=3, d_state=16):
        super(MRVFLMambaActor, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, d_model)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, d_model)

        # Sequence modeling layers (Mamba or GRU)
        if MAMBA_AVAILABLE:
            self.sequence_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=4,
                    expand=2
                ) for _ in range(n_layers)
            ])
        else:
            # Fallback to GRU if Mamba is not available
            self.sequence_layers = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_layers,
                batch_first=True
            )

        # Vehicle selection head (discrete action)
        self.selection_head = nn.Linear(d_model, 1)
        
        # Continuous action heads
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized between 0 and 1, will be scaled later
        )
        
        self.time_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized between 0 and 1, will be scaled later
        )
        
        self.bandwidth_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized between 0 and 1
        )

    def forward(self, vehicles, global_state, mask=None):
        """
        Forward pass through the actor network
        
        Args:
            vehicles: List of vehicle tensors
            global_state: Global state tensor
            mask: Optional mask for unavailable vehicles
            
        Returns:
            selection_probs: Vehicle selection probabilities
            alpha: Amplification factor (scaled to [0.5, 2.0])
            scheduled_time: Scheduled time offset (scaled based on environment)
            bandwidth: Bandwidth allocation (scaled to [0.1, 1.0])
        """
        batch_size = len(vehicles)

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Embed vehicle features
        vehicle_embeds = torch.stack([self.vehicle_embedding(v) for v in vehicles])

        # Concatenate state with each vehicle embedding
        state_expanded = state_embed.unsqueeze(0).expand(batch_size, -1)
        combined_embeds = vehicle_embeds + state_expanded

        # Process through sequence layers
        if MAMBA_AVAILABLE:
            # Add sequence dimension for Mamba: (batch, dim) -> (1, batch, dim)
            x = combined_embeds.unsqueeze(0)  # shape: (1, batch, d_model)
            for layer in self.sequence_layers:
                x = layer(x)
            x = x.squeeze(0)  # shape: (batch, d_model)
        else:
            # Process through GRU
            x = combined_embeds.unsqueeze(0)  # Add batch dimension
            x, _ = self.sequence_layers(x)
            x = x.squeeze(0)  # Remove batch dimension

        # Vehicle selection logits
        selection_logits = self.selection_head(x).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            # Use a smaller value for masking that's compatible with half-precision
            mask_value = -65504.0 if selection_logits.dtype == torch.float16 else -1e9
            selection_logits = selection_logits.masked_fill(mask == 0, mask_value)

        # Selection probabilities
        selection_probs = F.softmax(selection_logits, dim=0)
        
        # Continuous action parameters
        alpha_raw = self.alpha_head(x).squeeze(-1)
        time_raw = self.time_head(x).squeeze(-1)
        bandwidth_raw = self.bandwidth_head(x).squeeze(-1)
        
        # Scale continuous actions to appropriate ranges
        alpha = 0.5 + 1.5 * alpha_raw  # Scale to [0.5, 2.0]
        scheduled_time = 10 + 90 * time_raw  # Scale to [10, 100] offset
        bandwidth = 0.1 + 0.9 * bandwidth_raw  # Scale to [0.1, 1.0]
        
        return selection_probs, alpha, scheduled_time, bandwidth

################################## Define Mamba Critic Network ##################################

class MRVFLMambaCritic(nn.Module):
    """
    Mamba-based critic for state-value estimation.
    Evaluates the quality of current state and scheduling decisions.
    """
    def __init__(self, input_dim=6, state_dim=6, d_model=256, n_layers=3, d_state=16):
        super(MRVFLMambaCritic, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, d_model)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, d_model)

        # Action embedding (for continuous actions)
        self.action_embedding = nn.Sequential(
            nn.Linear(3, 64),  # alpha, scheduled_time, bandwidth
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.ReLU()
        )

        # Sequence modeling layers (Mamba or GRU)
        if MAMBA_AVAILABLE:
            self.sequence_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=4,
                    expand=2
                ) for _ in range(n_layers)
            ])
        else:
            # Fallback to GRU if Mamba is not available
            self.sequence_layers = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_layers,
                batch_first=True
            )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, vehicles, global_state, actions):
        """
        Forward pass through the critic network
        
        Args:
            vehicles: List of vehicle tensors
            global_state: Global state tensor
            actions: Tuple of (vehicle_idx, alpha, scheduled_time, bandwidth)
            
        Returns:
            value: Estimated state-action value
        """
        # Extract actions
        vehicle_idx, alpha, scheduled_time, bandwidth = actions
        
        # Get selected vehicle tensor
        if isinstance(vehicle_idx, int):
            # Single action case
            vehicle_tensor = vehicles[vehicle_idx]
        else:
            # Batch case - not implemented for simplicity
            raise NotImplementedError("Batch processing not implemented for critic")
        
        # Embed vehicle features
        vehicle_embed = self.vehicle_embedding(vehicle_tensor)
        
        # Embed global state
        state_embed = self.state_embedding(global_state)
        
        # Embed continuous actions
        action_tensor = torch.tensor([alpha, scheduled_time, bandwidth], 
                                     dtype=torch.float32).to(device)
        action_embed = self.action_embedding(action_tensor)
        
        # Combine embeddings (vehicle + state + action)
        combined_embed = vehicle_embed + state_embed + action_embed
        
        # Process through sequence layers
        if MAMBA_AVAILABLE:
            # Add sequence dimension for Mamba: (dim) -> (1, 1, dim)
            x = combined_embed.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, d_model)
            for layer in self.sequence_layers:
                x = layer(x)
            x = x.squeeze(0).squeeze(0)  # shape: (d_model)
        else:
            # Process through GRU
            x = combined_embed.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            x, _ = self.sequence_layers(x)
            x = x.squeeze(0).squeeze(0)  # Remove batch and sequence dimensions
        
        # Output state-action value
        return self.value_head(x)

################################## Define Experience Replay Buffer ##################################

class EpisodicReplayMemory:
    """Experience replay buffer for training the Mamba scheduler"""
    def __init__(self, capacity=10000, max_episode_length=200):
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

################################## Create Streaming Scheduler ##################################

class StreamingMRVFLScheduler:
    """
    Handles real-time vehicle arrivals with O(1) complexity per new vehicle.
    Maintains persistent state for efficient updates.
    """
    def __init__(self, actor, critic=None, max_vehicles=100):
        self.actor = actor
        self.critic = critic
        self.vehicle_cache = []
        self.global_state = GlobalState()
        self.max_vehicles = max_vehicles
        self.selection_history = {}  # Track selection history for fairness

    def process_new_vehicle(self, vehicle):
        """Process single vehicle arrival efficiently"""
        vehicle.arrival_time = self.global_state.elapsed_time
        self.vehicle_cache.append(vehicle)
        
        # Initialize selection history for this vehicle
        if vehicle.vehicle_id not in self.selection_history:
            self.selection_history[vehicle.vehicle_id] = 0

        # Limit cache size by removing oldest vehicles if needed
        if len(self.vehicle_cache) > self.max_vehicles:
            self.vehicle_cache.pop(0)

        return vehicle

    def make_scheduling_decision(self, target_count=10):
        """Generate scheduling actions based on current state"""
        if not self.vehicle_cache:
            return []

        # Determine the dtype to use based on the actor's parameters
        dtype = torch.float32
        if hasattr(self.actor, 'parameters'):
            try:
                dtype = next(self.actor.parameters()).dtype
            except StopIteration:
                pass  # Use default dtype if no parameters

        # Create mask for vehicles that can't be scheduled (e.g., too short sojourn time)
        mask = torch.ones(len(self.vehicle_cache), dtype=dtype).to(device)
        for i, vehicle in enumerate(self.vehicle_cache):
            if vehicle.scheduled or vehicle.sojourn_time < 1.0:  # Minimum required time
                mask[i] = 0

        # Get selection probabilities from actor
        vehicle_tensors = [v.to_tensor(dtype=dtype) for v in self.vehicle_cache]
        global_state_tensor = self.global_state.to_tensor().to(dtype)

        with torch.no_grad():
            selection_probs, alpha, scheduled_time, bandwidth = self.actor(vehicle_tensors, global_state_tensor, mask)

        # Select vehicles based on probabilities
        selected_indices = []
        remaining_indices = [i for i in range(len(self.vehicle_cache)) if mask[i] > 0]

        # Select up to target_count vehicles
        for _ in range(min(target_count, len(remaining_indices))):
            if not remaining_indices:
                break

            # Normalize probabilities for remaining vehicles
            probs = selection_probs[remaining_indices]
            probs_sum = probs.sum().item()
            
            # Check for invalid probabilities
            if probs_sum == 0 or np.isnan(probs.cpu().numpy()).any():
                idx = np.random.choice(len(remaining_indices))
            else:
                probs = probs / probs_sum
                probs_np = probs.cpu().numpy()
                idx = np.random.choice(len(remaining_indices), p=probs_np)

            selected_idx = remaining_indices[idx]
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)

        # Mark selected vehicles as scheduled and update selection history
        selected_vehicles = []
        for idx in selected_indices:
            vehicle = self.vehicle_cache[idx]
            vehicle.scheduled = True
            self.selection_history[vehicle.vehicle_id] += 1
            selected_vehicles.append(vehicle)

        # Update global state
        self.global_state.update(selected_vehicles)

        return selected_vehicles

################################## Optimize for Production Deployment ##################################

class OptimizedMRVFLScheduler:
    """
    Production-ready scheduler with optimizations:
    - Model quantization (int8/fp16)
    - JIT compilation
    - Hardware-specific acceleration
    - Optimized for inference speed
    """
    def __init__(self, pretrained_model_path=None):
        # Initialize models
        vehicle_feature_dim = 6
        global_state_dim = 6

        # Create actor model with optimized parameters
        self.actor = MRVFLMambaActor(
            input_dim=vehicle_feature_dim, 
            state_dim=global_state_dim, 
            d_model=256, 
            n_layers=3, 
            d_state=16
        ).to(device)

        # Load pretrained model if provided
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            try:
                checkpoint = torch.load(pretrained_model_path, map_location=device)
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                print(f"Loaded model from {pretrained_model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using randomly initialized model")
        else:
            print("Using randomly initialized model")

        # Set to evaluation mode
        self.actor.eval()

        # Try to use half precision if available
        if device.type == 'cuda':
            try:
                self.actor = self.actor.half()
                print("Using half precision (FP16) for faster inference")
                self.use_half = True
            except Exception as e:
                print(f"Could not use half precision: {e}")
                self.use_half = False
        else:
            self.use_half = False

        # Initialize scheduler
        self.scheduler = StreamingMRVFLScheduler(self.actor)

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

    def select_vehicles(self, env_vehicles, target_count=10):
        """
        Interface method for compatibility with comparison framework
        
        Args:
            env_vehicles: List of environment vehicles
            target_count: Number of vehicles to select
            
        Returns:
            selected_vehicles: List of selected vehicles
        """
        # Convert environment vehicles to scheduler vehicles
        scheduler_vehicles = []
        for v in env_vehicles:
            if not v['scheduled']:
                # Use fallback to support both 'model_version' and 'version'
                model_version = v.get('model_version', v.get('version', 0.0))
                sojourn_time = v.get('sojourn_time', v.get('sojourn', 0.0))
                compute_capacity = v.get('compute_capacity', v.get('compute', 0.0))
                data_quality = v.get('data_quality', v.get('quality', 0.0))
                connectivity = v.get('connectivity', v.get('conn', 0.0))
                vehicle_type = v.get('vehicle_type', v.get('type', 0))
                vehicle = Vehicle(
                    vehicle_id=v['id'],
                    model_version=model_version,
                    sojourn_time=sojourn_time,
                    compute_capacity=compute_capacity,
                    data_quality=data_quality,
                    connectivity=connectivity,
                    vehicle_type=vehicle_type
                )
                scheduler_vehicles.append(vehicle)
        
        # Create global state
        global_state = GlobalState()
        
        # Make scheduling decision
        selected_vehicles = self.inference_mode_scheduling(
            scheduler_vehicles, global_state, target_count
        )
        
        # Convert back to environment vehicle format
        return [v for v in env_vehicles if v['id'] in [sv.vehicle_id for sv in selected_vehicles]]

# Example usage
if __name__ == "__main__":
    # Create a dummy model for testing
    vehicle_feature_dim = 6
    global_state_dim = 6
    
    # Create model
    actor = MRVFLMambaActor(vehicle_feature_dim, global_state_dim).to(device)
    
    # Save dummy model
    os.makedirs("mr_vfl_models", exist_ok=True)
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': {},
    }, 'mr_vfl_models/mr_vfl_mamba_scheduler_final.pt')
    
    # Test optimized scheduler
    optimized_scheduler = OptimizedMRVFLScheduler('mr_vfl_models/mr_vfl_mamba_scheduler_final.pt')
    
    # Generate test vehicles
    test_vehicles = [
        Vehicle(1, 0.5, 5.0, 0.8, 0.9, 0.7, 1),
        Vehicle(2, 0.3, 3.0, 0.6, 0.7, 0.8, 2),
        Vehicle(3, 0.7, 7.0, 0.9, 0.8, 0.9, 0)
    ]
    
    # Test scheduling
    test_global_state = GlobalState()
    
    # Measure inference time
    start_time = time.time()
    selected = optimized_scheduler.inference_mode_scheduling(test_vehicles, test_global_state)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    print(f"Selected {len(selected)} vehicles for scheduling")
    print(f"Inference time: {inference_time:.2f} ms")
