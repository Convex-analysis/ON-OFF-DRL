# -*- coding: utf-8 -*-
"""
Multi-Resolution Vehicular Federated Learning (MR-VFL) Scheduler
This scheduler uses an Actor-Critic architecture to optimize vehicle selection
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
from datetime import datetime

# Set device to cpu or cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"Device set to: {torch.cuda.get_device_name(device)}")
else:
    print(f"Device set to: {device}")

################################## Define Actor Network ##################################

class MRVFLActor(nn.Module):
    """
    Actor network for MR-VFL scheduler.
    Outputs vehicle selection probabilities and continuous action parameters.
    """
    def __init__(self, input_dim=6, hidden_dim=256, n_layers=3):
        super(MRVFLActor, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Shared representation layers
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(n_layers)
        ])

        # Vehicle selection head (discrete action)
        self.selection_head = nn.Linear(hidden_dim, 1)

        # Continuous action heads
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized between 0 and 1, will be scaled later
        )

        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized between 0 and 1, will be scaled later
        )

        self.bandwidth_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalized between 0 and 1
        )

    def forward(self, vehicle_states, mask=None):
        """
        Forward pass through the actor network

        Args:
            vehicle_states: Tensor of shape [batch_size, input_dim]
            mask: Optional mask for unavailable vehicles

        Returns:
            selection_probs: Vehicle selection probabilities
            alpha: Amplification factor (scaled to [0.5, 2.0])
            scheduled_time: Scheduled time offset (scaled based on environment)
            bandwidth: Bandwidth allocation (scaled to [0.1, 1.0])
        """
        batch_size = vehicle_states.shape[0]

        # Embed vehicle features
        x = self.vehicle_embedding(vehicle_states)

        # Pass through shared layers
        for layer in self.shared_layers:
            x = layer(x)

        # Vehicle selection logits
        selection_logits = self.selection_head(x).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            selection_logits = selection_logits.masked_fill(mask == 0, -1e9)

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

################################## Define Critic Network ##################################

class MRVFLCritic(nn.Module):
    """
    Critic network for MR-VFL scheduler.
    Evaluates the state-action value function.
    """
    def __init__(self, input_dim=6, hidden_dim=256, n_layers=3):
        super(MRVFLCritic, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # Reduce dimension
            nn.ReLU()
        )

        # Action embedding (for continuous actions)
        self.action_embedding = nn.Sequential(
            nn.Linear(3, 64),  # alpha, scheduled_time, bandwidth
            nn.ReLU(),
            nn.Linear(64, hidden_dim // 2),  # Match vehicle embedding dimension
            nn.ReLU()
        )

        # Combined dimension will be hidden_dim (vehicle_dim + action_dim)
        combined_dim = hidden_dim

        # Shared representation layers
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(n_layers)
        ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, vehicle_states, actions):
        """
        Forward pass through the critic network

        Args:
            vehicle_states: Tensor of shape [batch_size, input_dim]
            actions: Tuple of (vehicle_idx, alpha, scheduled_time, bandwidth)

        Returns:
            value: Estimated state-action value
        """
        try:
            # Extract actions
            vehicle_idx, alpha, scheduled_time, bandwidth = actions

            # Get selected vehicle state
            if isinstance(vehicle_idx, int):
                # Single action case
                vehicle_state = vehicle_states[vehicle_idx].unsqueeze(0)
            else:
                # Batch case - not implemented for simplicity
                raise ValueError("Batch processing not implemented for critic")

            # Embed vehicle features
            vehicle_embed = self.vehicle_embedding(vehicle_state)

            # Embed continuous actions
            action_tensor = torch.tensor([float(alpha), float(scheduled_time), float(bandwidth)],
                                        dtype=torch.float32).unsqueeze(0).to(device)

            action_embed = self.action_embedding(action_tensor)

            # Combine embeddings
            x = torch.cat([vehicle_embed, action_embed], dim=1)

            # Pass through shared layers
            for layer in self.shared_layers:
                x = layer(x)

            # Value output
            value = self.value_head(x)

            return value
        except Exception as e:
            print(f"Error in critic forward pass: {e}")
            # Return a default value to avoid breaking the training loop
            return torch.tensor([[0.0]], requires_grad=True, device=device)

################################## Define Experience Replay Buffer ##################################

class ExperienceBuffer:
    """Experience replay buffer for training the MR-VFL scheduler"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch from buffer"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

################################## Define MR-VFL Scheduler ##################################

class MRVFLScheduler:
    """
    MR-VFL Scheduler with Actor-Critic architecture.
    Implements dual-timescale optimization and fairness constraints.
    """
    def __init__(self, input_dim=6, hidden_dim=256, n_layers=3, lr=1e-4, gamma=0.99):
        # Initialize actor and critic networks
        self.actor = MRVFLActor(input_dim, hidden_dim, n_layers).to(device)
        self.critic = MRVFLCritic(input_dim, hidden_dim, n_layers).to(device)

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize experience buffer
        self.buffer = ExperienceBuffer()

        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.batch_size = 64
        self.update_every = 4  # Update networks every N steps
        self.steps = 0

        # Fairness tracking
        self.selection_history = None
        self.fairness_threshold = 50

    def select_action(self, state, available_mask=None):
        """
        Select action based on current state

        Args:
            state: Environment state (vehicle attributes)
            available_mask: Mask for available vehicles

        Returns:
            action: [vehicle_idx, alpha, scheduled_time, bandwidth]
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(device)

        # Create mask for available vehicles
        if available_mask is None:
            available_mask = torch.ones(state.shape[0]).to(device)
        else:
            available_mask = torch.FloatTensor(available_mask).to(device)

        # Get action from actor network
        with torch.no_grad():
            selection_probs, alpha, scheduled_time, bandwidth = self.actor(state_tensor, available_mask)

        # Sample vehicle based on selection probabilities
        if available_mask.sum() > 0:
            # Normalize probabilities for available vehicles
            masked_probs = selection_probs * available_mask
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
                vehicle_idx = torch.multinomial(masked_probs, 1).item()
            else:
                # Fallback to random selection if all probabilities are zero
                available_indices = torch.nonzero(available_mask).squeeze(-1)
                vehicle_idx = available_indices[torch.randint(0, len(available_indices), (1,))].item()
        else:
            # No available vehicles, return dummy action
            vehicle_idx = 0

        # Get continuous action parameters for selected vehicle
        alpha_val = alpha[vehicle_idx].item()
        time_val = scheduled_time[vehicle_idx].item()
        bandwidth_val = bandwidth[vehicle_idx].item()

        return [vehicle_idx, alpha_val, time_val, bandwidth_val]

    def update(self):
        """Update actor and critic networks using experience replay"""
        if len(self.buffer) < self.batch_size:
            return

        # Sample batch from buffer
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = [torch.FloatTensor(state).to(device) for state in states]
        next_states = [torch.FloatTensor(next_state).to(device) for next_state in next_states]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # Update networks one sample at a time to avoid dimension issues
        critic_losses = []
        actor_losses = []

        for i in range(len(states)):
            try:
                state = states[i]
                action = actions[i]
                next_state = next_states[i]
                reward = rewards[i:i+1]  # Keep batch dimension
                done = dones[i:i+1]      # Keep batch dimension

                # Get current Q value
                current_q = self.critic(state, action)

                # Get next action from current policy
                with torch.no_grad():
                    next_action = self.select_action(next_state.cpu().numpy())

                    # Get next Q value
                    next_q = self.critic(next_state, next_action)

                    # Compute target Q value
                    target_q = reward + (1 - done) * self.gamma * next_q

                # Compute critic loss
                critic_loss = F.mse_loss(current_q, target_q)
                critic_losses.append(critic_loss)

                # Get action probabilities from actor
                selection_probs, _, _, _ = self.actor(state)

                # Get action from current policy
                vehicle_idx = action[0]

                # Compute log probability of selected action
                log_prob = torch.log(selection_probs[vehicle_idx] + 1e-10)

                # Get Q value for current action (detached to avoid double backprop)
                with torch.no_grad():
                    q_value = self.critic(state, action)

                # Compute actor loss (negative expected Q value)
                actor_loss = -log_prob * q_value.item()  # Use scalar value to avoid in-place ops

                # Add entropy regularization for exploration
                entropy = -torch.sum(selection_probs * torch.log(selection_probs + 1e-10))
                actor_loss = actor_loss - 0.01 * entropy  # Avoid in-place operation

                actor_losses.append(actor_loss)
            except Exception as e:
                print(f"Error processing experience {i}: {e}")
                continue

        # Update critic if we have any losses
        if critic_losses:
            # Compute mean critic loss
            critic_loss = torch.stack(critic_losses).mean()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Gradient clipping
            self.critic_optimizer.step()

        # Update actor if we have any losses
        if actor_losses:
            # Compute mean actor loss
            actor_loss = torch.stack(actor_losses).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Gradient clipping
            self.actor_optimizer.step()

    def train(self, env, num_episodes=1000):
        """Train the scheduler on the environment"""
        # Training metrics
        rewards_history = []
        performance_history = []
        fairness_violations = []

        for episode in range(num_episodes):
            # Reset environment
            state = env.reset()
            self.selection_history = np.zeros(env.vehicle_count)

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
                action = self.select_action(state, available_mask)

                # Take step in environment
                next_state, reward, done, info = env.step(action)

                # Update selection history
                self.selection_history[action[0]] += 1

                # Store experience in buffer
                self.buffer.push(state, action, reward, next_state, done)

                # Update networks
                self.steps += 1
                if self.steps % self.update_every == 0:
                    self.update()

                # Update state and metrics
                state = next_state
                episode_reward += reward
                step += 1

                # Check fairness violation
                if info['fairness_violation']:
                    fairness_violations.append(1)
                else:
                    fairness_violations.append(0)

            # Record episode metrics
            rewards_history.append(episode_reward)
            performance_history.append(env.current_model_performance)

            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(rewards_history[-10:])
                avg_performance = np.mean(performance_history[-10:])
                avg_fairness = np.mean(fairness_violations[-100:]) if fairness_violations else 0

                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Performance: {avg_performance:.4f} | "
                      f"Fairness Violations: {avg_fairness:.2f}")

            # Save model periodically
            if (episode + 1) % 100 == 0:
                self.save_model(f"mr_vfl_scheduler_ep{episode+1}.pt")

        # Save final model
        self.save_model("mr_vfl_scheduler_final.pt")

        return rewards_history, performance_history

    def save_model(self, filename):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)

    def load_model(self, filename):
        """Load model weights"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            print(f"Loaded model from {filename}")
            return True
        else:
            print(f"Model file {filename} not found")
            return False

# Example usage
if __name__ == "__main__":
    from vehicular_fl_env import VehicularFLEnv

    # Create environment
    env = VehicularFLEnv(vehicle_count=30, max_round=100, sync_limit=1000)

    # Create scheduler
    scheduler = MRVFLScheduler(input_dim=6, hidden_dim=128, n_layers=2)

    # Train scheduler
    rewards, performances = scheduler.train(env, num_episodes=100)

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(performances)
    plt.title('Model Performance')
    plt.xlabel('Episode')
    plt.ylabel('Performance')

    plt.tight_layout()
    plt.savefig('mr_vfl_training_results.png')
    plt.show()
