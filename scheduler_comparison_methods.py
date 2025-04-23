# -*- coding: utf-8 -*-
"""
State-of-the-Art Scheduler Methods for Comparison with Mamba-Based Scheduler
This file implements several SOTA approaches for vehicle scheduling in federated learning.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import time
from collections import deque
from torch.distributions import Categorical

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################## 1. Transformer-Based Scheduler ##################################

class TransformerScheduler(nn.Module):
    """
    Transformer-based scheduler for vehicle selection.
    Uses self-attention to model relationships between vehicles.
    """
    def __init__(self, input_dim, state_dim, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerScheduler, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, d_model)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder with batch_first=True
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Action head for vehicle selection probabilities
        self.action_head = nn.Linear(d_model, 1)

    def forward(self, vehicles, global_state, mask=None):
        batch_size = len(vehicles)

        # Embed vehicle features
        vehicle_embeds = torch.stack([self.vehicle_embedding(v) for v in vehicles])

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Add global state to each vehicle embedding
        state_expanded = state_embed.unsqueeze(0).expand(batch_size, -1)
        combined_embeds = vehicle_embeds + state_expanded

        # For transformer with batch_first=True, we need to add a sequence dimension
        # Reshape from [batch_size, d_model] to [batch_size, 1, d_model]
        combined_embeds = combined_embeds.unsqueeze(1)

        # Add positional encoding
        combined_embeds = self.pos_encoder(combined_embeds)

        # Process through transformer without mask (we'll apply mask after)
        transformer_out = self.transformer_encoder(combined_embeds)

        # Remove the sequence dimension [batch_size, 1, d_model] -> [batch_size, d_model]
        transformer_out = transformer_out.squeeze(1)

        # Generate selection probabilities
        logits = self.action_head(transformer_out).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            mask_value = -65504.0 if logits.dtype == torch.float16 else -1e9
            logits = logits.masked_fill(mask == 0, mask_value)

        # Return selection probabilities
        return F.softmax(logits, dim=0)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # For 2D input (batch_size, embedding_dim), just add positional encoding directly
            # Use the first position encoding for all items in batch
            pe = self.pe[0:1, :].expand(x.size(0), -1)  # [batch_size, d_model]
            x = x + pe
        else:  # 3D input
            # For 3D input (batch_size, seq_len, embedding_dim)
            seq_len = x.size(1)
            pe = self.pe[:seq_len].unsqueeze(0)  # [1, seq_len, d_model]
            x = x + pe

        return self.dropout(x)

################################## 2. GNN-Based Scheduler ##################################

class GNNScheduler(nn.Module):
    """
    Graph Neural Network-based scheduler.
    Models vehicles as nodes in a graph with learned edge relationships.
    """
    def __init__(self, input_dim, state_dim, hidden_dim=256, num_layers=3):
        super(GNNScheduler, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, hidden_dim)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # Edge prediction network
        self.edge_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Action head for vehicle selection probabilities
        self.action_head = nn.Linear(hidden_dim, 1)

    def forward(self, vehicles, global_state, mask=None):
        batch_size = len(vehicles)

        # Embed vehicle features
        vehicle_embeds = torch.stack([self.vehicle_embedding(v) for v in vehicles])

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Add global state to each vehicle embedding
        state_expanded = state_embed.unsqueeze(0).expand(batch_size, -1)
        node_features = vehicle_embeds + state_expanded

        # Compute edge weights
        edge_weights = torch.zeros(batch_size, batch_size, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    # Concatenate node features to predict edge weight
                    edge_input = torch.cat([node_features[i], node_features[j]], dim=0)
                    edge_weights[i, j] = self.edge_network(edge_input)

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            node_features = gnn_layer(node_features, edge_weights)

        # Generate selection probabilities
        logits = self.action_head(node_features).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            mask_value = -65504.0 if logits.dtype == torch.float16 else -1e9
            logits = logits.masked_fill(mask == 0, mask_value)

        # Return selection probabilities
        return F.softmax(logits, dim=0)

class GNNLayer(nn.Module):
    """
    Graph Neural Network layer for message passing between nodes.
    """
    def __init__(self, hidden_dim):
        super(GNNLayer, self).__init__()

        # Message passing network
        self.message_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update network
        self.update_network = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, node_features, edge_weights):
        batch_size = node_features.size(0)
        messages = torch.zeros_like(node_features)

        # Compute messages
        for i in range(batch_size):
            # Aggregate messages from neighbors
            neighbor_msgs = torch.zeros(node_features.size(1), device=device)
            for j in range(batch_size):
                if i != j:
                    # Weight message by edge weight
                    msg_input = torch.cat([node_features[i], node_features[j]], dim=0)
                    msg = self.message_network(msg_input) * edge_weights[i, j]
                    neighbor_msgs += msg

            # Update node feature with aggregated messages
            messages[i] = neighbor_msgs

        # Update node features
        updated_features = torch.zeros_like(node_features)
        for i in range(batch_size):
            updated_features[i] = self.update_network(messages[i].unsqueeze(0), node_features[i].unsqueeze(0))

        return updated_features

################################## 3. LSTM-Based Scheduler ##################################

class LSTMScheduler(nn.Module):
    """
    LSTM-based scheduler for sequential vehicle selection.
    Processes vehicles as a sequence and maintains hidden state.
    """
    def __init__(self, input_dim, state_dim, hidden_dim=256, num_layers=2):
        super(LSTMScheduler, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, hidden_dim)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Action head for vehicle selection probabilities
        self.action_head = nn.Linear(hidden_dim, 1)

    def forward(self, vehicles, global_state, mask=None, hidden=None):
        batch_size = len(vehicles)

        # Embed vehicle features
        vehicle_embeds = torch.stack([self.vehicle_embedding(v) for v in vehicles])

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Add global state to each vehicle embedding
        state_expanded = state_embed.unsqueeze(0).expand(batch_size, -1)
        combined_embeds = vehicle_embeds + state_expanded

        # Process through LSTM
        combined_embeds = combined_embeds.unsqueeze(0)  # Add batch dimension
        lstm_out, hidden = self.lstm(combined_embeds, hidden)
        lstm_out = lstm_out.squeeze(0)  # Remove batch dimension

        # Generate selection probabilities
        logits = self.action_head(lstm_out).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            mask_value = -65504.0 if logits.dtype == torch.float16 else -1e9
            logits = logits.masked_fill(mask == 0, mask_value)

        # Return selection probabilities and hidden state
        return F.softmax(logits, dim=0), hidden

################################## 4. Multi-Agent Reinforcement Learning Scheduler ##################################

class MARLScheduler(nn.Module):
    """
    Multi-Agent Reinforcement Learning scheduler.
    Treats each vehicle as an agent with its own policy.
    """
    def __init__(self, input_dim, state_dim, hidden_dim=128):
        super(MARLScheduler, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, hidden_dim)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # Vehicle policy network
        self.vehicle_policy = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Binary decision: select or not
        )

    def forward(self, vehicles, global_state, mask=None, temperature=1.0):
        batch_size = len(vehicles)

        # Embed vehicle features
        vehicle_embeds = [self.vehicle_embedding(v) for v in vehicles]

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Compute selection probabilities for each vehicle
        selection_probs = []
        for i in range(batch_size):
            # Concatenate vehicle and state embeddings
            combined = torch.cat([vehicle_embeds[i], state_embed], dim=0)

            # Get policy logits
            logits = self.vehicle_policy(combined)

            # Apply temperature scaling
            logits = logits / temperature

            # Apply mask if provided
            if mask is not None and mask[i] == 0:
                logits[1] = -65504.0 if logits.dtype == torch.float16 else -1e9

            # Convert to probabilities
            probs = F.softmax(logits, dim=0)

            # Store selection probability (probability of selecting)
            selection_probs.append(probs[1])

        # Stack probabilities
        selection_probs = torch.stack(selection_probs)

        return selection_probs

################################## 5. Hierarchical Scheduler ##################################

class HierarchicalScheduler(nn.Module):
    """
    Hierarchical scheduler that first clusters vehicles and then selects from clusters.
    Uses a two-level decision process for improved scalability.
    """
    def __init__(self, input_dim, state_dim, hidden_dim=256, num_clusters=5):
        super(HierarchicalScheduler, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, hidden_dim)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)

        # Cluster assignment network
        self.cluster_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clusters)
        )

        # Cluster selection network
        self.cluster_selection = nn.Sequential(
            nn.Linear(hidden_dim + num_clusters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_clusters)
        )

        # Vehicle selection network
        self.vehicle_selection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.num_clusters = num_clusters

    def forward(self, vehicles, global_state, mask=None):
        batch_size = len(vehicles)

        # Embed vehicle features
        vehicle_embeds = torch.stack([self.vehicle_embedding(v) for v in vehicles])

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Assign vehicles to clusters
        cluster_logits = self.cluster_network(vehicle_embeds)
        cluster_probs = F.softmax(cluster_logits, dim=1)

        # Compute cluster representations
        cluster_reps = torch.zeros(self.num_clusters, vehicle_embeds.size(1), device=device)
        for c in range(self.num_clusters):
            # Weighted sum of vehicle embeddings based on cluster assignment
            weights = cluster_probs[:, c].unsqueeze(1)
            cluster_reps[c] = (vehicle_embeds * weights).sum(0)

        # Select clusters based on global state
        cluster_info = torch.cat([state_embed, cluster_probs.mean(0)], dim=0)
        cluster_selection_logits = self.cluster_selection(cluster_info)
        cluster_selection_probs = F.softmax(cluster_selection_logits, dim=0)

        # Select vehicles based on cluster selection
        selection_logits = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            # Compute vehicle selection probability based on cluster assignment
            vehicle_cluster_probs = cluster_probs[i]
            vehicle_selection = 0

            for c in range(self.num_clusters):
                # Combine vehicle embedding with cluster representation
                combined = torch.cat([vehicle_embeds[i], cluster_reps[c]], dim=0)

                # Get selection score for this vehicle in this cluster
                score = self.vehicle_selection(combined).item()

                # Weight score by cluster assignment and cluster selection
                vehicle_selection += score * vehicle_cluster_probs[c] * cluster_selection_probs[c]

            selection_logits[i] = vehicle_selection

        # Apply mask if provided
        if mask is not None:
            mask_value = -65504.0 if selection_logits.dtype == torch.float16 else -1e9
            selection_logits = selection_logits.masked_fill(mask == 0, mask_value)

        # Return selection probabilities
        return F.softmax(selection_logits, dim=0)

################################## 6. Attention-Based Scheduler ##################################

class AttentionScheduler(nn.Module):
    """
    Pure attention-based scheduler without transformer architecture.
    Uses multi-head attention to model relationships between vehicles.
    """
    def __init__(self, input_dim, state_dim, d_model=256, num_heads=8):
        super(AttentionScheduler, self).__init__()

        # Vehicle feature embedding
        self.vehicle_embedding = nn.Linear(input_dim, d_model)

        # Global state embedding
        self.state_embedding = nn.Linear(state_dim, d_model)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, num_heads)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Action head for vehicle selection probabilities
        self.action_head = nn.Linear(d_model, 1)

    def forward(self, vehicles, global_state, mask=None):
        batch_size = len(vehicles)

        # Embed vehicle features
        vehicle_embeds = torch.stack([self.vehicle_embedding(v) for v in vehicles])

        # Embed global state
        state_embed = self.state_embedding(global_state)

        # Add global state to each vehicle embedding
        state_expanded = state_embed.unsqueeze(0).expand(batch_size, -1)
        combined_embeds = vehicle_embeds + state_expanded

        # For MultiheadAttention, input should be [seq_len, batch_size, embed_dim]
        # So we need to transpose from [batch_size, embed_dim] to [1, batch_size, embed_dim]
        combined_embeds_t = combined_embeds.unsqueeze(0)  # [1, batch_size, embed_dim]

        # Self-attention without mask (we'll apply mask after)
        attn_output, _ = self.attention(
            combined_embeds_t, combined_embeds_t, combined_embeds_t
        )

        # Transpose back to [batch_size, embed_dim]
        attn_output = attn_output.squeeze(0)

        # Add & norm
        combined_embeds = self.norm1(combined_embeds + attn_output)

        # Feed-forward
        ffn_output = self.ffn(combined_embeds)

        # Add & norm
        combined_embeds = self.norm2(combined_embeds + ffn_output)

        # Generate selection probabilities
        logits = self.action_head(combined_embeds).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            mask_value = -65504.0 if logits.dtype == torch.float16 else -1e9
            logits = logits.masked_fill(mask == 0, mask_value)

        # Return selection probabilities
        return F.softmax(logits, dim=0)

################################## 7. Heuristic-Based Scheduler ##################################

class HeuristicScheduler:
    """
    Heuristic-based scheduler that uses domain knowledge.
    Combines multiple heuristics with learned weights.
    """
    def __init__(self, heuristic_weights=None):
        # Default heuristic weights if not provided
        if heuristic_weights is None:
            self.heuristic_weights = {
                'data_quality': 0.3,
                'compute_capacity': 0.2,
                'connectivity': 0.2,
                'sojourn_time': 0.15,
                'model_version': 0.15
            }
        else:
            self.heuristic_weights = heuristic_weights

    def select_vehicles(self, vehicles, target_count=10, min_sojourn_time=1.0):
        """Select vehicles based on weighted heuristics"""
        # Filter vehicles with sufficient sojourn time
        eligible_vehicles = [v for v in vehicles if v.sojourn_time >= min_sojourn_time and not v.scheduled]

        if not eligible_vehicles:
            return []

        # Calculate scores for each vehicle
        scores = []
        for vehicle in eligible_vehicles:
            score = (
                self.heuristic_weights['data_quality'] * vehicle.data_quality +
                self.heuristic_weights['compute_capacity'] * vehicle.compute_capacity +
                self.heuristic_weights['connectivity'] * vehicle.connectivity +
                self.heuristic_weights['sojourn_time'] * min(1.0, vehicle.sojourn_time / 10.0) +
                self.heuristic_weights['model_version'] * (1.0 - vehicle.model_version)  # Lower version = higher priority
            )
            scores.append((vehicle, score))

        # Sort by score in descending order
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select top vehicles
        selected_count = min(target_count, len(eligible_vehicles))
        selected_vehicles = [item[0] for item in scores[:selected_count]]

        return selected_vehicles

################################## 8. Hybrid Scheduler ##################################

class HybridScheduler:
    """
    Hybrid scheduler that combines ML models with heuristics.
    Uses ML for complex patterns and heuristics for interpretability.
    """
    def __init__(self, ml_model, heuristic_scheduler, ml_weight=0.7):
        self.ml_model = ml_model
        self.heuristic_scheduler = heuristic_scheduler
        self.ml_weight = ml_weight

    def select_vehicles(self, vehicles, global_state, target_count=10):
        """Select vehicles using a hybrid approach"""
        # Get eligible vehicles
        eligible_vehicles = [v for v in vehicles if not v.scheduled]

        if not eligible_vehicles:
            return []

        # Create mask for eligible vehicles
        mask = torch.ones(len(eligible_vehicles)).to(device)
        for i, vehicle in enumerate(eligible_vehicles):
            if vehicle.sojourn_time < 1.0:  # Minimum required time
                mask[i] = 0

        # Get ML model probabilities
        with torch.no_grad():
            vehicle_tensors = [v.to_tensor() for v in eligible_vehicles]
            ml_probs = self.ml_model(vehicle_tensors, global_state.to_tensor(), mask)

        # Get heuristic scores
        heuristic_vehicles = self.heuristic_scheduler.select_vehicles(eligible_vehicles, target_count=len(eligible_vehicles))

        # Create heuristic probabilities (1 for selected, 0 for others)
        heuristic_probs = torch.zeros(len(eligible_vehicles)).to(device)
        for i, vehicle in enumerate(eligible_vehicles):
            if vehicle in heuristic_vehicles:
                # Assign probability based on position in heuristic selection
                pos = heuristic_vehicles.index(vehicle)
                heuristic_probs[i] = 1.0 - (pos / len(heuristic_vehicles))

        # Normalize heuristic probabilities
        if heuristic_probs.sum() > 0:
            heuristic_probs = heuristic_probs / heuristic_probs.sum()

        # Combine probabilities
        combined_probs = self.ml_weight * ml_probs + (1 - self.ml_weight) * heuristic_probs

        # Apply mask
        combined_probs = combined_probs * mask

        # Normalize combined probabilities
        if combined_probs.sum() > 0:
            combined_probs = combined_probs / combined_probs.sum()

        # Select vehicles based on combined probabilities
        selected_indices = []
        remaining_indices = list(range(len(eligible_vehicles)))
        remaining_indices = [i for i in remaining_indices if mask[i] > 0]

        # Select up to target_count vehicles
        for _ in range(min(target_count, len(remaining_indices))):
            if not remaining_indices:
                break

            # Normalize probabilities for remaining vehicles
            probs = combined_probs[remaining_indices]
            probs = probs / probs.sum()

            # Sample based on probabilities
            idx = np.random.choice(len(remaining_indices), p=probs.cpu().numpy())
            selected_idx = remaining_indices[idx]
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)

        # Return selected vehicles
        selected_vehicles = [eligible_vehicles[i] for i in selected_indices]
        return selected_vehicles

################################## Comparison Utilities ##################################

def compare_schedulers(environment, schedulers, num_episodes=10, max_rounds=100):
    """
    Compare multiple schedulers on the same environment.
    Returns performance metrics for each scheduler.
    """
    results = {}

    for name, scheduler in schedulers.items():
        print(f"Evaluating {name}...")

        episode_rewards = []
        episode_performances = []
        episode_lengths = []
        decision_times = []

        for episode in range(num_episodes):
            state = environment.reset()
            total_reward = 0

            for round_num in range(max_rounds):
                # Get new vehicles
                new_vehicles = environment.get_new_vehicles()

                # Make scheduling decision
                start_time = time.time()
                selected_vehicles = scheduler.select_vehicles(environment.active_vehicles)
                decision_time = (time.time() - start_time) * 1000  # ms

                # Take step in environment
                next_state, reward, done, info = environment.step([v.vehicle_id for v in selected_vehicles])

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

def plot_comparison_results(results, save_path=None):
    """
    Plot comparison results between different schedulers.
    """
    import matplotlib.pyplot as plt

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot average reward
    rewards = [results[name]['avg_reward'] for name in results]
    axes[0, 0].bar(results.keys(), rewards)
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot average performance
    performances = [results[name]['avg_performance'] for name in results]
    axes[0, 1].bar(results.keys(), performances)
    axes[0, 1].set_title('Average Performance')
    axes[0, 1].set_ylabel('Performance')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot average episode length
    lengths = [results[name]['avg_length'] for name in results]
    axes[1, 0].bar(results.keys(), lengths)
    axes[1, 0].set_title('Average Episode Length')
    axes[1, 0].set_ylabel('Rounds')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot average decision time
    times = [results[name]['avg_decision_time'] for name in results]
    axes[1, 1].bar(results.keys(), times)
    axes[1, 1].set_title('Average Decision Time')
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

if __name__ == "__main__":
    print("Scheduler comparison methods implemented.")
    print("Import this module and use the compare_schedulers function to evaluate different approaches.")
