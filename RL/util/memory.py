import torch
import torch.nn.functional as F

import pdb
from scipy.spatial.distance import cosine
import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from model.net import *
from model.multipatchformer import *


def custom_kernel(h, hi):
    squared_distance = np.sum((h - hi) ** 2)
    return 1 / (squared_distance + 1e-3)

class episodicmemory():
    def __init__(self, capacity, k, state_dim, state_dim_2, hidden_dim, device):
        self.capacity = capacity
        self.k = k
        self.current_size = 0
        self.count = 0
        self.device = device
        self.buffer = {"single_state": np.zeros((self.capacity, state_dim)),
                        "trend_state": np.zeros((self.capacity, state_dim_2)),
                        "previous_action": np.zeros((self.capacity)),
                        "hidden_state": np.zeros((self.capacity, hidden_dim)),
                        "action": np.zeros((self.capacity)),
                        "q_value": np.zeros((self.capacity))
                       }

    def add(self, hidden_state, action, q_value, single_state, trend_state, previous_action):
        self.buffer["single_state"][self.count] = single_state
        self.buffer["trend_state"][self.count] = trend_state
        self.buffer["previous_action"][self.count] = previous_action
        self.buffer["hidden_state"][self.count] = hidden_state
        self.buffer["action"][self.count] = action
        self.buffer["q_value"][self.count] = q_value
        self.count = (self.count + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def query(self, query_hidden_state, action):
        if self.current_size != self.capacity:
            weighted_q_value = np.nan
        else:
            kernel_values = np.array([custom_kernel(query_hidden_state, hs) for hs in self.buffer["hidden_state"]])
            top_k_indices = np.argsort(kernel_values)[-self.k:]
            top_k_actions = self.buffer["action"][top_k_indices]
            top_k_q_values = self.buffer["q_value"][top_k_indices]
            mask = (top_k_actions == action).astype(float)
            weights = kernel_values[top_k_indices] / np.sum(kernel_values[top_k_indices])
            masked_weights = weights * mask
            normalized_weights = masked_weights / np.sum(masked_weights)
            weighted_q_value = np.dot(normalized_weights, top_k_q_values)

        return weighted_q_value

    def re_encode(self, model):
        batch_size = 512
        for i in range(0, self.capacity, batch_size):
            batch_end = min(i + batch_size, self.capacity)
            single_states_batch = torch.tensor(self.buffer["single_state"][i:batch_end], dtype=torch.float32).to(self.device)
            trend_states_batch = torch.tensor(self.buffer["trend_state"][i:batch_end], dtype=torch.float32).to(self.device)
            previous_actions_batch = torch.tensor(self.buffer["previous_action"][i:batch_end], dtype=torch.long).to(self.device)
            with torch.no_grad():
                updated_hidden_states = model.encode(single_states_batch, trend_states_batch, previous_actions_batch).cpu().numpy()
            self.buffer["hidden_state"][i:batch_end] = updated_hidden_states


class SequenceEpisodicMemory():
    """Episodic Memory for sequence data"""
    
    def __init__(self, capacity, k, state_dim, state_dim_2, hidden_dim, device, seq_len=20):
        self.capacity = capacity
        self.k = k
        self.seq_len = seq_len
        self.current_size = 0
        self.count = 0
        self.device = device
        
        # Buffer for sequence data
        self.buffer = {
            "single_state": np.zeros((self.capacity, seq_len, state_dim)),      # Sequence of single states
            "trend_state": np.zeros((self.capacity, seq_len, state_dim_2)),     # Sequence of trend states
            "previous_action": np.zeros((self.capacity)),                       # Previous action (scalar)
            "hidden_state": np.zeros((self.capacity, hidden_dim)),              # Hidden state representation
            "action": np.zeros((self.capacity)),                                # Current action
            "q_value": np.zeros((self.capacity))                                # Q-value
        }
        
        print(f"SequenceEpisodicMemory initialized:")
        print(f"  Capacity: {self.capacity}")
        print(f"  Top-k: {self.k}")
        print(f"  Sequence length: {self.seq_len}")
        print(f"  Single state shape: ({seq_len}, {state_dim})")
        print(f"  Trend state shape: ({seq_len}, {state_dim_2})")
        print(f"  Hidden dimension: {hidden_dim}")

    def add(self, hidden_state, action, q_value, single_state, trend_state, previous_action):

        # Shape assertions for sequence data
        assert single_state.shape == (self.seq_len, self.buffer["single_state"].shape[2]), \
            f"Single state shape mismatch: expected {(self.seq_len, self.buffer['single_state'].shape[2])}, got {single_state.shape}"
        
        assert trend_state.shape == (self.seq_len, self.buffer["trend_state"].shape[2]), \
            f"Trend state shape mismatch: expected {(self.seq_len, self.buffer['trend_state'].shape[2])}, got {trend_state.shape}"
        
        assert hidden_state.shape == (self.buffer["hidden_state"].shape[1],), \
            f"Hidden state shape mismatch: expected ({self.buffer['hidden_state'].shape[1]},), got {hidden_state.shape}"
        
        # Store the experience
        self.buffer["single_state"][self.count] = single_state
        self.buffer["trend_state"][self.count] = trend_state
        self.buffer["previous_action"][self.count] = previous_action
        self.buffer["hidden_state"][self.count] = hidden_state
        self.buffer["action"][self.count] = action
        self.buffer["q_value"][self.count] = q_value
        
        # Update counters
        self.count = (self.count + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def query(self, query_hidden_state, action):

        if self.current_size != self.capacity:
            # Return NaN if memory is not full yet
            weighted_q_value = np.nan
        else:
            # Compute similarity with all stored hidden states
            kernel_values = np.array([
                custom_kernel(query_hidden_state, hs) 
                for hs in self.buffer["hidden_state"]
            ])
            
            # Get top-k most similar experiences
            top_k_indices = np.argsort(kernel_values)[-self.k:]
            top_k_actions = self.buffer["action"][top_k_indices]
            top_k_q_values = self.buffer["q_value"][top_k_indices]
            
            # Create mask for matching actions
            mask = (top_k_actions == action).astype(float)
            
            # Check if any experiences match the action
            if np.sum(mask) == 0:
                # No matching actions found
                weighted_q_value = np.nan
            else:
                # Compute weights based on similarity
                weights = kernel_values[top_k_indices] / np.sum(kernel_values[top_k_indices])
                
                # Apply action mask
                masked_weights = weights * mask
                
                # Normalize weights
                normalized_weights = masked_weights / np.sum(masked_weights)
                
                # Compute weighted Q-value
                weighted_q_value = np.dot(normalized_weights, top_k_q_values)
        
        return weighted_q_value

    def re_encode(self, model):

        batch_size = 512
        
        print(f"Re-encoding {self.capacity} experiences in batches of {batch_size}...")
        
        for i in range(0, self.capacity, batch_size):
            batch_end = min(i + batch_size, self.capacity)
            
            # Prepare batch data
            single_states_batch = torch.tensor(
                self.buffer["single_state"][i:batch_end], 
                dtype=torch.float32
            ).to(self.device)
            
            trend_states_batch = torch.tensor(
                self.buffer["trend_state"][i:batch_end], 
                dtype=torch.float32
            ).to(self.device)
            
            previous_actions_batch = torch.tensor(
                self.buffer["previous_action"][i:batch_end], 
                dtype=torch.long
            ).to(self.device)
            
            # Re-encode with updated model
            with torch.no_grad():
                updated_hidden_states = model.encode(
                    single_states_batch, 
                    trend_states_batch, 
                    previous_actions_batch
                ).cpu().numpy()
            
            # Update buffer
            self.buffer["hidden_state"][i:batch_end] = updated_hidden_states
            
            if i % (batch_size * 10) == 0:  # Progress logging
                progress = min(batch_end / self.capacity * 100, 100)
                print(f"  Progress: {progress:.1f}%")
        
        print("Re-encoding completed!")

    def get_statistics(self):
        """Get memory statistics for monitoring"""
        if self.current_size == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "is_full": False
            }
        
        stats = {
            "size": self.current_size,
            "capacity": self.capacity,
            "utilization": self.current_size / self.capacity,
            "is_full": self.current_size == self.capacity,
            "avg_q_value": np.mean(self.buffer["q_value"][:self.current_size]),
            "q_value_std": np.std(self.buffer["q_value"][:self.current_size]),
            "action_distribution": {}
        }
        
        # Action distribution
        if self.current_size > 0:
            actions = self.buffer["action"][:self.current_size]
            unique_actions, counts = np.unique(actions, return_counts=True)
            for action, count in zip(unique_actions, counts):
                stats["action_distribution"][int(action)] = int(count)
        
        return stats

    def get_similar_experiences(self, query_hidden_state, top_k=None):

        if top_k is None:
            top_k = self.k
            
        if self.current_size == 0:
            return {"experiences": [], "similarities": []}
        
        # Compute similarities
        kernel_values = np.array([
            custom_kernel(query_hidden_state, hs) 
            for hs in self.buffer["hidden_state"][:self.current_size]
        ])
        
        # Get top-k indices
        num_experiences = min(top_k, self.current_size)
        top_indices = np.argsort(kernel_values)[-num_experiences:][::-1]  # Descending order
        
        # Extract experiences
        experiences = []
        similarities = []
        
        for idx in top_indices:
            experience = {
                "index": idx,
                "action": self.buffer["action"][idx],
                "q_value": self.buffer["q_value"][idx],
                "previous_action": self.buffer["previous_action"][idx],
                "single_state": self.buffer["single_state"][idx].copy(),
                "trend_state": self.buffer["trend_state"][idx].copy(),
                "hidden_state": self.buffer["hidden_state"][idx].copy()
            }
            experiences.append(experience)
            similarities.append(kernel_values[idx])
        
        return {
            "experiences": experiences,
            "similarities": similarities
        }

    def __len__(self):
        """Return current size of memory"""
        return self.current_size

    def is_full(self):
        """Check if memory is at full capacity"""
        return self.current_size == self.capacity