import torch
import pdb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import sys
import os

max_punish = 1e12

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :].to(x.device)

class MultiHeadAttentionEncoder(nn.Module):
    """Multi-Head Attention encoder for sequence processing"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, seq_len=20):
        super(MultiHeadAttentionEncoder, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # attention Layer normalization
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        # ffn layer normalization
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply attention layers
        for i, (attn, attn_norm, ffn, ffn_norm) in enumerate(zip(self.attention_layers, self.attention_norms, self.ffns, self.ffn_norms)):
            # Multi-head attention with residual connection
            attn_out, _ = attn(x, x, x)
            x = attn_norm(x + attn_out)
            
            # Feed-forward with residual connection
            ffn_out = ffn(x)
            x = ffn_norm(x + ffn_out)
        
        x = self.final_norm(x)
        
        # Return: (batch_size, seq_len, d_model) and aggregated representation
        # Use mean pooling for aggregated representation
        aggregated = x.mean(dim=1)  # (batch_size, d_model)
        
        return x, aggregated

class ConvPatchEmbedding(nn.Module):
    """Convert sequence into patches using different convolutions"""
    def __init__(self, input_dim, d_model=128, seq_len=20):
        super(ConvPatchEmbedding, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Different convolution configurations for different patch sizes
        # Each creates patches of different temporal scales
        self.conv_configs = [
            {'kernel_size': 2, 'stride': 1, 'name': 'fine'},      # Fine-grained: overlapping small patches
            {'kernel_size': 4, 'stride': 2, 'name': 'medium'},    # Medium-grained: moderate patches
            {'kernel_size': 8, 'stride': 4, 'name': 'coarse'},    # Coarse-grained: large patches
            {'kernel_size': 5, 'stride': 3, 'name': 'irregular'}, # Irregular: different stride/kernel ratio
        ]
        
        # Create convolution layers for each configuration
        self.patch_convs = nn.ModuleDict()
        self.patch_projections = nn.ModuleDict()
        self.total_patches = 0
        
        for config in self.conv_configs:
            name = config['name']
            kernel_size = config['kernel_size']
            stride = config['stride']
            
            # Calculate output length: (seq_len - kernel_size) // stride + 1
            output_len = (seq_len - kernel_size) // stride + 1
            self.total_patches += output_len
            
            # 1D convolution for patch extraction
            self.patch_convs[name] = nn.Conv1d(
                in_channels=input_dim,
                out_channels=d_model,
                kernel_size=kernel_size,
                stride=stride,
                padding=0
            )
            
            # Additional projection layer
            self.patch_projections[name] = nn.Linear(d_model, d_model)
            
            print(f"Patch {name}: kernel={kernel_size}, stride={stride}, output_patches={output_len}")
        
        # Learnable position embeddings for all patches
        self.position_embeddings = nn.Parameter(torch.randn(1, self.total_patches, d_model))
        
        # Patch type embeddings to distinguish different patch types
        self.patch_type_embeddings = nn.Parameter(torch.randn(len(self.conv_configs), d_model))
        
        print(f"Total patches: {self.total_patches}")
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Transpose for 1D convolution: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        all_patches = []
        patch_idx = 0
        
        for i, config in enumerate(self.conv_configs):
            name = config['name']
            
            # Apply convolution to extract patches
            patches = self.patch_convs[name](x)  # (batch_size, d_model, num_patches)
            patches = patches.transpose(1, 2)    # (batch_size, num_patches, d_model)
            
            # Apply projection
            patches = self.patch_projections[name](patches)
            
            # Add patch type embedding
            patches = patches + self.patch_type_embeddings[i].unsqueeze(0).unsqueeze(0)
            
            # Add positional embedding for this patch group
            num_patches = patches.size(1)
            patches = patches + self.position_embeddings[:, patch_idx:patch_idx+num_patches, :]
            patch_idx += num_patches
            
            all_patches.append(patches)
        
        # Concatenate all patches along sequence dimension
        combined_patches = torch.cat(all_patches, dim=1)  # (batch_size, total_patches, d_model)
        
        return combined_patches

class MultiPatchFormer(nn.Module):
    """Multi-Patch Former using convolutions with different strides and kernels"""
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, seq_len=20):
        super(MultiPatchFormer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Convolution-based patch embedding
        self.patch_embedding = ConvPatchEmbedding(input_dim, d_model, seq_len)
        
        # Multi-head attention transformer layers
        self.encoder_layer = MultiHeadAttentionEncoder(d_model, d_model, nhead, num_layers, self.patch_embedding.total_patches)

        # Global pooling strategies
        self.pooling_type = 'attention'  # Options: 'mean', 'max', 'attention'
        
        if self.pooling_type == 'attention':
            # Learnable attention pooling
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=0.1,
                batch_first=True
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Convert to patches using different convolutions
        patches = self.patch_embedding(x)  # (batch_size, total_patches, d_model)
        
        # Apply multi-head attention across patches
        attended_patches, _ = self.encoder_layer(patches)  # (batch_size, total_patches, d_model)
        
        # Global aggregation
        if self.pooling_type == 'mean':
            aggregated = attended_patches.mean(dim=1)
        elif self.pooling_type == 'max':
            aggregated = attended_patches.max(dim=1)[0]
        elif self.pooling_type == 'attention':
            # Use learnable attention pooling
            pool_query = self.pool_query.expand(batch_size, -1, -1)
            aggregated, _ = self.attention_pool(
                pool_query, attended_patches, attended_patches
            )
            aggregated = aggregated.squeeze(1)  # (batch_size, d_model)
        
        # Final projection
        aggregated = self.output_projection(aggregated)
        
        return attended_patches, aggregated

class subagent_with_conv_patches(nn.Module):
    """SubAgent with Convolution-based Multi-Patch Former"""
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim, seq_len=20):
        super(subagent_with_conv_patches, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Convolution-based Multi-Patch Former encoders
        self.single_encoder = MultiPatchFormer(
            input_dim=state_dim_1, d_model=hidden_dim, seq_len=seq_len
        )
        self.trend_encoder = MultiPatchFormer(
            input_dim=state_dim_2, d_model=hidden_dim, seq_len=seq_len
        )
        
        # Action embedding
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        
        # Normalization and modulation
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        
        # Value and advantage networks
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, 1)
        )

        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(self,
                single_state: torch.tensor,  # (batch_size, seq_len, state_dim_1)
                trend_state: torch.tensor,   # (batch_size, seq_len, state_dim_2)
                previous_action: torch.tensor):  # (batch_size,)
        
        # Process sequences with convolution-based patches + attention
        _, single_aggregated = self.single_encoder(single_state)  # (batch_size, hidden_dim)
        _, trend_aggregated = self.trend_encoder(trend_state)     # (batch_size, hidden_dim)
        
        # Action embedding
        action_hidden = self.embedding(previous_action)  # (batch_size, hidden_dim)
        
        # Combine trend and action for modulation
        c = action_hidden + trend_aggregated
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        # Apply modulation to single state representation
        x = modulate(self.norm(single_aggregated), shift, scale)
        
        # Compute value and advantage
        value = self.value(x)
        advantage = self.advantage(x)
        
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class hyperagent_with_conv_patches(nn.Module):
    """HyperAgent with Convolution-based Multi-Patch Former"""
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim, seq_len=20):
        super(hyperagent_with_conv_patches, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Convolution-based Multi-Patch Former encoder
        self.combined_encoder = MultiPatchFormer(
            input_dim=state_dim_1 + state_dim_2, d_model=hidden_dim, seq_len=seq_len
        )
        
        # Class state processing (non-sequential)
        self.fc2 = nn.Linear(2, hidden_dim)
        
        # Normalization and modulation
        self.norm = nn.LayerNorm(hidden_dim * 2, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        
        # Output network
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, 6),
            nn.Softmax(dim=1)
        )
        
        # Initialize last layer to zero
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self,
                single_state: torch.tensor,    # (batch_size, seq_len, state_dim_1)
                trend_state: torch.tensor,     # (batch_size, seq_len, state_dim_2)
                class_state: torch.tensor,     # (batch_size, 2)
                previous_action: torch.tensor): # (batch_size,)
        
        # Combine and process sequences with convolution-based patches + attention
        combined_state = torch.cat([single_state, trend_state], dim=-1)  # (batch_size, seq_len, state_dim_1 + state_dim_2)
        _, state_aggregated = self.combined_encoder(combined_state)      # (batch_size, hidden_dim)
        
        # Action embedding
        action_hidden = self.embedding(previous_action)  # (batch_size, hidden_dim)
        
        # Combine action and state representations
        x = torch.cat([action_hidden, state_aggregated], dim=1)  # (batch_size, hidden_dim * 2)
        
        # Class state processing for modulation
        c = self.fc2(class_state)  # (batch_size, hidden_dim)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        
        # Apply modulation
        x = modulate(self.norm(x), shift, scale)
        
        # Generate weights
        weight = self.net(x)  # (batch_size, 6)
        
        return weight

    def encode(self,
               single_state: torch.tensor,    # (batch_size, seq_len, state_dim_1)
               trend_state: torch.tensor,     # (batch_size, seq_len, state_dim_2)
               previous_action: torch.tensor): # (batch_size,)
        
        # Combine and process sequences
        combined_state = torch.cat([single_state, trend_state], dim=-1)
        _, state_aggregated = self.combined_encoder(combined_state)
        
        # Action embedding
        action_hidden = self.embedding(previous_action)
        
        # Combine representations
        x = torch.cat([action_hidden, state_aggregated], dim=1)
        
        return x

def calculate_q(w, qs):
    """Calculate combined Q-values using weights"""
    q_tensor = torch.stack(qs)
    q_tensor = q_tensor.permute(1, 0, 2)
    weights_reshaped = w.view(-1, 1, 6)
    combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
    
    return combined_q

# Example usage and testing
def test_conv_patch_agents():
    """Test the convolution-based patch agents"""
    batch_size = 32
    seq_len = 20
    state_dim_1 = 8   # e.g., tech indicators
    state_dim_2 = 3   # e.g., trend indicators
    action_dim = 2
    hidden_dim = 128
    
    # Create test data
    single_state = torch.randn(batch_size, seq_len, state_dim_1)
    trend_state = torch.randn(batch_size, seq_len, state_dim_2)
    class_state = torch.randn(batch_size, 2)
    previous_action = torch.randint(0, action_dim, (batch_size,))
    
    print("Testing Convolution-based Multi-Patch Former...")
    print(f"Input shapes: single_state={single_state.shape}, trend_state={trend_state.shape}")
    
    # Test subagent with convolution patches
    print("\n=== SubAgent with Conv Patches ===")
    subagent_conv = subagent_with_conv_patches(
        state_dim_1, state_dim_2, action_dim, hidden_dim, seq_len
    )
    
    q_values = subagent_conv(single_state, trend_state, previous_action)
    print(f"SubAgent Q-values shape: {q_values.shape}")
    print(f"Q-values sample: {q_values[0]}")
    
    # Test hyperagent with convolution patches
    print("\n=== HyperAgent with Conv Patches ===")
    hyperagent_conv = hyperagent_with_conv_patches(
        state_dim_1, state_dim_2, action_dim, hidden_dim, seq_len
    )
    
    weights = hyperagent_conv(single_state, trend_state, class_state, previous_action)
    print(f"HyperAgent weights shape: {weights.shape}")
    print(f"Weights sample: {weights[0]}")
    print(f"Weights sum: {weights[0].sum()} (should be ~1.0)")
    
    # Test encoding
    print("\n=== Encoding Test ===")
    encoded = hyperagent_conv.encode(single_state, trend_state, previous_action)
    print(f"Encoded representation shape: {encoded.shape}")
    
    print("\nAll convolution-based patch tests passed!")

def demo_patch_extraction():
    """Demonstrate how different convolutions create different patches"""
    print("\n=== Patch Extraction Demo ===")
    
    input_dim = 8
    seq_len = 20
    d_model = 64
    batch_size = 2
    
    # Create demo input
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"Input shape: {x.shape}")
    
    # Create patch embedding
    patch_embed = ConvPatchEmbedding(input_dim, d_model, seq_len)
    
    # Extract patches
    patches = patch_embed(x)
    print(f"Total patches extracted: {patches.shape[1]}")
    print(f"Patch feature dimension: {patches.shape[2]}")

if __name__ == "__main__":
    # Test the convolution-based patch agents
    test_conv_patch_agents()
    
    # Demonstrate patch extraction
    demo_patch_extraction()