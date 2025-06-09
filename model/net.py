import torch
import torch.nn as nn
import torch.nn.functional as F
import math

max_punish = 1e12

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class NoisyLinear(nn.Module):
    """Noisy Linear layer for Rainbow DQN"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Non-learnable noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class subagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim, num_atoms=51, v_min=-10.0, v_max=10.0, use_noisy=True):
        super(subagent, self).__init__()
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.use_noisy = use_noisy
        
        self.fc1 = nn.Linear(state_dim_1, hidden_dim)
        self.fc2 = nn.Linear(state_dim_2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(self.action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

        # Advantage stream with optional noisy layer
        if use_noisy:
            self.advantage = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(approximate="tanh"),
                NoisyLinear(hidden_dim * 4, action_dim * num_atoms)
            )
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(approximate="tanh"),
                NoisyLinear(hidden_dim * 4, num_atoms)
            )
        else:
            self.advantage = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim * 4, action_dim * num_atoms)
            )
            self.value = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim * 4, num_atoms)
            )

        self.register_buffer("max_punish", torch.tensor(max_punish))
        support = torch.linspace(v_min, v_max, num_atoms)
        self.register_buffer('support', support)

    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.use_noisy:
            if hasattr(self.advantage[-1], 'reset_noise'):
                self.advantage[-1].reset_noise()
            if hasattr(self.value[-1], 'reset_noise'):
                self.value[-1].reset_noise()

    def forward(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                previous_action: torch.tensor,):
        if previous_action.dtype != torch.long:
            previous_action = previous_action.long()

        action_hidden = self.embedding(previous_action)
        single_state_hidden = self.fc1(single_state)
        trend_state_hidden = self.fc2(trend_state)
        c = action_hidden + trend_state_hidden
        
        ada_linear_layer = self.adaLN_modulation[-1]

        modulated_c = self.adaLN_modulation(c)
        
        shift, scale = modulated_c.chunk(2, dim=-1) 

        x = modulate(self.norm(single_state_hidden), shift, scale)
        
        value_logits = self.value(x).view(-1, 1, self.num_atoms)
        advantage_logits = self.advantage(x).view(-1, self.action_dim, self.num_atoms)
        
        q_logits = value_logits + advantage_logits - advantage_logits.mean(dim=1, keepdim=True)
        
        return q_logits


class hyperagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim):
        super(hyperagent, self).__init__()
        self.fc1 = nn.Linear(state_dim_1 + state_dim_2, hidden_dim)
        self.fc2 = nn.Linear(2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 6),
            nn.Softmax(dim=1)
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                class_state: torch.tensor,
                previous_action: torch.tensor,):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        c = self.fc2(class_state)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        weight = self.net(x)
        
        return weight

    def encode(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                previous_action: torch.tensor,):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        return x
