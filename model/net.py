import torch
import torch.nn as nn

max_punish = 1e12


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class subagent(nn.Module):
    def __init__(
        self, state_dim_1, state_dim_2, action_dim, hidden_dim, num_quantiles
    ):  # Added num_quantiles
        super(subagent, self).__init__()
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.fc1 = nn.Linear(state_dim_1, hidden_dim)
        self.fc2 = nn.Linear(state_dim_2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(
            action_dim, hidden_dim
        )  # action_dim here is for previous_action embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        # Advantage stream outputs action_dim * num_quantiles
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, action_dim * num_quantiles),
        )
        # Value stream outputs num_quantiles
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, num_quantiles),
        )

        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(
        self,
        single_state: torch.tensor,
        trend_state: torch.tensor,
        previous_action: torch.tensor,
    ):
        action_hidden = self.embedding(previous_action)
        single_state_hidden = self.fc1(single_state)
        trend_state_hidden = self.fc2(trend_state)
        c = action_hidden + trend_state_hidden
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(single_state_hidden), shift, scale)

        value_quantiles = self.value(x)  # Shape: (batch_size, num_quantiles)
        advantage_quantiles = self.advantage(
            x
        )  # Shape: (batch_size, action_dim * num_quantiles)

        # Reshape advantage quantiles
        advantage_quantiles = advantage_quantiles.view(
            -1, self.action_dim, self.num_quantiles
        )

        # Expand value_quantiles for broadcasting
        value_quantiles = value_quantiles.unsqueeze(
            1
        )  # Shape: (batch_size, 1, num_quantiles)

        # Combine value and advantage streams (Dueling architecture for quantiles)
        # Q(s,a,tau_i) = V(s,tau_i) + (A(s,a,tau_i) - mean_a'(A(s,a',tau_i)))
        q_quantiles = (
            value_quantiles
            + advantage_quantiles
            - advantage_quantiles.mean(dim=1, keepdim=True)
        )

        return q_quantiles  # Shape: (batch_size, action_dim, num_quantiles)


class hyperagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim):
        super(hyperagent, self).__init__()
        self.fc1 = nn.Linear(state_dim_1 + state_dim_2, hidden_dim)
        self.fc2 = nn.Linear(2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 6),
            nn.Softmax(dim=1),
        )
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(
        self,
        single_state: torch.tensor,
        trend_state: torch.tensor,
        class_state: torch.tensor,
        previous_action: torch.tensor,
    ):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        c = self.fc2(class_state)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        weight = self.net(x)

        return weight

    def encode(
        self,
        single_state: torch.tensor,
        trend_state: torch.tensor,
        previous_action: torch.tensor,
    ):
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        return x


def calculate_q(w, qs):
    q_tensor = torch.stack(qs)
    q_tensor = q_tensor.permute(1, 0, 2)
    weights_reshaped = w.view(-1, 1, 6)
    combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)

    return combined_q
