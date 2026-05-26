import torch
import torch.nn as nn
import torch.nn.functional as F

max_punish = 1e12

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class subagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim):
        super(subagent, self).__init__()
        self.fc1 = nn.Linear(state_dim_1, hidden_dim)
        self.fc2 = nn.Linear(state_dim_2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 1)
        )

        self.register_buffer("max_punish", torch.tensor(max_punish))

    def forward(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                previous_action: torch.tensor,):
        action_hidden = self.embedding(previous_action)
        single_state_hidden = self.fc1(single_state)
        trend_state_hidden = self.fc2(trend_state)
        c = action_hidden + trend_state_hidden
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(single_state_hidden), shift, scale)
        value = self.value(x)
        advantage = self.advantage(x)
        
        return value + advantage - advantage.mean()


class MarketStateAnalyzer(nn.Module):
    """市場狀態分析器，用於決定使用軟混合還是硬選擇"""
    def __init__(self, input_dim, hidden_dim=64):
        super(MarketStateAnalyzer, self).__init__()
        self.analyzer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 輸出[0,1]，0偏向硬選擇，1偏向軟混合
        )
        
        # 波動性和不確定性閾值（可學習參數）
        self.volatility_threshold = nn.Parameter(torch.tensor(0.02))
        self.uncertainty_threshold = nn.Parameter(torch.tensor(0.8))
        
    def forward(self, state_features, market_conditions):
        """
        state_features: 編碼後的狀態特徵 [batch_size, feature_dim]
        market_conditions: [slope, volatility] [batch_size, 2]
        """
        # 組合輸入
        combined_input = torch.cat([state_features, market_conditions], dim=-1)
        
        # 基於神經網絡的混合策略分數
        mixing_score = self.analyzer(combined_input)
        
        # 基於規則的調整
        volatility = market_conditions[:, 1:2]  # 取波動性
        
        # 高波動性時偏向硬選擇
        volatility_factor = torch.sigmoid(-(volatility - self.volatility_threshold) * 10)
        
        # 最終混合分數（結合神經網絡和規則）
        final_score = mixing_score * 0.7 + volatility_factor * 0.3
        
        return final_score.squeeze(-1)


class DynamicMixingStrategy(nn.Module):
    """動態軟硬混合策略"""
    def __init__(self, n_agents=6, state_dim=64):
        super(DynamicMixingStrategy, self).__init__()
        self.n_agents = n_agents
        
        # 軟混合權重生成器
        self.soft_mixer = nn.Sequential(
            nn.Linear(state_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_agents),
            nn.Softmax(dim=-1)
        )
        
        # 硬選擇器
        self.hard_selector = nn.Sequential(
            nn.Linear(state_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_agents)
        )
        
        # 溫度參數（可學習）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, state_features, market_conditions, mixing_weight, training=True):
        """
        state_features: 狀態特徵
        market_conditions: 市場條件
        mixing_weight: 混合權重 [0,1]，0=硬選擇，1=軟混合
        training: 是否在訓練模式
        """
        combined_input = torch.cat([state_features, market_conditions], dim=-1)
        
        # 軟混合權重
        soft_weights = self.soft_mixer(combined_input)
        
        # 硬選擇
        hard_logits = self.hard_selector(combined_input)
        if training:
            # 訓練時使用Gumbel Softmax保持可微性
            hard_weights = F.gumbel_softmax(hard_logits, tau=self.temperature, hard=False)
        else:
            # 推理時使用真正的hard selection
            hard_weights = F.one_hot(torch.argmax(hard_logits, dim=-1), num_classes=self.n_agents).float()
        
        # 動態混合
        mixing_weight = mixing_weight.unsqueeze(-1)  # [batch_size, 1]
        final_weights = mixing_weight * soft_weights + (1 - mixing_weight) * hard_weights
        
        return final_weights, soft_weights, hard_weights


class hyperagent(nn.Module):
    def __init__(self, state_dim_1, state_dim_2, action_dim, hidden_dim):
        super(hyperagent, self).__init__()
        self.state_dim_1 = state_dim_1
        self.state_dim_2 = state_dim_2
        self.hidden_dim = hidden_dim
        
        # 狀態編碼器
        self.fc1 = nn.Linear(state_dim_1 + state_dim_2, hidden_dim)
        self.fc2 = nn.Linear(2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim * 2, elementwise_affine=False, eps=1e-6)
        self.embedding = nn.Embedding(action_dim, hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=True)
        )
        
        # 市場狀態分析器
        self.market_analyzer = MarketStateAnalyzer(input_dim=hidden_dim * 2 + 2)
        
        # 動態混合策略
        self.dynamic_mixer = DynamicMixingStrategy(n_agents=6, state_dim=hidden_dim * 2)
        
        # 傳統軟混合（向後兼容）
        self.traditional_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim * 4, 6),
            nn.Softmax(dim=1)
        )
        nn.init.zeros_(self.traditional_net[-2].weight)
        nn.init.zeros_(self.traditional_net[-2].bias)
        
        # 策略切換記錄
        self.register_buffer('switch_history', torch.zeros(100))  # 記錄最近100次的策略選擇
        self.register_buffer('switch_counter', torch.tensor(0))

    def encode(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                previous_action: torch.tensor):
        """編碼狀態特徵"""
        action_hidden = self.embedding(previous_action)
        state_hidden = self.fc1(torch.cat([single_state, trend_state], dim=1))
        x = torch.cat([action_hidden, state_hidden], dim=1)
        return x

    def forward(self, 
                single_state: torch.tensor,
                trend_state: torch.tensor,
                class_state: torch.tensor,
                previous_action: torch.tensor,
                use_dynamic_mixing: bool = True):
        """
        use_dynamic_mixing: 是否使用動態混合策略
        """
        # 編碼狀態
        encoded_state = self.encode(single_state, trend_state, previous_action)
        
        if use_dynamic_mixing:
            # 動態混合策略
            c = self.fc2(class_state)
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            modulated_state = modulate(self.norm(encoded_state), shift, scale)
            
            # 市場狀態分析
            mixing_weight = self.market_analyzer(modulated_state, class_state)
            
            # 動態混合
            final_weights, soft_weights, hard_weights = self.dynamic_mixer(
                modulated_state, class_state, mixing_weight, self.training
            )
            
            # 記錄策略選擇歷史
            if self.training:
                avg_mixing_weight = mixing_weight.mean().item()
                self.switch_history[self.switch_counter % 100] = avg_mixing_weight
                self.switch_counter += 1
            
            return final_weights, mixing_weight, soft_weights, hard_weights
        else:
            # 傳統軟混合
            c = self.fc2(class_state)
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm(encoded_state), shift, scale)
            weight = self.traditional_net(x)
            
            return weight

    def get_strategy_stats(self):
        """獲取策略切換統計信息"""
        if self.switch_counter == 0:
            return {"avg_mixing_weight": 0.5, "strategy_switches": 0}
            
        history = self.switch_history[:min(self.switch_counter, 100)]
        avg_mixing_weight = history.mean().item()
        
        # 計算策略切換次數
        switches = 0
        for i in range(1, len(history)):
            if abs(history[i] - history[i-1]) > 0.3:  # 閾值可調
                switches += 1
                
        return {
            "avg_mixing_weight": avg_mixing_weight,
            "strategy_switches": switches,
            "soft_ratio": (history > 0.5).float().mean().item(),
            "hard_ratio": (history <= 0.5).float().mean().item()
        }


def calculate_q(w, qs):
    """計算加權Q值"""
    q_tensor = torch.stack(qs)
    q_tensor = q_tensor.permute(1, 0, 2)
    weights_reshaped = w.view(-1, 1, 6)
    combined_q = torch.bmm(weights_reshaped, q_tensor).squeeze(1)
    
    return combined_q