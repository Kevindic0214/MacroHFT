from cmath import sin
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import random

# Mock feature lists
tech_indicator_list = ['close', 'volume', 'high', 'low', 'rsi', 'macd', 'bb_upper', 'bb_lower']
tech_indicator_list_trend = ['trend_short', 'trend_medium', 'momentum']

def create_market_data(days: int = 1000, initial_price: float = 100.0) -> pd.DataFrame:
    """Create synthetic market data with technical indicators"""
    np.random.seed(42)
    
    # Generate price series with trends
    prices = [initial_price]
    for i in range(days):
        daily_return = np.random.normal(0.0005, 0.015)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1.0))
    
    prices = prices[1:]
    volumes = np.random.uniform(100000, 200000, days)
    
    df = pd.DataFrame({
        'close': prices,
        'volume': volumes,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
    })
    
    # Technical indicators
    df['rsi'] = 50 + 30 * np.sin(np.arange(days) * 0.1) + np.random.normal(0, 5, days)
    df['rsi'] = np.clip(df['rsi'], 0, 100)
    df['macd'] = np.random.normal(0, 0.5, days)
    df['bb_upper'] = df['close'] * 1.02
    df['bb_lower'] = df['close'] * 0.98
    
    # Trend indicators
    df['trend_short'] = df['close'].pct_change(5).fillna(0)
    df['trend_medium'] = df['close'].pct_change(20).fillna(0)
    df['momentum'] = df['close'].pct_change(10).fillna(0)
    
    return df

def make_q_table_reward(df: pd.DataFrame, num_action: int = 2, max_holding: float = 0.01,
                       reward_scale: float = 1, gamma: float = 0.99, 
                       commission_fee: float = 0.001, max_punish: float = 1e12) -> np.ndarray:
    """Simplified Q-table generation"""
    q_table = np.random.uniform(-0.1, 0.1, (len(df), num_action, num_action))
    return q_table

# Import the original environments
from low_level_env import Testing_Env, Training_Env

# Transformer Model for Trading
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TradingTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, seq_length=20):
        super(TradingTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_length = seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True  # Set to True for better performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.output_projection = nn.Linear(d_model, 64)
        self.action_head = nn.Linear(64, 2)  # Binary action: 0 or 1
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        batch_size = x.size(0)
        
        # Project input to d_model dimension
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding (batch_first=True)
        x = x.transpose(0, 1)  # (seq_length, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, d_model)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Use the last time step for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Apply output layers
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.relu(self.output_projection(x))
        x = self.dropout(x)
        
        # Action probabilities
        action_logits = self.action_head(x)
        
        return action_logits

class TransformerTradingAgent:
    def __init__(self, input_dim, seq_length=20, device='cpu'):
        self.device = device
        self.seq_length = seq_length
        self.model = TradingTransformer(input_dim, seq_length=seq_length).to(device)
        
    def predict(self, state_sequence):
        """Forward pass only - no training"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(state_sequence, np.ndarray):
                state_sequence = torch.FloatTensor(state_sequence)
            
            # Ensure correct shape: (1, seq_length, input_dim)
            if len(state_sequence.shape) == 2:
                state_sequence = state_sequence.unsqueeze(0)
            
            state_sequence = state_sequence.to(self.device)
            
            # Get action logits
            action_logits = self.model(state_sequence)
            
            # Convert to probabilities and sample action
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1).cpu().item()
            
            return action, action_probs.cpu().numpy()[0]

class TransformerStrategy:
    def __init__(self, input_dim, seq_length=20):
        self.name = "Transformer Strategy"
        self.seq_length = seq_length
        self.agent = TransformerTradingAgent(input_dim, seq_length)
        self.state_buffer = []
        
    def decide(self, observation: Tuple, info: Dict, step: int) -> int:
        single_state, trend_state = observation
        
        # Combine all states
        combined_state = np.concatenate([single_state, trend_state], axis=1)
        
        # Add to buffer
        self.state_buffer.append(combined_state[-1])  # Only keep the latest timestep
        
        # Keep only last seq_length states
        if len(self.state_buffer) > self.seq_length:
            self.state_buffer = self.state_buffer[-self.seq_length:]
        
        # If we don't have enough history, return random action
        if len(self.state_buffer) < self.seq_length:
            return np.random.choice([0, 1])
        
        # Create sequence tensor
        state_sequence = np.array(self.state_buffer)
        
        # Get action from transformer
        action, probs = self.agent.predict(state_sequence)
        
        return action

def run_strategy(env, strategy, max_steps: int = 300) -> Dict:
    """Run a complete trading session"""
    try:
        observation = env.reset()
        if len(observation) == 3:
            single_state, trend_state, info = observation
        else:
            single_state, trend_state = observation
            info = {}
    except Exception as e:
        print(f"Error during environment reset: {e}")
        return {
            'strategy': strategy.name,
            'results': {'actions': [], 'rewards': [], 'prices': [], 'positions': []},
            'performance': {'total_reward': 0, 'num_trades': 0, 'holding_ratio': 0, 'avg_reward': 0, 'sharpe_ratio': 0},
            'total_steps': 0
        }
    
    results = {
        'actions': [],
        'rewards': [],
        'prices': [],
        'positions': []
    }
    
    total_reward = 0
    
    for step in range(max_steps):
        try:
            action = strategy.decide((single_state, trend_state), info, step)
            single_state, trend_state, reward, done, info = env.step(action)
            
            results['actions'].append(action)
            results['rewards'].append(reward)
            results['prices'].append(env.data.iloc[-1]['close'])
            results['positions'].append(env.position)
            
            total_reward += reward
            
            if done:
                break
        except Exception as e:
            print(f"Error during step {step}: {e}")
            break
    
    # Calculate performance metrics
    if len(results['actions']) > 0:
        actions = np.array(results['actions'])
        rewards = np.array(results['rewards'])
        
        performance = {
            'total_reward': total_reward,
            'num_trades': np.sum(np.diff(actions) != 0) if len(actions) > 1 else 0,
            'holding_ratio': np.mean(actions) * 100,
            'avg_reward': np.mean(rewards),
            'sharpe_ratio': np.mean(rewards) / np.std(rewards) if np.std(rewards) > 0 else 0
        }
    else:
        performance = {
            'total_reward': 0,
            'num_trades': 0,
            'holding_ratio': 0,
            'avg_reward': 0,
            'sharpe_ratio': 0
        }
    
    return {
        'strategy': strategy.name,
        'results': results,
        'performance': performance,
        'total_steps': step + 1 if 'step' in locals() else 0
    }

def plot_results(all_results: List[Dict]):
    """Create visualization of trading results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, result in enumerate(all_results):
        color = f'C{i}'
        results = result['results']
        
        # Price and Position
        ax1 = axes[0, 0]
        ax1.plot(results['prices'], label=f"{result['strategy']} - Price", color=color, alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(results['positions'], '--', label=f"Position", alpha=0.8, color=color, linewidth=2)
        ax1.set_title('Price & Position Over Time')
        ax1.set_ylabel('Price')
        ax1_twin.set_ylabel('Position')
        ax1.legend(loc='upper left')
        
        # Cumulative Rewards
        ax2 = axes[0, 1]
        cumulative = np.cumsum(results['rewards'])
        ax2.plot(cumulative, label=result['strategy'], color=color, linewidth=2)
        ax2.set_title('Cumulative Rewards')
        ax2.set_ylabel('Cumulative Reward')
        ax2.legend()
        
        # Actions
        ax3 = axes[1, 0]
        ax3.plot(results['actions'], label=result['strategy'], alpha=0.7, color=color)
        ax3.set_title('Trading Actions (0=Sell, 1=Buy)')
        ax3.set_ylabel('Action')
        ax3.legend()
        
        # Performance Bar Chart
        ax4 = axes[1, 1]
        perf = result['performance']
        metrics = ['total_reward', 'num_trades', 'holding_ratio', 'sharpe_ratio']
        values = [perf[m] for m in metrics]
        x_pos = np.arange(len(metrics)) + i * 0.25
        ax4.bar(x_pos, values, width=0.25, label=result['strategy'], color=color, alpha=0.8)
        ax4.set_xticks(np.arange(len(metrics)) + 0.25)
        ax4.set_xticklabels(metrics, rotation=45)
        ax4.set_title('Performance Metrics')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('transformer_trading_results.png', dpi=150, bbox_inches='tight')
    print("Results saved to: transformer_trading_results.png")

def main():
    """Main testing function"""
    print("="*70)
    print("Transformer-based Trading System Test")
    print("="*70)
    
    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # 1. Create test data
    print("\n1. Creating market data...")
    df = create_market_data(days=1000)  # Increased from 800 to 1000
    print(f"   Data length: {len(df)} days")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Ensure we have enough data for 20-day lookback
    if len(df) < 50:
        print("   ERROR: Not enough data for 20-day lookback!")
        return
    
    # 2. Create environments with 20-day lookback
    print("\n2. Creating environments (20-day lookback)...")
    testing_env = Testing_Env(
        df=df,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        back_time_length=20  # Set to 20 days
    )
    
    training_env = Training_Env(
        df=df,
        tech_indicator_list=tech_indicator_list,
        tech_indicator_list_trend=tech_indicator_list_trend,
        back_time_length=20  # Set to 20 days
    )
    
    print(f"   Action space: {testing_env.action_space}")
    print(f"   Observation space: {testing_env.observation_space}")
    
    # 3. Initialize Transformer strategy
    print("\n3. Initializing Transformer model...")
    input_dim = len(tech_indicator_list) + len(tech_indicator_list_trend)
    print(f"   Input dimension: {input_dim}")
    print(f"   Sequence length: 20")
    
    # 4. Test strategies
    print("\n4. Testing strategies...")
    strategies = [
        TransformerStrategy(input_dim, seq_length=20),
    ]
    
    all_results = []
    
    # Test with environments
    for env_name, env in [("Testing", testing_env), ("Training", training_env)]:
        print(f"\n   {env_name} Environment:")
        for strategy in strategies:
            print(f"     Running {strategy.name}...")
            result = run_strategy(env, strategy, max_steps=400)
            result['env_type'] = env_name
            all_results.append(result)
            
            perf = result['performance']
            print(f"       Total reward: {perf['total_reward']:.4f}")
            print(f"       Trades: {perf['num_trades']}")
            print(f"       Holding ratio: {perf['holding_ratio']:.1f}%")
            print(f"       Sharpe ratio: {perf['sharpe_ratio']:.4f}")
    
    # 5. Results summary
    print("\n5. Results Summary:")
    print("-" * 80)
    print(f"{'Strategy':<18} {'Environment':<10} {'Total Reward':<12} {'Trades':<8} {'Hold%':<8} {'Sharpe':<8}")
    print("-" * 80)
    
    for result in all_results:
        perf = result['performance']
        print(f"{result['strategy']:<18} {result['env_type']:<10} "
              f"{perf['total_reward']:<12.4f} {perf['num_trades']:<8} "
              f"{perf['holding_ratio']:<8.1f} {perf['sharpe_ratio']:<8.3f}")
    
    # 6. Generate plots
    print("\n6. Generating visualizations...")
    plot_results(all_results[:3])  # Plot first 3 results (one of each strategy)
    
    # 7. Model architecture summary
    print("\n7. Transformer Model Architecture:")
    print(f"   - Input dimension: {input_dim}")
    print(f"   - Model dimension: 128")
    print(f"   - Number of heads: 8")
    print(f"   - Number of layers: 4")
    print(f"   - Sequence length: 20")
    print(f"   - Output: Binary action (0/1)")
    
    print("\n" + "="*70)
    print("Test completed!")
    print("Check 'transformer_trading_results.png' for detailed charts.")
    print("="*70)

if __name__ == "__main__":
    main()