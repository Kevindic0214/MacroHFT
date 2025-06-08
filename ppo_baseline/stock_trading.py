import pandas as pd
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import datetime
import os
import gym
from gym import spaces
import matplotlib.pyplot as plt

# 環境設計
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_amount=1000000, buy_cost_pct=0.001, sell_cost_pct=0.001, 
                 state_space=None, stock_dim=None, tech_indicator_list=None, reward_scaling=1e-4):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.state_space = state_space
        self.stock_dim = stock_dim
        self.tech_indicator_list = tech_indicator_list
        self.reward_scaling = reward_scaling
        self.max_steps = len(df[df['tic'] == df['tic'].iloc[0]]) - 1
        self.current_step = 0
        self.account_balance = initial_amount
        self.shares_held = np.zeros(stock_dim)
        self.net_worth = initial_amount
        self.history = {'total_asset': []}
        self.trade_log = []

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32)

        self.price_scaler = df.groupby('tic')['close'].max().values
        self.indicator_scaler = []
        for tic in df['tic'].unique():
            tic_data = df[df['tic'] == tic][tech_indicator_list]
            tic_std = tic_data.std().replace(0, 1).values
            self.indicator_scaler.extend(tic_std)
        self.indicator_scaler = np.array(self.indicator_scaler)

        self.state = self._initiate_state()

    def _initiate_state(self):
        current_data = self.df[self.df['date'] == self.df['date'].iloc[self.current_step]]
        prices = current_data[['close']].values.flatten()
        tech_indicators = current_data[self.tech_indicator_list].values.flatten()
        state = np.concatenate((
            [self.account_balance / self.initial_amount],
            prices / self.price_scaler,
            self.shares_held / 100,
            tech_indicators / self.indicator_scaler
        ))
        return state

    def step(self, actions):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            return self.state, 0, True, {}

        current_data = self.df[self.df['date'] == self.df['date'].iloc[self.current_step]]
        prices = current_data[['close']].values.flatten()
        tech_indicators = current_data[self.tech_indicator_list].values.flatten()

        for i in range(self.stock_dim):
            action = np.clip(actions[i], -1, 1)
            price = prices[i]
            date = current_data['date'].iloc[0]
            tic = current_data['tic'].iloc[i]
            if action > 0:
                shares_to_buy = min(self.account_balance // (price * (1 + self.buy_cost_pct)), action * 100)
                cost = shares_to_buy * price * (1 + self.buy_cost_pct)
                self.account_balance -= cost
                self.shares_held[i] += shares_to_buy
                self.trade_log.append({'date': date, 'tic': tic, 'action': 'buy', 'price': price, 'shares': shares_to_buy})
            elif action < 0:
                shares_to_sell = min(self.shares_held[i], -action * 100)
                revenue = shares_to_sell * price * (1 - self.sell_cost_pct)
                self.account_balance += revenue
                self.shares_held[i] -= shares_to_sell
                self.trade_log.append({'date': date, 'tic': tic, 'action': 'sell', 'price': price, 'shares': shares_to_sell})

        self.net_worth = self.account_balance + np.sum(self.shares_held * prices)
        self.history['total_asset'].append(self.net_worth)

        reward = (self.net_worth - self.initial_amount) * self.reward_scaling
        self.state = np.concatenate((
            [self.account_balance / self.initial_amount],
            prices / self.price_scaler,
            self.shares_held / 100,
            tech_indicators / self.indicator_scaler
        ))

        done = self.current_step >= self.max_steps or self.net_worth <= 0
        return self.state, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.account_balance = self.initial_amount
        self.shares_held = np.zeros(self.stock_dim)
        self.net_worth = self.initial_amount
        self.history = {'total_asset': []}
        self.trade_log = []
        return self._initiate_state()

    def render(self, mode='human'):
        return self.net_worth

# 建立資料夾
if not os.path.exists("./datasets"):
    os.makedirs("./datasets")

# 設定參數與下載 QQQ 資料
TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2020-12-31'
TRADE_START_DATE = '2021-01-01'
TRADE_END_DATE = '2023-12-31'

ticker = 'QQQ'
df = yf.download(ticker, start=TRAIN_START_DATE, end=TRADE_END_DATE)
df['tic'] = ticker
df = df.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'tic']]
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'tic']

# 技術指標
def add_technical_indicators(df):
    df = df.copy()
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mean_dev = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['cci_20'] = (typical_price - sma_tp) / (0.015 * mean_dev)
    return df

df = add_technical_indicators(df)
df = df.fillna(0)

df.to_csv('./datasets/qqq_data.csv', index=False)

# 建立環境
indicators_list = ['macd', 'rsi_14', 'cci_20']
stock_dimension = 1
state_space = 1 + 2 * stock_dimension + len(indicators_list) * stock_dimension
env_kwargs = {
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": indicators_list,
    "reward_scaling": 1e-4
}

train_df = df[(df.date >= TRAIN_START_DATE) & (df.date <= TRAIN_END_DATE)]
trade_df = df[(df.date >= TRADE_START_DATE) & (df.date <= TRADE_END_DATE)]

# 訓練模型
env = StockTradingEnv(df=train_df, **env_kwargs)
env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.01)
model.learn(total_timesteps=100000)
model.save("./ppo_qqq_model")

# 回測
env_trade = StockTradingEnv(df=trade_df, **env_kwargs)
obs = env_trade.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env_trade.step(action)

df_account_value = pd.DataFrame(env_trade.history['total_asset'], columns=['account_value'])
df_account_value.to_csv('./datasets/account_value.csv', index=False)
df_trade_records = pd.DataFrame(env_trade.trade_log)
df_trade_records.to_csv('./datasets/trade_records.csv', index=False)

# 視覺化交易行為
df_main = trade_df[['date', 'close']].reset_index(drop=True)
buy_points = df_trade_records[df_trade_records['action'] == 'buy']
sell_points = df_trade_records[df_trade_records['action'] == 'sell']

plt.figure(figsize=(15, 8))
plt.plot(df_main['date'], df_main['close'], label='QQQ Close Price', color='black')
plt.scatter(buy_points['date'], buy_points['price'], marker='^', color='green', label='Buy', s=100)
plt.scatter(sell_points['date'], sell_points['price'], marker='v', color='red', label='Sell', s=100)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title("Trading Actions on QQQ")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./datasets/trade_visualization.png')
plt.show()

print("QQQ 強化學習交易回測完成，結果已保存至 ./datasets/")
