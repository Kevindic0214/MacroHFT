import numpy as np
import pandas as pd


def make_q_table_reward(df: pd.DataFrame,
                        num_action,
                        max_holding,
                        reward_scale=1000, # Note: reward_scale in env is 1, here is 1000. This might need alignment.
                        gamma=0.999,
                        commission_fee=0.001,
                        max_punish=1e12,
                        # C51 parameters for teacher distribution
                        v_min=-5.0, 
                        v_max=5.0, 
                        num_atoms=51):
    
    delta_z = (v_max - v_min) / (num_atoms - 1)
    support = np.linspace(v_min, v_max, num_atoms)

    # q_table now stores distributions: (len(df), num_action_prev, num_action_curr, num_atoms)
    q_table = np.zeros((len(df), num_action, num_action, num_atoms))

    def calculate_value(price_information, position):
        return price_information["close"] * position

    scale_factor = num_action - 1

    for t in range(2, len(df) + 1):
        current_price_information = df.iloc[-t]
        future_price_information = df.iloc[-t + 1]
        for previous_action in range(num_action):
            for current_action in range(num_action):
                # Calculate expected Q value first (as before)
                if current_action > previous_action: # Buy
                    previous_position = previous_action / (scale_factor) * max_holding
                    current_position = current_action / (scale_factor) * max_holding
                    position_change = (current_action - previous_action) / scale_factor * max_holding
                    buy_money = position_change * current_price_information['close'] * (1 + commission_fee)
                    current_asset_value = calculate_value(current_price_information, previous_position)
                    # Expected value of taking current_action and then acting optimally from next state
                    # The Q-value from next state onwards is the expectation of the distribution
                    next_q_values_dist = q_table[len(df) - t + 1][current_action][:] # (num_action_next, num_atoms)
                    expected_next_q_max = np.max(np.sum(next_q_values_dist * support, axis=1)) # Max over next actions
                    
                    reward = calculate_value(future_price_information, current_position) - (current_asset_value + buy_money)
                    reward = reward_scale * reward
                    expected_q_target = reward + gamma * expected_next_q_max

                else: # Sell or Hold
                    previous_position = previous_action / (scale_factor) * max_holding
                    current_position = current_action / (scale_factor) * max_holding
                    position_change = (previous_action - current_action) / scale_factor * max_holding
                    sell_money = position_change * current_price_information['close'] * (1 - commission_fee)
                    current_asset_value = calculate_value(current_price_information, previous_position)
                    # Expected value of taking current_action and then acting optimally from next state
                    next_q_values_dist = q_table[len(df) - t + 1][current_action][:] # (num_action_next, num_atoms)
                    expected_next_q_max = np.max(np.sum(next_q_values_dist * support, axis=1)) # Max over next actions

                    reward = calculate_value(future_price_information, current_position) + sell_money - current_asset_value
                    reward = reward_scale * reward
                    expected_q_target = reward + gamma * expected_next_q_max
                
                # Convert expected_q_target to a one-hot distribution on the support
                clamped_q = np.clip(expected_q_target, v_min, v_max)
                atom_index = np.round((clamped_q - v_min) / delta_z).astype(int)
                # Ensure index is within bounds, can happen due to float precision
                atom_index = np.clip(atom_index, 0, num_atoms - 1) 
                
                q_dist = np.zeros(num_atoms)
                q_dist[atom_index] = 1.0
                q_table[len(df) - t][previous_action][current_action] = q_dist

    return q_table