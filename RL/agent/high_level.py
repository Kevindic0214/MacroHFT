import argparse
import os
import pathlib
import random
import sys
import warnings
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

ROOT = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(ROOT)
sys.path.insert(0, ".")

from env.high_level_env import Testing_Env, Training_Env
from model.net import hyperagent, subagent
from RL.util.memory import episodicmemory
from RL.util.replay_buffer import ReplayBuffer_High
from RL.util.utili import LinearDecaySchedule

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--buffer_size",type=int,default=1000000,)
parser.add_argument("--dataset",type=str,default="ETHUSDT")
parser.add_argument("--q_value_memorize_freq",type=int, default=10,)
parser.add_argument("--batch_size",type=int,default=512)
parser.add_argument("--eval_update_freq",type=int,default=512)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start",type=float,default=0.7)
parser.add_argument("--epsilon_end",type=float,default=0.3)
parser.add_argument("--decay_length",type=int,default=5)
parser.add_argument("--update_times",type=int,default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost",type=float,default=0.2 / 1000)
parser.add_argument("--back_time_length",type=int,default=1)
parser.add_argument("--seed",type=int,default=12345)
parser.add_argument("--n_step",type=int,default=1)
parser.add_argument("--epoch_number",type=int,default=5)
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--alpha",type=float,default=0.5)
parser.add_argument("--beta",type=float,default=5.0)
parser.add_argument("--exp",type=str,default="exp1")
parser.add_argument("--num_step",type=int,default=10)
parser.add_argument("--num_quantiles", type=int, default=51)
parser.add_argument("--hyperagent_hidden_dim", type=int, default=32)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        self.result_path = os.path.join("./result/high_level", '{}'.format(args.dataset), args.exp)
        self.model_path = os.path.join(self.result_path,
                                       "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.val_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.test_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "whole")
        self.dataset=args.dataset
        self.num_step = args.num_step
        if "BTC" in self.dataset:
            self.max_holding_number=0.01
        elif "ETH" in self.dataset:
            self.max_holding_number=0.2
        elif "DOT" in self.dataset:
            self.max_holding_number=10
        elif "LTC" in self.dataset:
            self.max_holding_number=10
        else:
            raise Exception ("we do not support other dataset yet")
        self.epoch_number = args.epoch_number
        
        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load('./data/feature_list/single_features.npy', allow_pickle=True).tolist()
        self.tech_indicator_list_trend = np.load('./data/feature_list/trend_features.npy', allow_pickle=True).tolist()
        self.clf_list = ['slope_360', 'vol_360']

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.slope_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, args.num_quantiles).to(self.device)
        self.slope_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, args.num_quantiles).to(self.device)
        self.slope_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, args.num_quantiles).to(self.device)
        self.vol_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, args.num_quantiles).to(self.device)
        self.vol_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, args.num_quantiles).to(self.device)
        self.vol_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, args.num_quantiles).to(self.device)        
        model_list_slope = [
            "./result/low_level/ETHUSDT/slope/0/label_3/seed_12345/best_model.pkl", 
            "./result/low_level/ETHUSDT/slope/1/label_1/seed_12345/best_model.pkl",
            "./result/low_level/ETHUSDT/slope/4/label_2/seed_12345/best_model.pkl"
        ]
        model_list_vol = [
            "./result/low_level/ETHUSDT/vol/1/label_2/seed_12345/best_model.pkl",
            "./result/low_level/ETHUSDT/vol/1/label_3/seed_12345/best_model.pkl",
            "./result/low_level/ETHUSDT/vol/4/label_1/seed_12345/best_model.pkl"
        ]
        self.slope_1.load_state_dict(
            torch.load(model_list_slope[0], map_location=self.device))
        self.slope_2.load_state_dict(
            torch.load(model_list_slope[1], map_location=self.device))
        self.slope_3.load_state_dict(
            torch.load(model_list_slope[2], map_location=self.device))
        self.vol_1.load_state_dict(
            torch.load(model_list_vol[0], map_location=self.device))
        self.vol_2.load_state_dict(
            torch.load(model_list_vol[1], map_location=self.device))
        self.vol_3.load_state_dict(
            torch.load(model_list_vol[2], map_location=self.device))
        self.slope_1.eval()
        self.slope_2.eval()
        self.slope_3.eval()
        self.vol_1.eval()
        self.vol_2.eval()
        self.vol_3.eval()
        self.slope_agents = {
            0: self.slope_1,
            1: self.slope_2,
            2: self.slope_3
        }
        self.vol_agents = {
            0: self.vol_1,
            1: self.vol_2,
            2: self.vol_3
        }
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, args.hyperagent_hidden_dim).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, args.hyperagent_hidden_dim).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(),
                                          lr=args.lr)
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.n_step = args.n_step
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.decay_length = args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end, decay_length=self.decay_length)
        self.epsilon = args.epsilon_start
        self.memory = episodicmemory(4320, 5, self.n_state_1, self.n_state_2, 64, self.device)
        self.args = args

        # QR-DQN specific parameters for high-level agent
        self.num_quantiles = self.args.num_quantiles # Assumes num_quantiles is in args
        self.kappa = getattr(self.args, 'kappa', 1.0) # Use args.kappa if available, else default to 1.0
        
        # Precompute tau values for quantile midpoints
        self.tau_values = (
            torch.arange(self.num_quantiles, device=self.device, dtype=torch.float32) + 0.5
        ) / self.num_quantiles
        
        # Ensure alpha and beta for loss components are attributes of the class
        self.alpha = args.alpha # for memory_error weight
        self.beta = args.beta # for KL_loss weight

    def calculate_q_distribution(self, w, qs_distributions):
        # qs_distributions is a list of tensors, each shape (batch_size, action_dim, num_quantiles)
        # w (weights from hyperagent) has shape (batch_size, 6)
        
        stacked_qs_distributions = torch.stack(qs_distributions) # Shape: (6, batch_size, action_dim, num_quantiles)
        # Permute to (batch_size, 6, action_dim, num_quantiles)
        stacked_qs_distributions = stacked_qs_distributions.permute(1, 0, 2, 3) 
        
        # Reshape weights for broadcasting: (batch_size, 6) -> (batch_size, 6, 1, 1)
        weights_reshaped = w.view(-1, stacked_qs_distributions.size(1), 1, 1) 
        
        # Weighted sum of distributions
        # Output shape: (batch_size, action_dim, num_quantiles)
        combined_q_dist = torch.sum(weights_reshaped * stacked_qs_distributions, dim=1)
        return combined_q_dist

    def update(self, replay_buffer):
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        
        with torch.no_grad():
            w_next_target = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
            w_next_eval = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])

        qs_current_distributions = [
                    self.slope_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.slope_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.slope_agents[2](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[0](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[1](batch['state'], batch['state_trend'], batch['previous_action']),
                    self.vol_agents[2](batch['state'], batch['state_trend'], batch['previous_action'])
        ]
        qs_next_distributions = [
                    self.slope_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.slope_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.slope_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[0](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[1](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']),
                    self.vol_agents[2](batch['next_state'], batch['next_state_trend'], batch['next_previous_action'])
        ]
        
        # Get current Q-distribution for all actions
        q_dist_current_all_actions = self.calculate_q_distribution(w_current, qs_current_distributions)
        
        # Gather the Q-distribution for the actions taken
        action_batch = batch['action'].long() # Shape: (batch_size, 1)
        # Expand action_batch to gather across num_quantiles: (batch_size, 1, num_quantiles)
        action_batch_expanded = action_batch.unsqueeze(-1).expand(-1, -1, self.num_quantiles)
        current_z = q_dist_current_all_actions.gather(1, action_batch_expanded).squeeze(1) # Shape: (batch_size, num_quantiles)

        with torch.no_grad():
            # Select next actions using eval_net (mean of quantiles from combined distribution)
            q_dist_next_all_actions_eval = self.calculate_q_distribution(w_next_eval, qs_next_distributions)
            q_values_next_for_action_selection = q_dist_next_all_actions_eval.mean(dim=2) # Shape: (batch_size, action_dim)
            a_argmax = q_values_next_for_action_selection.argmax(dim=-1, keepdim=True) # Shape: (batch_size, 1)

            # Get next state quantiles from target_net's combined distribution for the selected actions
            q_dist_next_all_actions_target = self.calculate_q_distribution(w_next_target, qs_next_distributions)
            # Expand a_argmax to gather across num_quantiles: (batch_size, 1, num_quantiles)
            a_argmax_expanded = a_argmax.unsqueeze(-1).expand(-1, -1, self.num_quantiles)
            next_best_quantiles = q_dist_next_all_actions_target.gather(1, a_argmax_expanded).squeeze(1) # Shape: (batch_size, num_quantiles)

            # Compute target quantile values (Bellman update for quantiles)
            target_z = (
                batch["reward"].unsqueeze(-1)
                + self.gamma
                * (1 - batch["terminal"].unsqueeze(-1))
                * next_best_quantiles
            ) # Shape: (batch_size, num_quantiles)

        # Compute Quantile Huber Loss for TD error
        delta_ij = target_z.unsqueeze(1) - current_z.unsqueeze(2) 
        abs_delta_ij = torch.abs(delta_ij)
        huber_loss_values = torch.where(
            abs_delta_ij <= self.kappa,
            0.5 * delta_ij.pow(2),
            self.kappa * (abs_delta_ij - 0.5 * self.kappa),
        )
        
        # self.tau_values is tau_i (for current quantiles from current_z)
        # Shape: (num_quantiles,) -> Unsqueeze to (1, num_quantiles, 1) for broadcasting against delta_ij
        tau_i_expanded = self.tau_values.unsqueeze(0).unsqueeze(-1)

        # Pairwise quantile loss: |tau_i - I(delta_ij < 0)| * huber_loss
        quantile_huber_loss_pairwise = (
            torch.abs(tau_i_expanded - (delta_ij < 0).float()) * huber_loss_values
        )
        # Average over target quantiles (dim 2), sum over current quantiles (dim 1), mean over batch
        td_error = quantile_huber_loss_pairwise.mean(dim=2).sum(dim=1).mean()

        # Clamp the td_error to prevent numerical instability
        td_error = torch.clamp(td_error, max=100.0)

        # Memory error calculation (comparing mean of current_z with q_memory)
        q_current_mean_for_memory = current_z.mean(dim=1) # Mean of current quantiles
        q_memory_from_batch = batch['q_memory']
        
        q_memory_valid_mask = ~torch.isnan(q_memory_from_batch)
        
        if q_memory_valid_mask.any():
            valid_q_current_mean = q_current_mean_for_memory[q_memory_valid_mask]
            valid_q_memory = q_memory_from_batch[q_memory_valid_mask]
            memory_error_val = F.mse_loss(valid_q_current_mean, valid_q_memory)
        else:
            memory_error_val = torch.tensor(0.0, device=self.device)

        demonstration = batch['demo_action']
        # KL_loss calculation uses mean of Q-distribution for softmax
        q_values_for_kl = q_dist_current_all_actions.mean(dim=2) # Shape: (batch_size, action_dim)
        KL_loss = F.kl_div(
            (q_values_for_kl.softmax(dim=-1) + 1e-8).log(),
            (demonstration.softmax(dim=-1) + 1e-8), 
            reduction="batchmean",
        )

        loss = td_error + self.alpha * memory_error_val + self.beta * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1)
        self.optimizer.step()
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_counter += 1
        return td_error.cpu(), memory_error_val.cpu(), KL_loss.cpu(), q_current_mean_for_memory.mean().cpu(), target_z.mean().cpu() # Return mean of target_z quantiles

    def act(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        if np.random.uniform() < (1-self.epsilon):
            with torch.no_grad(): # Added no_grad for inference
                qs_distributions = [
                        self.slope_agents[0](x1, x2, previous_action),
                        self.slope_agents[1](x1, x2, previous_action),
                        self.slope_agents[2](x1, x2, previous_action),
                        self.vol_agents[0](x1, x2, previous_action),
                        self.vol_agents[1](x1, x2, previous_action),
                        self.vol_agents[2](x1, x2, previous_action)
                ]
                w = self.hyperagent(x1, x2, x3, previous_action)
                
                # Get combined Q-distribution
                combined_q_dist = self.calculate_q_distribution(w, qs_distributions)
                # Take mean over quantiles to get Q-values for action selection
                actions_value = combined_q_dist.mean(dim=2) # Shape: (batch_size, action_dim)
                
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()
                action = action[0]
        else:
            action_choice = [0,1]
            action = random.choice(action_choice)
        return action

    def act_test(self, state, state_trend, state_clf, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor(info["previous_action"]).long().to(self.device),
                0).to(self.device)
            qs_distributions = [
                    self.slope_agents[0](x1, x2, previous_action),
                    self.slope_agents[1](x1, x2, previous_action),
                    self.slope_agents[2](x1, x2, previous_action),
                    self.vol_agents[0](x1, x2, previous_action),
                    self.vol_agents[1](x1, x2, previous_action),
                    self.vol_agents[2](x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)

            # Get combined Q-distribution
            combined_q_dist = self.calculate_q_distribution(w, qs_distributions)
            # Take mean over quantiles to get Q-values for action selection
            actions_value = combined_q_dist.mean(dim=2) # Shape: (batch_size, action_dim)

            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action

    def q_estimate(self, state, state_trend, state_clf, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).to(self.device)
            x2 = torch.FloatTensor(state_trend).to(self.device)
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.unsqueeze(
                torch.tensor(info["previous_action"]).long().to(self.device),
                0).to(self.device)
            qs_distributions = [
                self.slope_agents[0](x1, x2, previous_action),
                self.slope_agents[1](x1, x2, previous_action),
                self.slope_agents[2](x1, x2, previous_action),
                self.vol_agents[0](x1, x2, previous_action),
                self.vol_agents[1](x1, x2, previous_action),
                self.vol_agents[2](x1, x2, previous_action)
            ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            combined_q_dist = self.calculate_q_distribution(w, qs_distributions)
            # For q_estimate, typically we want the Q-value of the chosen action, or all Q-values.
            # Returning the mean Q-values for all actions.
            actions_value = combined_q_dist.mean(dim=2) 
            q = torch.max(actions_value, 1)[0].detach().cpu().numpy() # Q-value of the best action
            # If you need all Q-values:
            # q_all_actions = actions_value.squeeze(0).detach().cpu().numpy() 
            return q

    def calculate_hidden(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0).to(self.device)
        with torch.no_grad():
            hs = self.hyperagent.encode(x1, x2, previous_action).cpu().numpy()
        return hs


    def train(self):
        print("Starting training...")
        total_training_start_time = time.time()

        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        step_counter = 0
        episode_counter = 0
        self.epoch_counter = 0
        best_return_rate = -float('inf')
        best_model = None
        self.replay_buffer = ReplayBuffer_High(self.args, self.n_state_1, self.n_state_2, self.n_action)
        for sample in range(self.epoch_number):
            self.epoch_start_time = time.time()
            print('epoch ', self.epoch_counter + 1)
            self.df = pd.read_feather(
                os.path.join(self.train_data_path, "train.feather"))
            
            
            train_env = Training_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    clf_list=self.clf_list,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=random.choices(range(self.n_action), k=1)[0],
                    alpha = 0)
            s, s2, s3, info = train_env.reset()
            episode_reward_sum = 0
            
            while True:
                a = self.act(s, s2, s3, info)
                s_, s2_, s3_, r, done, info_ = train_env.step(a)
                hs = self.calculate_hidden(s, s2, info)
                q = r + self.gamma * (1 - done) * self.q_estimate(s_, s2_, s3_, info_)
                q_memory = self.memory.query(hs, a)
                if np.isnan(q_memory):
                    q_memory = q
                self.replay_buffer.store_transition(s, s2, s3, info['previous_action'], info['q_value'], a, r, s_, s2_, s3_, info_['previous_action'],
                                info_['q_value'], done, q_memory)
                self.memory.add(hs, a, q, s, s2, info['previous_action'])
                episode_reward_sum += r

                s, s2, s3, info = s_, s2_, s3_, info_
                step_counter += 1
                if step_counter % self.eval_update_freq == 0 and step_counter > (
                        self.batch_size + self.n_step):
                    for i in range(self.update_times):
                        td_error, memory_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                        if self.update_counter % self.q_value_memorize_freq == 1:
                            self.writer.add_scalar(
                                tag="td_error",
                                scalar_value=td_error,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="memory_error",
                                scalar_value=memory_error,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="KL_loss",
                                scalar_value=KL_loss,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="q_eval",
                                scalar_value=q_eval,
                                global_step=self.update_counter,
                                walltime=None)
                            self.writer.add_scalar(
                                tag="q_target",
                                scalar_value=q_target,
                                global_step=self.update_counter,
                                walltime=None)
                    if step_counter > 4320:
                        self.memory.re_encode(self.hyperagent)
                if done:
                    break
            episode_counter += 1
            final_balance, required_money = train_env.final_balance, train_env.required_money
            self.writer.add_scalar(tag="return_rate_train",
                                scalar_value=final_balance / (required_money),
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="final_balance_train",
                                scalar_value=final_balance,
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="required_money_train",
                                scalar_value=required_money,
                                global_step=episode_counter,
                                walltime=None)
            self.writer.add_scalar(tag="reward_sum_train",
                                scalar_value=episode_reward_sum,
                                global_step=episode_counter,
                                walltime=None)
            epoch_return_rate_train_list.append(final_balance / (required_money))
            epoch_final_balance_train_list.append(final_balance)
            epoch_required_money_train_list.append(required_money)
            epoch_reward_sum_train_list.append(episode_reward_sum)
                

            self.epoch_counter += 1
            self.epsilon = self.epsilon_scheduler.get_epsilon(self.epoch_counter)
            mean_return_rate_train = np.mean(epoch_return_rate_train_list)
            mean_final_balance_train = np.mean(epoch_final_balance_train_list)
            mean_required_money_train = np.mean(epoch_required_money_train_list)
            mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
            self.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=mean_return_rate_train,
                    global_step=self.epoch_counter,
                    walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_final_balance_train",
                scalar_value=mean_final_balance_train,
                global_step=self.epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_required_money_train",
                scalar_value=mean_required_money_train,
                global_step=self.epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_reward_sum_train",
                scalar_value=mean_reward_sum_train,
                global_step=self.epoch_counter,
                walltime=None,
                )
            epoch_path = os.path.join(self.model_path,
                                        "epoch_{}".format(self.epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            torch.save(self.hyperagent.state_dict(),
                        os.path.join(epoch_path, "trained_model.pkl"))  
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                    os.makedirs(val_path)
            return_rate_eval = self.val_cluster(epoch_path, val_path)
            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model = self.hyperagent.state_dict()
            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []
        best_model_path = os.path.join("./result/high_level", 
                                        '{}'.format(self.dataset), 'best_model.pkl')
        torch.save(best_model, best_model_path)
        final_result_path = os.path.join("./result/high_level", '{}'.format(self.dataset))
        self.test_cluster(best_model_path, final_result_path)

        total_training_end_time = time.time()
        total_training_duration = total_training_end_time - total_training_start_time
        print(f"Total training finished. Total duration: {total_training_duration:.2f} seconds")
        self.writer.add_scalar(
            tag="total_training_duration_seconds",
            scalar_value=total_training_duration,
            global_step=self.epoch_counter
        )

    def val_cluster(self, model_path, save_path):
        self.hyperagent.load_state_dict(
            torch.load(os.path.join(model_path, "trained_model.pkl")))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(
            os.path.join(self.val_data_path, "val.feather"))
        
        val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = val_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = val_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
            slient=True)
        final_balance = val_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)
        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action_val.npy"), action_list)
        np.save(os.path.join(save_path, "reward_val.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance_val.npy"),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money_val.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val.npy"),
                commission_fee_list)
        return_rate = final_balance / required_money

        epoch_duration = time.time() - self.epoch_start_time
        print(f"Epoch {self.epoch_counter} finished. Duration: {epoch_duration:.2f} seconds")
        self.writer.add_scalar(
            tag="epoch_duration_seconds",
            scalar_value=epoch_duration,
            global_step=self.epoch_counter
        )

        return return_rate

    def test_cluster(self, epoch_path, save_path):
        self.hyperagent.load_state_dict(
            torch.load(epoch_path))
        self.hyperagent.eval()
        counter = False
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        self.df = pd.read_feather(
            os.path.join(self.test_data_path, "test.feather"))
        
        test_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                clf_list=self.clf_list,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=0)
        s, s2, s3, info = test_env.reset()
        done = False
        action_list_episode = []
        reward_list_episode = []
        while not done:
            a = self.act_test(s, s2, s3, info)
            s_, s2_, s3_, r, done, info_ = test_env.step(a)
            reward_list_episode.append(r)
            s, s2, s3, info = s_, s2_, s3_, info_
            action_list_episode.append(a)
        portfit_magine, final_balance, required_money, commission_fee = test_env.get_final_return_rate(
            slient=True)
        final_balance = test_env.final_balance
        action_list.append(action_list_episode)
        reward_list.append(reward_list_episode)
        final_balance_list.append(final_balance)
        required_money_list.append(required_money)
        commission_fee_list.append(commission_fee)

        action_list = np.array(action_list)
        reward_list = np.array(reward_list)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(os.path.join(save_path, "action.npy"), action_list)
        np.save(os.path.join(save_path, "reward.npy"), reward_list)
        np.save(os.path.join(save_path, "final_balance.npy"),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money.npy"),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history.npy"),
                commission_fee_list)
            
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
