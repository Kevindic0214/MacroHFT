import argparse
import os
import pathlib
import random
import sys
import warnings

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
parser.add_argument("--epoch_number",type=int,default=15)
parser.add_argument("--device",type=str,default="cuda:0")
parser.add_argument("--alpha",type=float,default=0.5)
parser.add_argument("--beta",type=int,default=5)
parser.add_argument("--exp",type=str,default="exp1")
parser.add_argument("--num_step",type=int,default=10)


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
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.slope_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_1 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_2 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)
        self.vol_3 = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64).to(self.device)        
        model_list_slope = [
            "./result/low_level/ETHUSDT/slope.C51/0.0/label_3/seed_12345/best_model.pkl", 
            "./result/low_level/ETHUSDT/slope.C51/1.0/label_1/seed_12345/best_model.pkl",
            "./result/low_level/ETHUSDT/slope.C51/4.0/label_2/seed_12345/best_model.pkl"
        ]
        model_list_vol = [
            "./result/low_level/ETHUSDT/vol.C51/1.0/label_2/seed_12345/best_model.pkl",
            "./result/low_level/ETHUSDT/vol.C51/1.0/label_3/seed_12345/best_model.pkl",
            "./result/low_level/ETHUSDT/vol.C51/4.0/label_1/seed_12345/best_model.pkl"
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
        self.hyperagent = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target = hyperagent(self.n_state_1, self.n_state_2, self.n_action, 32).to(self.device)
        self.hyperagent_target.load_state_dict(self.hyperagent.state_dict())
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.hyperagent.parameters(),
                                          lr=args.lr)
        self.loss_func = nn.MSELoss()
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

        # For Distributional RL
        # Assuming all subagents share the same atom configuration from their init
        # (e.g., num_atoms=51, v_min=-5.0, v_max=5.0 by default in subagent class)
        self.num_atoms = self.slope_agents[0].num_atoms
        self.support = self.slope_agents[0].support.to(self.device)  # Shape: (num_atoms,)
        self.v_min = self.support[0].item()
        self.v_max = self.support[-1].item()
        if self.num_atoms > 1:
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        else: # Avoid division by zero if num_atoms is 1 (should not happen for distributional)
            self.delta_z = 1.0

    def calculate_q(self, w, qs):
        # qs is a list of 6 tensors, each with shape (batch_size, action_dim, num_atoms)
        # w is a tensor with shape (batch_size, 6)
        
        # Stack qs to shape (6, batch_size, action_dim, num_atoms)
        q_tensor = torch.stack(qs, dim=0)
        
        # Permute to (batch_size, 6, action_dim, num_atoms)
        q_tensor = q_tensor.permute(1, 0, 2, 3)
        
        # Reshape w for broadcasting: (batch_size, 6) -> (batch_size, 6, 1, 1)
        # Use w.shape[0] for current batch_size
        current_batch_size = w.shape[0]
        w_reshaped = w.view(current_batch_size, 6, 1, 1)
        
        # Weighted sum: (B, 6, 1, 1) * (B, 6, A, N) -> (B, 6, A, N)
        # Sum over dim=1 (the '6' subagents dimension) -> (B, A, N)
        combined_q_dist = (w_reshaped * q_tensor).sum(dim=1)
        
        return combined_q_dist # Shape (batch_size, action_dim, num_atoms)

    def update(self, replay_buffer):
        batch, tree_indices, IS_weights = replay_buffer.sample()
        if batch is None:
            # Return dummy tensors with appropriate types if needed by caller
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        current_batch_size = batch['reward'].shape[0] # Actual batch size from sample
        IS_weights_tensor = torch.tensor(IS_weights, device=self.device, dtype=torch.float32).squeeze() # (batch_size,)

        # Log PER-related metrics
        if hasattr(replay_buffer, 'PER_b') and hasattr(replay_buffer, 'max_priority'): # Check if PER is active
            self.writer.add_scalar('PER/beta', replay_buffer.PER_b, global_step=self.update_counter)
            self.writer.add_scalar('PER/avg_IS_weight', IS_weights_tensor.mean().item(), global_step=self.update_counter)
            self.writer.add_scalar('PER/max_IS_weight', IS_weights_tensor.max().item(), global_step=self.update_counter)
            self.writer.add_scalar('PER/buffer_max_priority', replay_buffer.max_priority, global_step=self.update_counter)

        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get Q-distributions for current states and actions
        w_current = self.hyperagent(batch['state'], batch['state_trend'], batch['state_clf'], batch['previous_action'])
        qs_current_list = [
            self.slope_agents[i](batch['state'], batch['state_trend'], batch['previous_action']) for i in range(3)
        ] + [
            self.vol_agents[i](batch['state'], batch['state_trend'], batch['previous_action']) for i in range(3)
        ]
        # q_distribution_all_actions: (batch_size, action_dim, num_atoms)
        q_distribution_all_actions = self.calculate_q(w_current, qs_current_list)
        
        # Gather Q-distribution for the action taken: (batch_size, num_atoms)
        # batch['action'] shape is (batch_size, 1)
        current_action_q_logits = q_distribution_all_actions.gather(
            1, batch['action'].unsqueeze(-1).expand(-1, -1, self.num_atoms)
        ).squeeze(1)

        # Get Q-distributions for next states (online net for action selection - Double DQN)
        w_next_online = self.hyperagent(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
        qs_next_online_list = [
            self.slope_agents[i](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']) for i in range(3)
        ] + [
            self.vol_agents[i](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']) for i in range(3)
        ]
        # q_next_online_dist_all_actions: (batch_size, action_dim, num_atoms)
        q_next_online_dist_all_actions = self.calculate_q(w_next_online, qs_next_online_list)
        
        # Calculate expected Q-values for action selection: (batch_size, action_dim)
        expected_q_next_online = (q_next_online_dist_all_actions * self.support.view(1, 1, -1)).sum(dim=2)
        # Select best action using online network: (batch_size, 1)
        a_argmax_next = expected_q_next_online.argmax(dim=1, keepdim=True)

        # Get Q-distributions for next states (target net for Q-value evaluation - Double DQN)
        with torch.no_grad():
            w_next_target = self.hyperagent_target(batch['next_state'], batch['next_state_trend'], batch['next_state_clf'], batch['next_previous_action'])
            qs_next_target_list = [
                self.slope_agents[i](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']) for i in range(3)
            ] + [
                self.vol_agents[i](batch['next_state'], batch['next_state_trend'], batch['next_previous_action']) for i in range(3)
            ]
            # q_next_target_dist_all_actions: (batch_size, action_dim, num_atoms)
            q_next_target_dist_all_actions = self.calculate_q(w_next_target, qs_next_target_list)

            # Gather Q-distribution for the best action a_argmax_next from target network: (batch_size, num_atoms)
            next_action_target_q_logits = q_next_target_dist_all_actions.gather(
                1, a_argmax_next.unsqueeze(-1).expand(-1, -1, self.num_atoms)
            ).squeeze(1) # These are logits, not probabilities yet.
            
            # Compute target distribution (projection)
            # rewards: (batch_size,), terminals: (batch_size,)
            # support: (num_atoms,)
            # next_action_target_q_probs: (batch_size, num_atoms)
            next_action_target_q_probs = F.softmax(next_action_target_q_logits, dim=1)

            rewards_b = batch['reward']
            terminals_b = batch['terminal']
            
            # tz = R + gamma * z', clamped. Shape: (batch_size, num_atoms)
            tz = rewards_b.unsqueeze(1) + self.gamma * self.support.unsqueeze(0) * (1 - terminals_b.unsqueeze(1))
            tz = torch.clamp(tz, self.v_min, self.v_max)

            # b_j = (tz_j - v_min) / delta_z. Shape: (batch_size, num_atoms)
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Handle cases where l = u for atoms at v_min or v_max after clamping
            # (prevents all mass on one atom if tz is exactly on a support atom)
            # This ensures that mass is distributed even if l=u initially, unless tz is an exact boundary.
            l[(u > 0) * (l == u)] -= 1 
            l = torch.clamp(l, 0, self.num_atoms - 1) # Ensure l is within bounds after adjustment
            u[(l < (self.num_atoms - 1)) * (l == u)] += 1
            u = torch.clamp(u, 0, self.num_atoms - 1) # Ensure u is within bounds after adjustment

            # Projected distribution. Shape: (batch_size, num_atoms)
            projected_target_distribution = torch.zeros_like(next_action_target_q_probs)
            
            # Distribute probabilities
            # m_probs_next is next_action_target_q_probs (B, N)
            # l, u are (B, N) - indices for atoms
            # b is (B, N) - fractional position
            
            # Equivalent to:
            # for i in range(current_batch_size):
            #    for j in range(self.num_atoms):
            #        uidx, lidx = u[i, j], l[i, j]
            #        prob = next_action_target_q_probs[i, j]
            #        projected_target_distribution[i, lidx] += prob * (u[i, j].float() - b[i, j])
            #        projected_target_distribution[i, uidx] += prob * (b[i, j] - l[i, j].float())
            # Vectorized version:
            m_l = next_action_target_q_probs * (u.float() - b) # mass to lower index
            m_u = next_action_target_q_probs * (b - l.float()) # mass to upper index

            # Iterate over batch elements for scatter_add_ to avoid issues with duplicate indices in one call
            for i in range(current_batch_size):
                projected_target_distribution[i].scatter_add_(0, l[i], m_l[i])
                projected_target_distribution[i].scatter_add_(0, u[i], m_u[i])
        
        # Distributional Loss (Cross-Entropy)
        # current_action_q_logits: (batch_size, num_atoms)
        # projected_target_distribution: (batch_size, num_atoms), should be detached
        log_p_current = F.log_softmax(current_action_q_logits, dim=1)
        # Cross-entropy: -sum(target_prob * log(current_prob)) per sample
        element_wise_cross_entropy_loss = -(projected_target_distribution.detach() * log_p_current).sum(dim=1)
        
        # Log raw distributional loss (before IS weighting)
        self.writer.add_scalar('Loss/raw_dist_loss_mean', element_wise_cross_entropy_loss.mean().item(), global_step=self.update_counter)
        
        loss_td = (IS_weights_tensor * element_wise_cross_entropy_loss).mean()
        
        # Memory error (compare expected Q value of current action with q_memory)
        expected_q_current_action = (F.softmax(current_action_q_logits, dim=1) * self.support.view(1, -1)).sum(dim=1)
        memory_error = self.loss_func(expected_q_current_action, batch['q_memory']) # q_memory is scalar

        # KL_loss (compare expected Q values of all actions with teacher_q_values)
        # q_distribution_all_actions: (batch_size, action_dim, num_atoms)
        expected_q_all_actions = (F.softmax(q_distribution_all_actions, dim=2) * self.support.view(1, 1, -1)).sum(dim=2) # (B, A)
        
        teacher_q_values_batch = batch['teacher_q_values'] # (B, A)
        # Ensure teacher_q_values are probabilities if used in KLDiv like this
        # Assuming teacher_q_values_batch are logits that need softmax
        log_softmax_expected_q = F.log_softmax(expected_q_all_actions, dim=-1)
        softmax_teacher_q = F.softmax(teacher_q_values_batch, dim=-1)

        KL_loss = F.kl_div(
            log_softmax_expected_q,
            softmax_teacher_q,
            reduction="batchmean", # computes sum_i(target_i * (log(target_i) - log(input_i))) / batch_size
            log_target=False # if teacher_q_values_batch is already log_softmax, set True
        )
        # If teacher_q_values are already probabilities, kl_div expects log_target=False
        # If we want standard KL div D_KL(P || Q) = sum P * log(P/Q)
        # Here P = softmax_teacher_q, Q = softmax_expected_q
        # F.kl_div input is Q.log(), target is P
        # So this is D_KL( softmax_teacher_q || softmax_expected_q )
        
        loss = loss_td + args.alpha * memory_error + args.beta * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.hyperagent.parameters(), 1.0) # Clip grad norm
        self.optimizer.step()
        
        # Update target network
        for param, target_param in zip(self.hyperagent.parameters(), self.hyperagent_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update PER priorities
        abs_td_errors = element_wise_cross_entropy_loss.abs().detach().cpu().numpy()
        replay_buffer.update_priorities(tree_indices, abs_td_errors)
        
        self.update_counter += 1
        # Return expected Q-values for logging, consistent with previous version if possible
        # q_eval for current action, q_target could be expected value of projected distribution
        q_eval_log = expected_q_current_action.mean().detach().cpu()
        
        # For q_target_log, calculate expected value of the projected target distribution
        # This is tricky because projected_target_distribution is for chosen next actions based on online net,
        # but uses target net values.
        # A simpler consistent log might be the expected value of next_action_target_q_logits
        expected_next_q_target_log = (F.softmax(next_action_target_q_logits.detach(), dim=1) * self.support.view(1, -1)).sum(dim=1).mean().cpu()

        return loss_td.detach().cpu(), memory_error.detach().cpu(), KL_loss.detach().cpu(), q_eval_log, expected_next_q_target_log

    def act(self, state, state_trend, state_clf, info):
        x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dim
        x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device) # Add batch dim
        x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
        previous_action = torch.tensor([info["previous_action"]], dtype=torch.long, device=self.device).unsqueeze(0) # Add batch dim, ensure 2D

        if np.random.uniform() < (1-self.epsilon):
            with torch.no_grad():
                qs_list = [
                        self.slope_agents[i](x1, x2, previous_action) for i in range(3)
                    ] + [
                        self.vol_agents[i](x1, x2, previous_action) for i in range(3)
                    ]
                w = self.hyperagent(x1, x2, x3, previous_action)
                # actions_value_dist: (1, action_dim, num_atoms)
                actions_value_dist = self.calculate_q(w, qs_list)
                # expected_actions_value: (1, action_dim)
                expected_actions_value = (F.softmax(actions_value_dist, dim=2) * self.support.view(1, 1, -1)).sum(dim=2)
                action = torch.max(expected_actions_value, 1)[1].data.cpu().numpy()
                action = action[0]
        else:
            action_choice = [0,1] # Assuming self.n_action is 2
            action = random.choice(action_choice)
        return action

    def act_test(self, state, state_trend, state_clf, info):
        with torch.no_grad():
            x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Add batch dim
            x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device) # Add batch dim
            x3 = torch.FloatTensor(state_clf).unsqueeze(0).to(self.device)
            previous_action = torch.tensor([info["previous_action"]], dtype=torch.long, device=self.device).unsqueeze(0) # Add batch dim

            qs_list = [
                    self.slope_agents[i](x1, x2, previous_action) for i in range(3)
                ] + [
                    self.vol_agents[i](x1, x2, previous_action) for i in range(3)
                ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            # actions_value_dist: (1, action_dim, num_atoms)
            actions_value_dist = self.calculate_q(w, qs_list)
            # expected_actions_value: (1, action_dim)
            expected_actions_value = (F.softmax(actions_value_dist, dim=2) * self.support.view(1, 1, -1)).sum(dim=2)
            action = torch.max(expected_actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            return action

    def q_estimate(self, state, state_trend, state_clf, info):
        # Ensure inputs are correctly unsqueezed if they represent a single sample
        if state.ndim == 1: state = state[np.newaxis, :]
        if state_trend.ndim == 1: state_trend = state_trend[np.newaxis, :]
        if state_clf.ndim == 1: state_clf = state_clf[np.newaxis, :]
        
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        x3 = torch.FloatTensor(state_clf).to(self.device)
        
        # Ensure previous_action is a tensor and has a batch dimension
        if isinstance(info["previous_action"], (int, float)):
            prev_action_val = [info["previous_action"]]
        else: # Handle list or numpy array
            prev_action_val = info["previous_action"]
        if not isinstance(prev_action_val, torch.Tensor):
             previous_action = torch.tensor(prev_action_val, dtype=torch.long, device=self.device)
        else:
            previous_action = prev_action_val.to(dtype=torch.long, device=self.device)

        if previous_action.ndim == 1: # e.g. tensor([0])
             previous_action = previous_action.unsqueeze(0) # -> tensor([[0]]) to match batch size of x1,x2,x3
        elif previous_action.ndim == 0: # e.g. tensor(0)
             previous_action = previous_action.view(1,1)

        with torch.no_grad():
            qs_list = [
                    self.slope_agents[i](x1, x2, previous_action) for i in range(3)
                ] + [
                    self.vol_agents[i](x1, x2, previous_action) for i in range(3)
                ]
            w = self.hyperagent(x1, x2, x3, previous_action)
            # actions_value_dist: (batch_size, action_dim, num_atoms)
            actions_value_dist = self.calculate_q(w, qs_list)
            # expected_actions_value: (batch_size, action_dim)
            expected_actions_value = (F.softmax(actions_value_dist, dim=2) * self.support.view(1, 1, -1)).sum(dim=2)
            q = torch.max(expected_actions_value, 1)[0].detach().cpu().numpy() # Max expected Q over actions
            # If q_estimate is called with a single state, q will be a single value array.
            if q.shape[0] == 1:
                return q.item() # Return scalar if single estimate
        return q # Return array if batched estimate

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
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        step_counter = 0
        episode_counter = 0
        epoch_counter = 0
        best_return_rate = -float('inf')
        best_model = None
        self.replay_buffer = ReplayBuffer_High(args, self.n_state_1, self.n_state_2, self.n_action) 
        for sample in range(self.epoch_number):
            print('epoch ', epoch_counter + 1)
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
                

            epoch_counter += 1
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            mean_return_rate_train = np.mean(epoch_return_rate_train_list)
            mean_final_balance_train = np.mean(epoch_final_balance_train_list)
            mean_required_money_train = np.mean(epoch_required_money_train_list)
            mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
            self.writer.add_scalar(
                    tag="epoch_return_rate_train",
                    scalar_value=mean_return_rate_train,
                    global_step=epoch_counter,
                    walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_final_balance_train",
                scalar_value=mean_final_balance_train,
                global_step=epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_required_money_train",
                scalar_value=mean_required_money_train,
                global_step=epoch_counter,
                walltime=None,
                )
            self.writer.add_scalar(
                tag="epoch_reward_sum_train",
                scalar_value=mean_reward_sum_train,
                global_step=epoch_counter,
                walltime=None,
                )
            epoch_path = os.path.join(self.model_path,
                                        "epoch_{}".format(epoch_counter))
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
