import argparse
import os
import pathlib
import pickle
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

from env.low_level_env import Testing_Env, Training_Env
from model.net import subagent
from RL.util.replay_buffer import ReplayBuffer
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
parser.add_argument("--eval_update_freq",type=int,default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start",type=float,default=0.5)
parser.add_argument("--epsilon_end",type=float,default=0.1)
parser.add_argument("--decay_length",type=int,default=5)
parser.add_argument("--update_times",type=int,default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost",type=float,default=2.0 / 10000)
parser.add_argument("--back_time_length",type=int,default=1)
parser.add_argument("--seed",type=int,default=12345)
parser.add_argument("--n_step",type=int,default=1)
parser.add_argument("--epoch_number",type=int,default=15)
parser.add_argument("--label",type=str,default="label_1")
parser.add_argument("--clf",type=str,default="slope")
parser.add_argument("--alpha",type=float,default="0")
parser.add_argument("--device",type=str,default="cuda:0")


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def calculate_alpha(diff, k):
    alpha = 16 * (1 - torch.exp(-k * diff))
    return torch.clip(alpha, 0, 16)


class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.seed = args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(args.device)
        else:
            self.device = torch.device("cpu")
        
        # C51 parameters (consistent for agent and teacher generation)
        self.v_min = -10.0
        self.v_max = 10.0
        self.num_atoms = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

        # Multi-step learning parameters
        self.n_step = args.n_step
        self.multi_step_buffer = []  # For storing n-step transitions
        self.gamma = args.gamma

        self.result_path = os.path.join("./result/low_level", 
                                        '{}'.format(args.dataset), '{}.{}'.format(args.clf, "C51"), str(float(args.alpha)), args.label) # Use float for alpha in path
        self.label = int(args.label.split('_')[1])
        self.model_path = os.path.join(self.result_path,
                                       "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "train")
        self.val_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "val")
        self.test_data_path = os.path.join(ROOT, "MacroHFT",
                                        "data", args.dataset, "test")
        if args.clf == 'slope':
            with open(os.path.join(self.train_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'slope_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)
        elif args.clf == 'vol':
            with open(os.path.join(self.train_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.val_index = pickle.load(file)
            with open(os.path.join(self.test_data_path, 'vol_labels.pkl'), 'rb') as file:
                self.test_index = pickle.load(file)


        self.dataset=args.dataset
        self.clf = args.clf
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

        self.transcation_cost = args.transcation_cost
        self.back_time_length = args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)
        self.eval_net, self.target_net = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, self.num_atoms, self.v_min, self.v_max, use_noisy=True).to(self.device), subagent(
                self.n_state_1, self.n_state_2, self.n_action,
                64, self.num_atoms, self.v_min, self.v_max, use_noisy=True).to(self.device)
        self.hardupdate()
        self.update_times = args.update_times
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=args.lr)
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.n_step = args.n_step
        self.eval_update_freq = args.eval_update_freq
        self.buffer_size = args.buffer_size
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.decay_length = args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(start_epsilon=self.epsilon_start, end_epsilon=self.epsilon_end, decay_length=self.decay_length)
        self.epsilon = args.epsilon_start

    def update(self, replay_buffer):
        self.eval_net.train()
        batch, tree_indices, IS_weights = replay_buffer.sample()
        if batch is None:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

        IS_weights_tensor = torch.tensor(IS_weights, device=self.device, dtype=torch.float32).squeeze()

        if hasattr(replay_buffer, 'PER_b') and hasattr(replay_buffer, 'max_priority'):
            self.writer.add_scalar('PER/beta', replay_buffer.PER_b, global_step=self.update_counter)
            self.writer.add_scalar('PER/avg_IS_weight', IS_weights_tensor.mean().item(), global_step=self.update_counter)
            self.writer.add_scalar('PER/max_IS_weight', IS_weights_tensor.max().item(), global_step=self.update_counter)
            if hasattr(replay_buffer, 'max_priority'):
                 self.writer.add_scalar('PER/buffer_max_priority', replay_buffer.max_priority, global_step=self.update_counter)


        current_batch_size = batch['state'].shape[0]
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # C51: Calculate current Q-distribution for taken actions
        q_logits_current = self.eval_net(batch['state'], batch['state_trend'], batch['previous_action'])
        # Gather the logits for the action taken
        action_indices = batch['action'].long()
        # Reshape action_indices to (batch_size, 1, 1) and expand to match q_logits_current dim 2
        action_indices_expanded = action_indices.unsqueeze(1).expand(-1, -1, self.num_atoms)
        q_log_dist_current_action = F.log_softmax(q_logits_current.gather(1, action_indices_expanded).squeeze(1), dim=1)


        with torch.no_grad():
            # C51: Calculate target Q-distribution
            q_logits_next = self.target_net(batch['next_state'], batch['next_state_trend'], batch['next_previous_action'])
            q_dist_next = F.softmax(q_logits_next, dim=2) # (batch_size, num_actions, num_atoms)

            # DDQN: Use eval_net to select best next action
            q_logits_next_eval = self.eval_net(batch['next_state'], batch['next_state_trend'], batch['next_previous_action'])
            q_dist_next_eval = F.softmax(q_logits_next_eval, dim=2)
            expected_q_values_next_eval = (q_dist_next_eval * self.support.unsqueeze(0).unsqueeze(0)).sum(2) # (batch_size, num_actions)
            next_best_actions = expected_q_values_next_eval.argmax(dim=1) # (batch_size,)
            
            # Gather the Q-distribution for the next_best_actions from target_net's output
            next_best_actions_expanded = next_best_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.num_atoms)
            next_q_dist_target_action = q_dist_next.gather(1, next_best_actions_expanded).squeeze(1) # (batch_size, num_atoms)

            # Compute target distribution projection with n-step discount
            rewards = batch['reward'].unsqueeze(1) 
            terminals = batch['terminal'].unsqueeze(1)
            
            # Use n-step discount factor
            gamma_n = self.gamma ** self.n_step
            
            Tz = rewards + (1 - terminals) * gamma_n * self.support.unsqueeze(0) # (batch_size, num_atoms)
            Tz = Tz.clamp(min=self.v_min, max=self.v_max)

            b = (Tz - self.v_min) / self.delta_z # (batch_size, num_atoms)
            l = b.floor().long()
            u = b.ceil().long()

            # Fix for l == u cases, important for atoms at boundaries
            # If l == u, it means Tz is exactly on an atom.
            # Probability mass should go to l (or u).
            # The (u-b) and (b-l) terms handle this naturally if not for floating point issues sometimes.
            # Explicitly handle boundary cases for l and u for safety:
            l.clamp_(0, self.num_atoms - 1)
            u.clamp_(0, self.num_atoms - 1)
            
            m = torch.zeros(current_batch_size, self.num_atoms, device=self.device)
            # Broadcasting for batch operation (more efficient than loop)
            # Create batch indices for advanced indexing
            batch_indices = torch.arange(current_batch_size, device=self.device).unsqueeze(1).expand_as(next_q_dist_target_action)

            # Distribute probability of next_q_dist_target_action[b_idx, atom_idx]
            # to m[b_idx, l[b_idx, atom_idx]] and m[b_idx, u[b_idx, atom_idx]]
            m.scatter_add_(1, l, next_q_dist_target_action * (u.float() - b))
            m.scatter_add_(1, u, next_q_dist_target_action * (b - l.float()))
            # Ensure m sums to 1 (or very close) per row - usually it does due to projection.
            # It represents the target probability distribution.

        # C51: Calculate loss (Categorical Cross-Entropy)
        # q_log_dist_current_action is already log_softmax
        element_wise_loss = - (m * q_log_dist_current_action).sum(1) # (batch_size,)
        
        self.writer.add_scalar('TD_Error/abs_mean_raw_C51_XEntropy', element_wise_loss.abs().mean().item(), global_step=self.update_counter)

        loss_td = (IS_weights_tensor * element_wise_loss).mean()

        # Imitation Learning KL_loss 
        KL_loss = torch.tensor(0.0, device=self.device)
        alpha_imitation = float(args.alpha) # Ensure alpha is float for comparison

        if alpha_imitation > 0:
            # teacher_q_values from buffer has shape (batch_size, action_dim, num_atoms)
            teacher_q_dist_all_actions = batch['teacher_q_values'] 
            
            # Gather the teacher distribution for the actions taken by the agent
            # action_indices is (batch_size, 1), needs to be (batch_size, 1, 1) and expanded for gather
            action_indices_for_teacher_gather = action_indices.unsqueeze(2).expand(-1, -1, self.num_atoms)
            teacher_dist_taken_action = teacher_q_dist_all_actions.gather(1, action_indices_for_teacher_gather).squeeze(1)
            # teacher_dist_taken_action shape: (batch_size, num_atoms)

            # q_log_dist_current_action is log_softmax output from eval_net for the taken action
            # teacher_dist_taken_action contains probabilities (from one-hot encoding in make_q_table_reward)
            
            # Ensure no zero probabilities in teacher_dist_taken_action before taking log if log_target=True
            # However, F.kl_div with log_target=False expects target to be probabilities.
            # We also need to handle the case where teacher_dist_taken_action might be zero everywhere if the
            # one-hot encoding failed or if the Q value was exactly on a boundary in a way that scatter_add missed it.
            # For one-hot, it should be fine. Add a small epsilon for stability if needed, but F.kl_div should handle it.

            KL_loss_val = F.kl_div(
                input=q_log_dist_current_action,  # log-probabilities from agent
                target=teacher_dist_taken_action, # probabilities from teacher
                reduction="batchmean",
                log_target=False # Specifies that target is not log-probabilities
            )
            KL_loss = KL_loss_val

        loss = loss_td + alpha_imitation * KL_loss
        
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 10) # Using 10 for C51, can be tuned
        self.optimizer.step()
        for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update PER priorities using the element-wise C51 loss
        abs_errors_for_per = element_wise_loss.abs().detach().cpu().numpy()
        replay_buffer.update_priorities(tree_indices, abs_errors_for_per)

        self.update_counter += 1
        self.eval_net.eval()
        
        # Calculate expected Q-values for logging
        with torch.no_grad():
            q_dist_current_taken_action_softmax = F.softmax(q_logits_current.gather(1, action_indices_expanded).squeeze(1), dim=1)
            expected_q_current = (q_dist_current_taken_action_softmax * self.support.unsqueeze(0)).sum(1).mean()
            
            # Target Q value is based on projected distribution m
            expected_q_target = (m * self.support.unsqueeze(0)).sum(1).mean()


        return loss_td.detach().cpu(), KL_loss.detach().cpu(), expected_q_current.cpu(), expected_q_target.cpu()

    def hardupdate(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def reset_noise(self):
        """Reset noise in noisy networks"""
        self.eval_net.reset_noise()
        self.target_net.reset_noise()

    def act(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device)
        previous_action = torch.tensor([info["previous_action"]], dtype=torch.long).to(self.device)
        
        # With noisy networks, we can reduce epsilon-greedy exploration
        epsilon_threshold = self.epsilon if not hasattr(self.eval_net, 'use_noisy') or not self.eval_net.use_noisy else self.epsilon * 0.1
        
        if np.random.uniform() < (1 - epsilon_threshold):
            self.eval_net.eval()
            with torch.no_grad():
                q_logits = self.eval_net(x1, x2, previous_action)
                q_dist = F.softmax(q_logits, dim=2)
                # Calculate expected Q-values from the distribution
                expected_q_values = (q_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(2)
                action = torch.max(expected_q_values, 1)[1].data.cpu().numpy()[0]
        else:
            action_choice = [0,1]
            action = random.choice(action_choice)
        return action

    def act_test(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        x2 = torch.FloatTensor(state_trend).unsqueeze(0).to(self.device)
        previous_action = torch.tensor([info["previous_action"]], dtype=torch.long).to(self.device)
        
        self.eval_net.eval()
        with torch.no_grad():
            q_logits = self.eval_net(x1, x2, previous_action)
            q_dist = F.softmax(q_logits, dim=2)
            expected_q_values = (q_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(2)
            action = torch.max(expected_q_values, 1)[1].data.cpu().numpy()[0]
        return action

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        df_list = self.train_index[self.label]
        df_number=int(len(df_list))       
        step_counter = 0
        episode_counter = 0
        epoch_counter = 0        
        self.replay_buffer = ReplayBuffer(args, self.n_state_1, self.n_state_2, self.n_action, num_atoms=self.num_atoms)
        best_return_rate = -float('inf')
        best_model = None
        for sample in range(self.epoch_number):
            epoch_start_time = time.time()
            print('epoch ', epoch_counter + 1)
            random_list = self.train_index[self.label]
            random.shuffle(random_list)
            random_position_list = random.choices(range(self.n_action), k=df_number)
            print(random_list)
            
            for i in range(df_number):
                df_index = random_list[i]
                print("training with df", df_index)
                self.df = pd.read_feather(
                    os.path.join(self.train_data_path, "df_{}.feather".format(df_index)))
                self.eval_net.eval()
                               
                train_env = Training_Env(
                        df=self.df,
                        tech_indicator_list=self.tech_indicator_list,
                        tech_indicator_list_trend=self.tech_indicator_list_trend,
                        transcation_cost=self.transcation_cost,
                        back_time_length=self.back_time_length,
                        max_holding_number=self.max_holding_number,
                        initial_action=random_position_list[i],
                        alpha = float(args.alpha), # Pass agent's alpha, though env doesn't use it directly for q_table gen
                        # Pass C51 parameters consistent with the agent for teacher signal generation
                        v_min_teacher=self.v_min,
                        v_max_teacher=self.v_max,
                        num_atoms_teacher=self.num_atoms
                        )
                s, s2, info = train_env.reset()
                episode_reward_sum = 0
                
                # Reset noise at the beginning of each episode for better exploration
                self.reset_noise()
                
                while True:
                    a = self.act(s, s2, info)
                    s_, s2_, r, done, info_ = train_env.step(a)
                    self.store_transition_with_nstep(s, s2, info['previous_action'], info['q_value'], a, r, s_, s2_, info_['previous_action'],
                                    info_['q_value'], done)
                    episode_reward_sum += r

                    s, s2, info = s_, s2_, info_
                    step_counter += 1
                    if step_counter % self.eval_update_freq == 0 and step_counter > (
                            self.batch_size + self.n_step):
                        # Reset noise before training updates for better gradient estimates
                        self.reset_noise()
                        for i in range(self.update_times):
                            td_error, KL_loss, q_eval, q_target = self.update(self.replay_buffer)
                            if self.update_counter % self.q_value_memorize_freq == 1:
                                self.writer.add_scalar(
                                    tag="td_error",
                                    scalar_value=td_error,
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
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            self.epsilon = self.epsilon_scheduler.get_epsilon(epoch_counter)
            mean_return_rate_train = np.mean(epoch_return_rate_train_list)
            mean_final_balance_train = np.mean(epoch_final_balance_train_list)
            mean_required_money_train = np.mean(epoch_required_money_train_list)
            mean_reward_sum_train = np.mean(epoch_reward_sum_train_list)
            print(f"Epoch {epoch_counter} completed in {epoch_duration:.2f}s. Avg Train Balance: {mean_final_balance_train:.2f}, Avg Train Required Money: {mean_required_money_train:.2f}")
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
            torch.save(self.eval_net.state_dict(),
                        os.path.join(epoch_path, "trained_model.pkl"))
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                    os.makedirs(val_path)
            return_rate_0 = self.val_cluster(epoch_path, val_path, 0)
            return_rate_1 = self.val_cluster(epoch_path, val_path, 1)
            return_rate_eval = (return_rate_0 + return_rate_1) / 2
            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model = self.eval_net.state_dict()
                print("best model updated to epoch ", epoch_counter)
            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []
        best_model_dir = os.path.join("./result/low_level", 
                                    '{}'.format(self.dataset), '{}.{}'.format(self.clf, "C51"), str(float(args.alpha)), args.label, "seed_{}".format(self.seed))

        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        best_model_path = os.path.join(best_model_dir, 'best_model.pkl')
        torch.save(best_model, best_model_path)
        print(f"Best model saved to: {best_model_path}")
        print(f"Best validation return rate: {best_return_rate}")

    def val_cluster(self, epoch_path, save_path, initial_action):
        self.eval_net.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl")))
        self.eval_net.eval()
        df_list = self.val_index[self.label]
        df_number=int(len(df_list)) 
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        for i in range(df_number):
            print("validating on df", df_list[i])
            self.df = pd.read_feather(
                os.path.join(self.val_data_path, "df_{}.feather".format(df_list[i])))
            
            val_env = Testing_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=initial_action)
            s, s2, info = val_env.reset()
            done = False
            action_list_episode = []
            reward_list_episode = []
            while not done:
                a = self.act_test(s, s2, info)
                s_, s2_, r, done, info_ = val_env.step(a)
                reward_list_episode.append(r)
                s, s2, info = s_, s2_, info_
                action_list_episode.append(a)
            portfit_magine, final_balance, required_money, commission_fee = val_env.get_final_return_rate(
                slient=True)
            final_balance = val_env.final_balance
            required_money = val_env.required_money
            print(f"  > df {df_list[i]}: final_balance={final_balance:.2f}, required_money={required_money:.2f}, profit_margin={(final_balance/required_money if required_money != 0 else 0):.4f}")
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
        np.save(os.path.join(save_path, "action_val_{}.npy".format(initial_action)), action_list)
        np.save(os.path.join(save_path, "reward_val_{}.npy".format(initial_action)), reward_list)
        np.save(os.path.join(save_path, "final_balance_val_{}.npy".format(initial_action)),
            final_balance_list)
        np.save(os.path.join(save_path, "require_money_val_{}.npy".format(initial_action)),
                required_money_list)
        np.save(os.path.join(save_path, "commission_fee_history_val_{}.npy".format(initial_action)),
                commission_fee_list)
        return_rate_mean = np.nan_to_num(final_balance_list / required_money_list).mean()
        np.save(os.path.join(save_path, "return_rate_mean_val_{}.npy".format(initial_action)),
                return_rate_mean)
        return return_rate_mean

    def compute_n_step_returns(self, batch_size):
        """Compute n-step returns for current batch"""
        if self.n_step == 1:
            return None  # Use regular 1-step returns
        
        n_step_returns = []
        n_step_states = []
        n_step_dones = []
        
        for i in range(batch_size):
            if len(self.multi_step_buffer) >= self.n_step:
                # Calculate n-step return
                n_step_return = 0
                state_idx = max(0, len(self.multi_step_buffer) - self.n_step)
                
                for j in range(self.n_step):
                    if state_idx + j < len(self.multi_step_buffer):
                        transition = self.multi_step_buffer[state_idx + j]
                        reward = transition['reward']
                        done = transition['done']
                        
                        n_step_return += (self.gamma ** j) * reward
                        
                        if done:
                            n_step_states.append(transition['next_state'])
                            n_step_dones.append(True)
                            break
                    
                if not done and state_idx + self.n_step - 1 < len(self.multi_step_buffer):
                    # If episode didn't end, use next state after n steps
                    final_transition = self.multi_step_buffer[state_idx + self.n_step - 1]
                    n_step_states.append(final_transition['next_state'])
                    n_step_dones.append(False)
                
                n_step_returns.append(n_step_return)
            else:
                n_step_returns.append(0)  # Not enough steps yet
                n_step_states.append(None)
                n_step_dones.append(False)
        
        return {
            'n_step_returns': n_step_returns,
            'n_step_states': n_step_states,
            'n_step_dones': n_step_dones
        }

    def store_transition_with_nstep(self, state, state_trend, previous_action, teacher_q_values, 
                                   action, reward, next_state, next_state_trend, next_previous_action, 
                                   next_teacher_q_values, terminal):
        """Store transition with n-step learning support"""
        transition = {
            'state': state,
            'state_trend': state_trend,
            'previous_action': previous_action,
            'teacher_q_values': teacher_q_values,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'next_state_trend': next_state_trend,
            'next_previous_action': next_previous_action,
            'next_teacher_q_values': next_teacher_q_values,
            'done': terminal
        }
        
        self.multi_step_buffer.append(transition)
        
        # Keep buffer size manageable
        if len(self.multi_step_buffer) > self.n_step * 2:
            self.multi_step_buffer.pop(0)
        
        # Store in replay buffer when we have enough steps or episode ends
        if len(self.multi_step_buffer) >= self.n_step or terminal:
            if len(self.multi_step_buffer) >= self.n_step:
                # Get the transition from n_step ago
                base_transition = self.multi_step_buffer[-self.n_step]
                
                # Calculate n-step return
                n_step_return = 0
                gamma_power = 1
                
                for i in range(self.n_step):
                    if len(self.multi_step_buffer) > i:
                        step_transition = self.multi_step_buffer[-(self.n_step - i)]
                        n_step_return += gamma_power * step_transition['reward']
                        gamma_power *= self.gamma
                        
                        if step_transition['done']:
                            break
                
                # Store with n-step return
                self.replay_buffer.store_transition(
                    base_transition['state'],
                    base_transition['state_trend'], 
                    base_transition['previous_action'],
                    base_transition['teacher_q_values'],
                    base_transition['action'],
                    n_step_return,  # Use n-step return instead of single reward
                    transition['next_state'],  # Use current next_state
                    transition['next_state_trend'],
                    transition['next_previous_action'],
                    transition['next_teacher_q_values'],
                    terminal
                )
            
            # Clear buffer if episode ends
            if terminal:
                self.multi_step_buffer.clear()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
