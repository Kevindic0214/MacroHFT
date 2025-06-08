import argparse
import os
import pathlib
import pickle
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

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.append(ROOT)
sys.path.insert(0, ".")

print(f"Current file: {__file__}")
print(f"ROOT directory: {ROOT}")
print(f"Python path: {sys.path}")

from env.low_level_env import Testing_Env, Training_Env
from model.net import *
from RL.util.replay_buffer import ReplayBuffer
from RL.util.utili import LinearDecaySchedule

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["F_ENABLE_ONEDNN_OPTS"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--buffer_size",
    type=int,
    default=1000000,
)
parser.add_argument("--dataset", type=str, default="ETHUSDT")
parser.add_argument(
    "--q_value_memorize_freq",
    type=int,
    default=10,
)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--eval_update_freq", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epsilon_start", type=float, default=0.5)
parser.add_argument("--epsilon_end", type=float, default=0.1)
parser.add_argument("--decay_length", type=int, default=5)
parser.add_argument("--update_times", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--transcation_cost", type=float, default=2.0 / 10000)
parser.add_argument("--back_time_length", type=int, default=1)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--epoch_number", type=int, default=15)
parser.add_argument("--label", type=str, default="label_1")
parser.add_argument("--clf", type=str, default="slope")
parser.add_argument("--alpha", type=float, default=0.0)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_quantiles", type=int, default=51)
parser.add_argument(
    "--kappa", type=float, default=1.0
)


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_alpha(diff, k):
    alpha = 16 * (1 - torch.exp(-k * diff))
    return torch.clip(alpha, 0, 16)


class DQN(object):
    def __init__(self, args):
        self.args = args
        self.seed = self.args.seed
        seed_torch(self.seed)
        if torch.cuda.is_available():
            self.device = torch.device(self.args.device)
        else:
            self.device = torch.device("cpu")

        self.alpha_kl = float(self.args.alpha)

        self.result_path = os.path.join(
            ROOT,
            "result",
            "low_level",
            "{}".format(self.args.dataset),
            "{}".format(self.args.clf),
            str(int(self.args.alpha)),
            self.args.label,
        )
        self.label = int(self.args.label.split("_")[1])
        self.model_path = os.path.join(self.result_path, "seed_{}".format(self.seed))
        self.train_data_path = os.path.join(
            ROOT, "data", self.args.dataset, "train"
        )
        self.val_data_path = os.path.join(ROOT, "data", self.args.dataset, "val")
        self.test_data_path = os.path.join(
            ROOT, "data", self.args.dataset, "test"
        )
        if self.args.clf == "slope":
            with open(
                os.path.join(self.train_data_path, "slope_labels.pkl"), "rb"
            ) as file:
                self.train_index = pickle.load(file)
            with open(
                os.path.join(self.val_data_path, "slope_labels.pkl"), "rb"
            ) as file:
                self.val_index = pickle.load(file)
            with open(
                os.path.join(self.test_data_path, "slope_labels.pkl"), "rb"
            ) as file:
                self.test_index = pickle.load(file)
        elif self.args.clf == "vol":
            with open(
                os.path.join(self.train_data_path, "vol_labels.pkl"), "rb"
            ) as file:
                self.train_index = pickle.load(file)
            with open(os.path.join(self.val_data_path, "vol_labels.pkl"), "rb") as file:
                self.val_index = pickle.load(file)
            with open(
                os.path.join(self.test_data_path, "vol_labels.pkl"), "rb"
            ) as file:
                self.test_index = pickle.load(file)

        self.dataset = self.args.dataset
        self.clf = self.args.clf
        if "BTC" in self.dataset:
            self.max_holding_number = 0.01
        elif "ETH" in self.dataset:
            self.max_holding_number = 0.2
        elif "DOT" in self.dataset:
            self.max_holding_number = 10
        elif "LTC" in self.dataset:
            self.max_holding_number = 10
        else:
            raise Exception("we do not support other dataset yet")
        self.epoch_number = self.args.epoch_number

        self.log_path = os.path.join(self.model_path, "log")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)
        self.update_counter = 0
        self.q_value_memorize_freq = self.args.q_value_memorize_freq

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.tech_indicator_list = np.load(
            os.path.join(ROOT, "data", "feature_list", "single_features.npy"), allow_pickle=True
        ).tolist()
        self.tech_indicator_list_trend = np.load(
            os.path.join(ROOT, "data", "feature_list", "trend_features.npy"), allow_pickle=True
        ).tolist()

        self.transcation_cost = self.args.transcation_cost
        self.back_time_length = self.args.back_time_length
        self.n_action = 2
        self.n_state_1 = len(self.tech_indicator_list)
        self.n_state_2 = len(self.tech_indicator_list_trend)

        # QR-DQN specific parameters
        self.num_quantiles = self.args.num_quantiles
        self.kappa = self.args.kappa
        # Precompute tau values (cumulative probabilities) for quantile midpoints
        self.tau_values = (
            torch.arange(self.num_quantiles, device=self.device, dtype=torch.float32)
            + 0.5
        ) / self.num_quantiles

        self.eval_net, self.target_net = subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, self.num_quantiles
        ).to(self.device), subagent(
            self.n_state_1, self.n_state_2, self.n_action, 64, self.num_quantiles
        ).to(
            self.device
        )
        self.hardupdate()
        self.update_times = self.args.update_times
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.lr)
        self.batch_size = self.args.batch_size
        self.gamma = self.args.gamma
        self.tau = self.args.tau
        self.n_step = self.args.n_step
        self.eval_update_freq = self.args.eval_update_freq
        self.buffer_size = self.args.buffer_size
        self.epsilon_start = self.args.epsilon_start
        self.epsilon_end = self.args.epsilon_end
        self.decay_length = self.args.decay_length
        self.epsilon_scheduler = LinearDecaySchedule(
            start_epsilon=self.epsilon_start,
            end_epsilon=self.epsilon_end,
            decay_length=self.decay_length,
        )
        self.epsilon = self.args.epsilon_start

    def update(self, replay_buffer):
        self.eval_net.train()
        batch, _, _ = replay_buffer.sample()
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            # Get next state quantiles from target network
            next_q_quantiles_target = self.target_net(
                batch["next_state"],
                batch["next_state_trend"],
                batch["next_previous_action"],
            )
            # Select next actions using eval_net (mean of quantiles)
            next_q_values_eval = self.eval_net(
                batch["next_state"],
                batch["next_state_trend"],
                batch["next_previous_action"],
            ).mean(dim=2)
            a_argmax = next_q_values_eval.argmax(
                dim=-1, keepdim=True
            )

            # Gather quantiles for the selected next actions from target_net's output
            a_argmax_expanded = a_argmax.unsqueeze(-1).expand(
                -1, -1, self.num_quantiles
            )
            next_best_quantiles = next_q_quantiles_target.gather(
                1, a_argmax_expanded
            ).squeeze(
                1
            )

            # Compute target quantile values (Bellman update for quantiles)
            # target_z_j = r + gamma * z_target_j(s', a'*)
            target_z = (
                batch["reward"].unsqueeze(-1)
                + self.gamma
                * (1 - batch["terminal"].unsqueeze(-1))
                * next_best_quantiles
            )
            # target_z shape: (batch_size, num_quantiles)

        # Get current state quantiles from eval_net
        current_q_quantiles_eval = self.eval_net(
            batch["state"], batch["state_trend"], batch["previous_action"]
        )
        # current_q_quantiles_eval shape: (batch_size, action_dim, num_quantiles)

        # Gather the quantiles for the actions taken in the batch
        action_batch = batch["action"].long()
        action_batch_expanded = action_batch.unsqueeze(-1).expand(
            -1, -1, self.num_quantiles
        )
        current_z = current_q_quantiles_eval.gather(1, action_batch_expanded).squeeze(1)
        # current_z shape: (batch_size, num_quantiles)

        # Compute Quantile Huber Loss
        # delta_ij = target_z_j - current_z_i
        # target_z has N_target quantiles (self.num_quantiles)
        # current_z has N_current quantiles (self.num_quantiles)
        delta_ij = target_z.unsqueeze(1) - current_z.unsqueeze(2)
        # delta_ij shape: (batch_size, N_eval, N_target)
        
        abs_delta_ij = torch.abs(delta_ij)
        huber_loss_values = torch.where(
            abs_delta_ij <= self.kappa,
            0.5 * delta_ij.pow(2),
            self.kappa * (abs_delta_ij - 0.5 * self.kappa),
        )

        # self.tau_values is tau_i (for current quantiles)
        # Shape: (N_eval_quantiles,) -> reshape to (1, N_eval, 1) for broadcasting
        tau_i = self.tau_values.unsqueeze(0).unsqueeze(-1)

        # Pairwise quantile loss: |tau_i - I(delta_ij < 0)| * huber_loss
        quantile_huber_loss_pairwise = (
            torch.abs(tau_i - (delta_ij < 0).float()) * huber_loss_values
        )

        # Average over target quantiles (dim 2), sum over current quantiles (dim 1), mean over batch
        quantile_regression_loss = quantile_huber_loss_pairwise.mean(dim=2).sum(dim=1).mean()

        # Clamp the loss to prevent numerical instability
        quantile_regression_loss = torch.clamp(quantile_regression_loss, max=100.0)

        # KL divergence for imitation learning (optional)
        # Use mean of quantiles for Q-values in KL divergence
        q_values_for_kl = current_q_quantiles_eval.mean(
            dim=2
        )
        demonstration = batch["demo_action"]

        # Ensure q_values_for_kl and demonstration have the same shape for kl_div
        # demonstration might be (batch_size, action_dim) if it's already a distribution
        # or (batch_size, 1) if it's an action index. Assuming it's a distribution.
        if demonstration.shape != q_values_for_kl.shape:
            # This case needs to be handled based on what demo_action contains.
            # If demo_action is an index, convert to one-hot then to softmax-like dist.
            # for now, assuming demo_action is already a probability distribution over actions.
            # If it's one-hot, it should be fine with softmax on q_values_for_kl.
            # The original code had: (demonstration.softmax(dim=-1) + 1e-8)
            # This implies demonstration might not be a proper distribution.
            # Let's assume demonstration is like q_table from the environment.
            pass

        KL_loss = F.kl_div(
            (q_values_for_kl.softmax(dim=-1) + 1e-8).log(),
            (
                demonstration.softmax(dim=-1) + 1e-8
            ),
            reduction="batchmean",
        )

        loss = quantile_regression_loss + self.alpha_kl * KL_loss
        self.optimizer.zero_grad()
        loss.backward()

        # Add gradient monitoring for debugging
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1)
        
        # Log gradient information occasionally
        if self.update_counter % (self.q_value_memorize_freq * 10) == 1:
            self.writer.add_scalar(
                tag="gradient_norm",
                scalar_value=total_grad_norm,
                global_step=self.update_counter,
                walltime=None,
            )
        
        self.optimizer.step()
        for param, target_param in zip(
            self.eval_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        self.update_counter += 1
        self.eval_net.eval()

        # For logging
        q_current_mean_for_log = current_z.mean(dim=1).mean().cpu()
        q_target_mean_for_log = target_z.mean(dim=1).mean().cpu()

        return (
            quantile_regression_loss.cpu(),
            KL_loss.cpu(),
            q_current_mean_for_log,
            q_target_mean_for_log,
        )

    def hardupdate(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def act(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        
        # Normalize states to prevent numerical issues
        x1 = torch.nan_to_num(x1, nan=0.0, posinf=1e6, neginf=-1e6)
        x2 = torch.nan_to_num(x2, nan=0.0, posinf=1e6, neginf=-1e6)
        
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long().to(self.device),
            0,
        ).to(self.device)
        if np.random.uniform() < (1 - self.epsilon):
            # Get quantiles from network
            actions_value_quantiles = self.eval_net(
                x1, x2, previous_action
            )
            # Take mean over quantiles to get Q-values
            actions_value = actions_value_quantiles.mean(
                dim=2
            )
            
            # Check for NaN or Inf values and handle gracefully
            if torch.isnan(actions_value).any() or torch.isinf(actions_value).any():
                print("Warning: NaN or Inf detected in Q-values, using random action")
                action_choice = [0, 1]
                action = random.choice(action_choice)
            else:
                action = torch.max(actions_value, 1)[1].data.cpu().numpy()
                action = action[0]
        else:
            action_choice = [0, 1]
            action = random.choice(action_choice)
        return action

    def act_test(self, state, state_trend, info):
        x1 = torch.FloatTensor(state).to(self.device)
        x2 = torch.FloatTensor(state_trend).to(self.device)
        
        # Normalize states to prevent numerical issues
        x1 = torch.nan_to_num(x1, nan=0.0, posinf=1e6, neginf=-1e6)
        x2 = torch.nan_to_num(x2, nan=0.0, posinf=1e6, neginf=-1e6)
        
        previous_action = torch.unsqueeze(
            torch.tensor(info["previous_action"]).long(), 0
        ).to(self.device)
        # Get quantiles from network
        actions_value_quantiles = self.eval_net(
            x1, x2, previous_action
        )
        # Take mean over quantiles to get Q-values
        actions_value = actions_value_quantiles.mean(
            dim=2
        )
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def train(self):
        epoch_return_rate_train_list = []
        epoch_final_balance_train_list = []
        epoch_required_money_train_list = []
        epoch_reward_sum_train_list = []
        df_list = self.train_index[self.label]
        df_number = int(len(df_list))
        step_counter = 0
        episode_counter = 0
        epoch_counter = 0
        self.replay_buffer = ReplayBuffer(
            self.args, self.n_state_1, self.n_state_2, self.n_action
        )
        best_return_rate = -float("inf")
        best_model_state_dict = None
        for sample in range(self.epoch_number):
            print("epoch ", epoch_counter + 1)
            random_list = self.train_index[self.label]
            random.shuffle(random_list)
            random_position_list = random.choices(range(self.n_action), k=df_number)
            print(random_list)

            for i in range(df_number):
                df_index = random_list[i]
                print("training with df", df_index)
                self.df = pd.read_feather(
                    os.path.join(self.train_data_path, "df_{}.feather".format(df_index))
                )
                self.eval_net.eval()

                train_env = Training_Env(
                    df=self.df,
                    tech_indicator_list=self.tech_indicator_list,
                    tech_indicator_list_trend=self.tech_indicator_list_trend,
                    transcation_cost=self.transcation_cost,
                    back_time_length=self.back_time_length,
                    max_holding_number=self.max_holding_number,
                    initial_action=random_position_list[i],
                    alpha=0,
                )
                s, s2, info = train_env.reset()
                episode_reward_sum = 0

                while True:
                    a = self.act(s, s2, info)
                    s_, s2_, r, done, info_ = train_env.step(a)
                    self.replay_buffer.store_transition(
                        s,
                        s2,
                        info["previous_action"],
                        info["q_value"],
                        a,
                        r,
                        s_,
                        s2_,
                        info_["previous_action"],
                        info_["q_value"],
                        done,
                    )
                    episode_reward_sum += r

                    s, s2, info = s_, s2_, info_
                    step_counter += 1
                    if step_counter % self.eval_update_freq == 0 and step_counter > (
                        self.batch_size + self.n_step
                    ):
                        for _ in range(self.update_times):
                            quantile_loss_val, kl_loss_val, q_eval_mean_val, q_target_mean_val = self.update(
                                self.replay_buffer
                            )
                            if self.update_counter % self.q_value_memorize_freq == 1:
                                self.writer.add_scalar(
                                    tag="quantile_regression_loss",
                                    scalar_value=quantile_loss_val,
                                    global_step=self.update_counter,
                                    walltime=None,
                                )
                                self.writer.add_scalar(
                                    tag="KL_loss",
                                    scalar_value=kl_loss_val,
                                    global_step=self.update_counter,
                                    walltime=None,
                                )
                                self.writer.add_scalar(
                                    tag="q_eval_mean",
                                    scalar_value=q_eval_mean_val,
                                    global_step=self.update_counter,
                                    walltime=None,
                                )
                                self.writer.add_scalar(
                                    tag="q_target_mean",
                                    scalar_value=q_target_mean_val,
                                    global_step=self.update_counter,
                                    walltime=None,
                                )
                    if done:
                        break
                episode_counter += 1
                final_balance, required_money = (
                    train_env.final_balance,
                    train_env.required_money,
                )
                self.writer.add_scalar(
                    tag="return_rate_train",
                    scalar_value=final_balance / (required_money) if required_money != 0 else 0,
                    global_step=episode_counter,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="final_balance_train",
                    scalar_value=final_balance,
                    global_step=episode_counter,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="required_money_train",
                    scalar_value=required_money,
                    global_step=episode_counter,
                    walltime=None,
                )
                self.writer.add_scalar(
                    tag="reward_sum_train",
                    scalar_value=episode_reward_sum,
                    global_step=episode_counter,
                    walltime=None,
                )
                if required_money != 0:
                    epoch_return_rate_train_list.append(final_balance / required_money)
                epoch_final_balance_train_list.append(final_balance)
                epoch_required_money_train_list.append(required_money)
                epoch_reward_sum_train_list.append(episode_reward_sum)

            epoch_counter += 1
            self.epsilon = self.epsilon_scheduler.get_epsilon(step_counter)
            mean_return_rate_train = np.mean(epoch_return_rate_train_list) if epoch_return_rate_train_list else 0
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
            epoch_path = os.path.join(self.model_path, "epoch_{}".format(epoch_counter))
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            torch.save(
                self.eval_net.state_dict(),
                os.path.join(epoch_path, "trained_model.pkl"),
            )
            val_path = os.path.join(epoch_path, "val")
            if not os.path.exists(val_path):
                os.makedirs(val_path)
            return_rate_0 = self.val_cluster(epoch_path, val_path, 0)
            return_rate_1 = self.val_cluster(epoch_path, val_path, 1)
            return_rate_eval = (return_rate_0 + return_rate_1) / 2
            if return_rate_eval > best_return_rate:
                best_return_rate = return_rate_eval
                best_model_state_dict = self.eval_net.state_dict()
                print("best model updated to epoch ", epoch_counter)
            epoch_return_rate_train_list = []
            epoch_final_balance_train_list = []
            epoch_required_money_train_list = []
            epoch_reward_sum_train_list = []

        # Save the best model to self.model_path (seed-specific directory)
        if best_model_state_dict is not None:
            best_model_path = os.path.join(self.model_path, "best_model.pkl")
            torch.save(best_model_state_dict, best_model_path)
            print(f"Best model saved to {best_model_path}")
        else:
            print("No best model was saved (possibly no improvement over initial).")


    def val_cluster(self, epoch_path, save_path, initial_action):
        self.eval_net.load_state_dict(
            torch.load(os.path.join(epoch_path, "trained_model.pkl"), map_location=self.device)
        )
        self.eval_net.eval()
        
        # Save current epsilon and set to 0 for deterministic validation
        original_epsilon = self.epsilon
        self.epsilon = 0.0
        
        df_list = self.val_index[self.label]
        df_number = int(len(df_list))
        action_list = []
        reward_list = []
        final_balance_list = []
        required_money_list = []
        commission_fee_list = []
        return_rate_list = []
        for i in range(df_number):
            print("validating on df", df_list[i])
            self.df = pd.read_feather(
                os.path.join(self.val_data_path, "df_{}.feather".format(df_list[i]))
            )

            val_env = Testing_Env(
                df=self.df,
                tech_indicator_list=self.tech_indicator_list,
                tech_indicator_list_trend=self.tech_indicator_list_trend,
                transcation_cost=self.transcation_cost,
                back_time_length=self.back_time_length,
                max_holding_number=self.max_holding_number,
                initial_action=initial_action,
            )
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
            portfit_magine, final_balance, required_money, commission_fee = (
                val_env.get_final_return_rate(slient=True)
            )
            final_balance = val_env.final_balance
            required_money = val_env.required_money
            action_list.append(action_list_episode)
            reward_list.append(reward_list_episode)
            final_balance_list.append(final_balance)
            required_money_list.append(required_money)
            commission_fee_list.append(commission_fee)
            return_rate_list.append(final_balance / required_money if required_money != 0 else 0)
        action_list = np.array(action_list, dtype=object)
        reward_list = np.array(reward_list, dtype=object)
        final_balance_list = np.array(final_balance_list)
        required_money_list = np.array(required_money_list)
        commission_fee_list = np.array(commission_fee_list)
        np.save(
            os.path.join(save_path, "action_val_{}.npy".format(initial_action)),
            action_list,
        )
        np.save(
            os.path.join(save_path, "reward_val_{}.npy".format(initial_action)),
            reward_list,
        )
        np.save(
            os.path.join(save_path, "final_balance_val_{}.npy".format(initial_action)),
            final_balance_list,
        )
        np.save(
            os.path.join(save_path, "require_money_val_{}.npy".format(initial_action)),
            required_money_list,
        )
        np.save(
            os.path.join(
                save_path, "commission_fee_history_val_{}.npy".format(initial_action)
            ),
            commission_fee_list,
        )
        # Handle division by zero for return_rate_mean
        valid_indices = required_money_list != 0
        if np.any(valid_indices):
            return_rate_mean = np.nan_to_num(
                final_balance_list[valid_indices] / required_money_list[valid_indices]
            ).mean()
        else:
            return_rate_mean = 0.0

        np.save(
            os.path.join(
                save_path, "return_rate_mean_val_{}.npy".format(initial_action)
            ),
            return_rate_mean,
        )
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        return np.mean(return_rate_list) if return_rate_list else 0


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    agent = DQN(args)
    agent.train()
