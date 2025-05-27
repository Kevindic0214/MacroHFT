import random

import numpy as np
import torch

# SumTree
# 資料來源: https://github.com/rlcode/per/blob/master/SumTree.py
class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for N priorities
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data_indices = np.zeros(capacity, dtype=object)  # Store data_index of experience for update
        # [--------------data indices-------------]
        #             size: capacity

    # 當葉節點的優先級更新時，向上傳播這個變化
    def _propagate(self, tree_index, change):
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # 更新葉節點的優先級
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    # 存儲優先級和數據索引
    def add(self, priority, data_idx):
        tree_index = self.data_pointer + self.capacity - 1
        self.data_indices[self.data_pointer] = data_idx # Store data_idx in leaf
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed capacity
            self.data_pointer = 0

    # 獲取葉節點信息
    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index_in_leaf = self.data_indices[leaf_index - self.capacity + 1]
        return leaf_index, self.tree[leaf_index], data_index_in_leaf

    def total_priority(self):
        return self.tree[0]  # Returns the root node


class ReplayBuffer(object):
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # Hyperparameter related to IS weights
    PER_b_increment_per_sampling = 0.001
    abs_err_upper = 1.  # Clipped abs error

    def __init__(self, args, state_dim, state_dim_2, action_dim):
        self.tree = SumTree(args.buffer_size)
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_size
        self.seed = args.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.buffer = {"state": np.zeros((self.buffer_capacity, state_dim)),
                        "state_trend": np.zeros((self.buffer_capacity, state_dim_2)),
                        "previous_action": np.zeros((self.buffer_capacity)),
                        "teacher_q_values": np.zeros((self.buffer_capacity, action_dim)),
                        "action": np.zeros((self.buffer_capacity, 1)),
                        "reward": np.zeros(self.buffer_capacity),
                        "next_state": np.zeros((self.buffer_capacity, state_dim)),
                        "next_state_trend": np.zeros((self.buffer_capacity, state_dim_2)),
                        "next_previous_action": np.zeros((self.buffer_capacity)),
                        "next_teacher_q_values": np.zeros((self.buffer_capacity, action_dim)),
                        "terminal": np.zeros(self.buffer_capacity),
                       }
        self.max_priority = 1.0 # Initialize max_priority
        self.buffer_write_idx = 0 # Pointer for writing to self.buffer, managed by ReplayBuffer

    def store_transition(self, state, state_trend, previous_action, teacher_q_values, action, reward, next_state, next_state_trend, next_previous_action, next_teacher_q_values, terminal):
        # Use self.buffer_write_idx to store data in self.buffer
        # This idx is what SumTree's self.data_indices will store internally for its leaves
        current_buffer_idx = self.buffer_write_idx

        self.buffer["state"][current_buffer_idx] = state
        self.buffer["state_trend"][current_buffer_idx] = state_trend
        self.buffer["previous_action"][current_buffer_idx] = previous_action
        self.buffer["teacher_q_values"][current_buffer_idx] = teacher_q_values
        self.buffer["action"][current_buffer_idx] = action
        self.buffer["reward"][current_buffer_idx] = reward
        self.buffer["next_state"][current_buffer_idx] = next_state
        self.buffer["next_state_trend"][current_buffer_idx] = next_state_trend
        self.buffer["next_previous_action"][
            current_buffer_idx] = next_previous_action
        self.buffer["next_teacher_q_values"][current_buffer_idx] = next_teacher_q_values
        self.buffer["terminal"][current_buffer_idx] = terminal
        
        # Add to SumTree with max_priority
        self.tree.add(self.max_priority, current_buffer_idx) 

        self.buffer_write_idx = (self.buffer_write_idx + 1) % self.buffer_capacity
        # self.current_size = min(self.current_size + 1, self.buffer_capacity) # SumTree handles effective size

    def sample(self):
        batch = {}
        tree_indices = np.empty((self.batch_size,), dtype=np.int32)
        IS_weights = np.empty((self.batch_size, 1), dtype=np.float32)

        # Ensure SumTree has enough entries, otherwise sample from what's available
        # SumTree's data_pointer indicates how many items have been added (and potentially overwritten)
        # For sampling, we consider the number of unique items currently in the tree for priority calculation.
        # The SumTree internally handles the circular buffer aspect for its own capacity.
        # The number of filled leaves in SumTree is self.tree.data_pointer if it hasn't wrapped around yet,
        # or self.tree.capacity if it has wrapped around and is full.
        # However, total_priority() gives the sum of priorities of all *active* leaves.
        
        num_entries_in_tree = self.tree.capacity if self.tree.tree[0] > 0 and self.tree.data_pointer == 0 and self.tree.tree[self.tree.capacity -1 + (self.tree.capacity-1)] !=0 else self.tree.data_pointer
        if self.tree.data_pointer == 0 and self.tree.tree[0] > 0 : # full and wrapped
            num_entries_in_tree = self.tree.capacity
        else:
            num_entries_in_tree = self.tree.data_pointer
        if num_entries_in_tree == 0: # Tree is empty
            # This case should ideally not be hit if training starts after buffer has some elements
            return None, None, None 

        priority_segment = self.tree.total_priority() / self.batch_size  # priority segment

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        # Use a more robust way to estimate max_weight if possible, or sample a bit to find min_prob
        # For simplicity, if tree is not full, this might not be the true max_weight
        current_min_priority = np.min(self.tree.tree[-self.tree.capacity:]) 
        if current_min_priority == 0: # Avoid division by zero if some priorities are still 0
            # This might happen if tree is not full and some leaves are 0
            # Fallback: use a very small probability estimate, or sample a few and find min non-zero
            # A robust way is to iterate through non-zero priorities if tree is not full.
            # For now, if min_priority is 0 and tree is not full, IS might be skewed.
            # A simple fix is to only calculate min_prob for non-zero priorities.
            non_zero_priorities = self.tree.tree[-self.tree.capacity:][self.tree.tree[-self.tree.capacity:] > 0]
            if len(non_zero_priorities) > 0:
                current_min_priority = np.min(non_zero_priorities)
            else: # All priorities are zero (e.g. empty or just initialized tree)
                 current_min_priority = self.PER_e # Use a small default if all are zero
        
        min_prob = current_min_priority / self.tree.total_priority()
        max_weight = (min_prob * num_entries_in_tree) ** (-self.PER_b) # num_entries_in_tree approximates N here
        if max_weight == float('inf'): max_weight = 1.0 # safety check
        if max_weight == 0 : max_weight = 1.0 # safety check


        for i in range(self.batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            v = np.random.uniform(a, b)
            tree_idx, p, data_idx = self.tree.get_leaf(v)
            
            prob = p / self.tree.total_priority()
            IS_weights[i, 0] = (prob * num_entries_in_tree) ** (-self.PER_b) # num_entries_in_tree approximates N 
            
            tree_indices[i] = tree_idx
            # batch[i] = self.buffer[data_idx] -> No, self.buffer is a dict
            for key in self.buffer.keys():
                if i == 0: # Initialize lists in batch dict for the first sample
                    batch[key] = []
                batch[key].append(self.buffer[key][data_idx])

        # Convert lists in batch to numpy arrays first, then to tensors
        for key in self.buffer.keys():
            batch[key] = np.array(batch[key])
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        
        IS_weights /= max_weight # Normalize weights

        return batch, tree_indices, IS_weights

    def update_priorities(self, tree_indices, abs_td_errors):
        abs_td_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_td_errors, self.abs_err_upper) # Clip errors
        priorities = np.power(clipped_errors, self.PER_a) # Calculate new priorities P = (TD_error + epsilon)^alpha

        for ti, p in zip(tree_indices, priorities):
            self.tree.update(ti, p)
            if p > self.max_priority and ti >= (self.tree.capacity -1): # Update max_priority only from leaf updates
                 self.max_priority = p


class ReplayBuffer_High(object):
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # Hyperparameter related to IS weights
    PER_b_increment_per_sampling = 0.001
    abs_err_upper = 1.  # Clipped abs error

    def __init__(self, args, state_dim, state_dim_2, action_dim):
        self.tree = SumTree(args.buffer_size)
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_size
        self.seed = args.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.buffer = {"state": np.zeros((self.buffer_capacity, state_dim)),
                        "state_trend": np.zeros((self.buffer_capacity, state_dim_2)),
                        "state_clf": np.zeros((self.buffer_capacity, 2)),
                        "previous_action": np.zeros((self.buffer_capacity)),
                        "teacher_q_values": np.zeros((self.buffer_capacity, action_dim)),
                        "action": np.zeros((self.buffer_capacity, 1)),
                        "reward": np.zeros(self.buffer_capacity),
                        "next_state": np.zeros((self.buffer_capacity, state_dim)),
                        "next_state_trend": np.zeros((self.buffer_capacity, state_dim_2)),
                        "next_state_clf": np.zeros((self.buffer_capacity, 2)),
                        "next_previous_action": np.zeros((self.buffer_capacity)),
                        "next_teacher_q_values": np.zeros((self.buffer_capacity, action_dim)),
                        "terminal": np.zeros(self.buffer_capacity),
                        "q_memory": np.zeros(self.buffer_capacity),
                       }
        self.max_priority = 1.0
        self.buffer_write_idx = 0

    def store_transition(self, state, state_trend, state_clf, previous_action, teacher_q_values, action, reward, 
                                next_state, next_state_trend, next_state_clf, next_previous_action, next_teacher_q_values, terminal, q_memory):
        current_buffer_idx = self.buffer_write_idx

        self.buffer["state"][current_buffer_idx] = state
        self.buffer["state_trend"][current_buffer_idx] = state_trend
        self.buffer["state_clf"][current_buffer_idx] = state_clf
        self.buffer["previous_action"][current_buffer_idx] = previous_action
        self.buffer["teacher_q_values"][current_buffer_idx] = teacher_q_values
        self.buffer["action"][current_buffer_idx] = action
        self.buffer["reward"][current_buffer_idx] = reward
        self.buffer["next_state"][current_buffer_idx] = next_state
        self.buffer["next_state_trend"][current_buffer_idx] = next_state_trend
        self.buffer["next_state_clf"][current_buffer_idx] = next_state_clf
        self.buffer["next_previous_action"][
            current_buffer_idx] = next_previous_action
        self.buffer["next_teacher_q_values"][current_buffer_idx] = next_teacher_q_values
        self.buffer["terminal"][current_buffer_idx] = terminal
        self.buffer["q_memory"][current_buffer_idx] = float(q_memory[0]) if (hasattr(q_memory, "__len__") and len(q_memory) > 0) else float(q_memory)
        
        self.tree.add(self.max_priority, current_buffer_idx)
        self.buffer_write_idx = (self.buffer_write_idx + 1) % self.buffer_capacity
        # self.current_size = min(self.current_size + 1, self.buffer_capacity)

    def sample(self):
        batch = {}
        tree_indices = np.empty((self.batch_size,), dtype=np.int32)
        IS_weights = np.empty((self.batch_size, 1), dtype=np.float32)

        num_entries_in_tree = self.tree.capacity if self.tree.tree[0] > 0 and self.tree.data_pointer == 0 and self.tree.tree[self.tree.capacity -1 + (self.tree.capacity-1)] !=0 else self.tree.data_pointer
        if self.tree.data_pointer == 0 and self.tree.tree[0] > 0: # full and wrapped
            num_entries_in_tree = self.tree.capacity
        else:
            num_entries_in_tree = self.tree.data_pointer
        
        if num_entries_in_tree == 0:
            return None, None, None 

        priority_segment = self.tree.total_priority() / self.batch_size
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        current_min_priority = np.min(self.tree.tree[-self.tree.capacity:]) 
        if current_min_priority == 0:
            non_zero_priorities = self.tree.tree[-self.tree.capacity:][self.tree.tree[-self.tree.capacity:] > 0]
            if len(non_zero_priorities) > 0:
                current_min_priority = np.min(non_zero_priorities)
            else:
                 current_min_priority = self.PER_e 
        
        min_prob = current_min_priority / self.tree.total_priority()
        if self.tree.total_priority() == 0 : # Avoid division by zero if tree is empty or all priorities are zero
            min_prob = 0 # effectively makes max_weight large, or could be handled to skip sampling
            # This case should be rare if PER_e > 0 and new samples get max_priority > 0
            # If total_priority is 0, priority_segment will also be 0, leading to issues in v = np.random.uniform(a,b)
            # A simple fallback: if total_priority is 0, return None, None, None or sample uniformly (but that defeats PER)
            # For now, this implies an issue if hit, as sampling from a zero-total-priority tree is problematic.
            # A better check might be at the start of sample().
            if num_entries_in_tree > 0 : # if tree has entries but total_priority is 0 (e.g. all priorities decayed to 0 by error)
                 pass # let it proceed, min_prob will be 0, max_weight might be inf
            else: # No entries, already handled by num_entries_in_tree == 0 check
                 pass 
        
        max_weight = (min_prob * num_entries_in_tree) ** (-self.PER_b)
        if max_weight == float('inf'): max_weight = 100.0 # Increased safety cap from 1.0
        if max_weight == 0 : max_weight = 1.0 

        for i in range(self.batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            # Ensure a and b are valid for uniform sampling, esp. if priority_segment is 0
            if a == b: # This can happen if total_priority is 0 or very small
                v = a
            else:
                v = np.random.uniform(a, b)
            tree_idx, p, data_idx = self.tree.get_leaf(v)
            
            prob = p / self.tree.total_priority() if self.tree.total_priority() > 0 else 0
            if prob == 0 and num_entries_in_tree > 0: # Avoid (0 * N)^-beta if prob is 0 but tree has entries
                 IS_weights[i,0] = 0 # or some other indicator of an issue / very low weight
            else:
                 IS_weights[i, 0] = (prob * num_entries_in_tree) ** (-self.PER_b) if num_entries_in_tree > 0 else 0

            tree_indices[i] = tree_idx
            for key in self.buffer.keys():
                if i == 0:
                    batch[key] = []
                batch[key].append(self.buffer[key][data_idx])

        for key in self.buffer.keys():
            batch[key] = np.array(batch[key])
            if key in ["action", "previous_action", "next_previous_action"]:
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        
        if max_weight > 0 : IS_weights /= max_weight # Normalize only if max_weight is positive

        return batch, tree_indices, IS_weights

    def update_priorities(self, tree_indices, abs_td_errors):
        abs_td_errors += self.PER_e
        clipped_errors = np.minimum(abs_td_errors, self.abs_err_upper)
        priorities = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_indices, priorities):
            self.tree.update(ti, p)
            if p > self.max_priority and ti >= (self.tree.capacity -1):
                 self.max_priority = p