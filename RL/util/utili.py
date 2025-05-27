def get_ada(ada, decay_freq=2, ada_counter=0, decay_coffient=0.5):
    """
    衰減學習率參數
    
    Args:
        ada: 當前學習率參數
        decay_freq: 衰減頻率，每隔多少次衰減一次
        ada_counter: 計數器，記錄當前迭代次數
        decay_coffient: 衰減係數，決定每次衰減的程度
        
    Returns:
        更新後的學習率參數
    """
    if ada_counter % decay_freq == 1:
        ada = decay_coffient * ada
    return ada

def get_epsilon(epsilon, max_epsilon=1, epsilon_counter=0, decay_freq=2, decay_coffient=0.5):
    """
    更新探索率參數
    
    Args:
        epsilon: 當前探索率
        max_epsilon: 最大探索率
        epsilon_counter: 計數器，記錄當前迭代次數
        decay_freq: 更新頻率，每隔多少次更新一次
        decay_coffient: 更新係數，決定每次更新的程度
        
    Returns:
        更新後的探索率
    """
    if epsilon_counter % decay_freq == 1:
        epsilon = epsilon + (max_epsilon - epsilon) * decay_coffient
    return epsilon

class LinearDecaySchedule(object):
    """
    線性衰減調度器，用於隨時間衰減探索率
    """
    def __init__(self, start_epsilon, end_epsilon, decay_length):
        """
        初始化線性衰減調度器
        
        Args:
            start_epsilon: 起始探索率
            end_epsilon: 最終探索率
            decay_length: 完成衰減所需的步數
        """
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_length = decay_length

    def get_epsilon(self, t):
        """
        根據當前步數計算探索率
        
        Args:
            t: 當前步數
            
        Returns:
            當前應使用的探索率值
        """
        return max(self.end_epsilon, self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (t / self.decay_length))
