import torch


def mps_available():
    # torch.backends.mps only exists on torch >= 1.12
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()

def resolve_device(requested="cuda:0"):
    """Resolve the training device, falling back when the request is unavailable.

    Honours an explicit request whenever its backend is present, so the
    cuda:N arguments hardcoded in scripts/ keep working on a CUDA host and
    --device mps is available to anyone who wants to try it.

    The automatic fallback is cuda -> cpu, deliberately skipping mps: this
    agent selects actions one state at a time, and a batch-1 forward pass is
    ~15x slower on mps than on cpu: the sub-agent is a 45k-parameter MLP, so
    kernel dispatch dwarfs the actual math. Measured end-to-end over 4000 env
    steps of low_level training on an M-series machine, cpu 3.4s vs mps 6.0s.
    """
    requested = str(requested)
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.startswith("mps") and mps_available():
        return torch.device("mps")
    if requested.startswith("cpu"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def get_ada(ada,decay_freq=2,ada_counter=0, decay_coffient=0.5):
    if ada_counter % decay_freq==1:
        ada = decay_coffient*ada
    return ada

def get_epsilon( epsilon,max_epsilon=1, epsilon_counter=0, decay_freq=2,decay_coffient=0.5):
    if epsilon_counter%decay_freq == 1:
        epsilon =epsilon+(max_epsilon-epsilon)*decay_coffient
    return epsilon

class LinearDecaySchedule(object):
    def __init__(self, start_epsilon, end_epsilon, decay_length):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_length = decay_length

    def get_epsilon(self, t):
        return max(self.end_epsilon, self.start_epsilon - (self.start_epsilon - self.end_epsilon) * (t / self.decay_length))
