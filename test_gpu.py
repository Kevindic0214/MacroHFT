import torch

print("Is GPU available:", torch.cuda.is_available())

print("Number of available GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i} name:", torch.cuda.get_device_name(i))
