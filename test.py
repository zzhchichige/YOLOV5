import torch
print(torch.__version__)      # 应该是 2.4.1+cu118
print(torch.cuda.is_available())  # 应该是 True，如果用 GPU
