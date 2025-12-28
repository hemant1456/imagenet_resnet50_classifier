import torch
import numpy as np
import random

def seed_everything(seed=1):
    # 1. Basic python and random module
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch CPU
    torch.manual_seed(seed)
    
    # 4. PyTorch GPU (all GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # 5. CUDNN Determinism (Crucial for ResNet)
    # This makes the convolution algorithms deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

