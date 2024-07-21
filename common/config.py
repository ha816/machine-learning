import os

import pandas as pd
import torch


def get_torch_gpu_device_if_available():
    if torch.cuda.is_available():
        print(f'Torch cuda is available. Use GPU.')
        return torch.device("cuda")
    else:
        print(f'Torch cuda is unavailable. Use CPU instead...')
        return torch.device("cpu")
