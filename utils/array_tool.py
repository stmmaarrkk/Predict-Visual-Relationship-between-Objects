import torch
import numpy as np

gpu_dev = 'cpu'  # Default gpu


def to_np(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch._C._TensorBase):
        return data.data.cpu().numpy()


def to_tensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, torch._C._TensorBase):
        tensor = data.clone().detach()
        #tensor = torch.tensor(data.data)
    if cuda:
        tensor = tensor.to(device=gpu_dev)
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, torch._C._TensorBase):
        return data.item()
