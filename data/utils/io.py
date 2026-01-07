import os
import torch


def save_pt(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Move tensors in meta to cpu for portability if desired, but we keep as-is for speed.
    torch.save(obj, path)
