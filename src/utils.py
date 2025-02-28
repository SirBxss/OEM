# src/utils.py

import torch

def compute_accuracy(outputs, labels):
    """
    outputs: (B, N, num_classes) logits
    labels:  (B, N) ground truth labels
    returns: accuracy (float)
    """
    preds = torch.argmax(outputs, dim=-1)
    correct = (preds == labels).float().sum()
    total = labels.numel()
    return (correct / total).item()

def random_jitter(points, sigma=0.01, clip=0.05):
    """
    Randomly jitter points by adding Gaussian noise.
    points: (B, N, 3)
    sigma: Standard deviation of Gaussian noise
    clip: Clipping range for noise
    """
    noise = torch.clamp(sigma * torch.randn_like(points), -clip, clip)
    return points + noise
