'''from plyfile import PlyData

ply = PlyData.read("/Users/amin/PycharmProject/OEM/data/synth_labeled/sample_0000.ply")
print("Fields:", ply['vertex'].data.dtype.names)
'''

# src/test.py

import os
import numpy as np
from plyfile import PlyData
import torch
from torch.utils.data import DataLoader
from dataset import PointCloudDataset

# Absolute path to a sample PLY (update as needed)
PLY_PATH = "/Users/amin/PycharmProject/OEM/data/synth_labeled/sample_0000.ply"

def test_ply_header(ply_path):
    ply = PlyData.read(ply_path)
    fields = ply['vertex'].data.dtype.names
    print("PLY header fields:", fields)

def test_segmentation_loader(ply_path, num_points=2048):
    # 1) Instantiate dataset and DataLoader
    ds = PointCloudDataset([ply_path], num_points=num_points)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # 2) Fetch one sample
    points, labels = next(iter(loader))

    # 3) Print shapes and label statistics
    print(f"Loaded points tensor shape: {points.shape}")   # (1, N, 3)
    print(f"Loaded labels tensor shape: {labels.shape}")   # (1, N)
    unique = torch.unique(labels)
    print(f"Unique labels in this sample: {unique.numpy()}")

if __name__ == "__main__":
    # Ensure script runs from project root
    print("Current working dir:", os.getcwd())
    print("\n--- Testing PLY header ---")
    test_ply_header(PLY_PATH)
    print("\n--- Testing PointCloudDataset loader ---")
    test_segmentation_loader(PLY_PATH)
