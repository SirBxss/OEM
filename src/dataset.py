# src/dataset.py

import torch
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):
    def __init__(self, ply_file_paths, transform=None):
        """
        :param ply_file_paths: List of paths to .ply files.
        :param transform: Optional callable for data augmentation or preprocessing.
        """
        self.files = ply_file_paths
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        plydata = PlyData.read(file_path)

        # Extract x, y, z coordinates
        x = plydata['vertex'].data['x']
        y = plydata['vertex'].data['y']
        z = plydata['vertex'].data['z']
        points = np.stack((x, y, z), axis=1)

        # Extract labels (assumed in a 'label' field)
        labels = np.array(plydata['vertex'].data['label'])

        # Convert to tensors
        points_tensor = torch.tensor(points, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Apply optional transform (e.g., augmentation)
        if self.transform:
            points_tensor, labels_tensor = self.transform(points_tensor, labels_tensor)

        return points_tensor, labels_tensor
