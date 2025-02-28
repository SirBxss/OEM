# src/dataset.py
import torch
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, ply_file_paths, num_points=2048, transform=None):
        """
        :param ply_file_paths: List of paths to .ply files.
        :param num_points: Fixed number of points to sample from each point cloud.
        :param transform: Optional transformations or augmentations.
        """
        self.files = ply_file_paths
        self.num_points = num_points
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

        # Extract labels (ensure the property name matches your PLY file)
        labels = np.array(plydata['vertex'].data['my_labels'])

        # Sample a fixed number of points
        N = points.shape[0]
        if N >= self.num_points:
            # Sample without replacement if there are enough points
            indices = np.random.choice(N, self.num_points, replace=False)
        else:
            # Sample with replacement if not enough points are available
            indices = np.random.choice(N, self.num_points, replace=True)

        points = points[indices]
        labels = labels[indices]

        # Convert to PyTorch tensors
        points_tensor = torch.tensor(points, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        if self.transform:
            points_tensor, labels_tensor = self.transform(points_tensor, labels_tensor)

        return points_tensor, labels_tensor
