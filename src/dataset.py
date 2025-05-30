# src/dataset.py
'''import torch
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

        return points_tensor, labels_tensor '''


# src/dataset.py

import os
import torch
import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset
import open3d as o3d  # only for the pose dataset


class PointCloudDataset(Dataset):
    """
    Per-point classification dataset (plug vs background).
    """
    def __init__(self, ply_file_paths, num_points=2048, transform=None):
        """
        :param ply_file_paths: List of paths to .ply files.
        :param num_points: Fixed number of points to sample from each point cloud.
        :param transform: Optional (points, labels) -> (points, labels).
        """
        self.files = ply_file_paths
        self.num_points = num_points
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1) Read with plyfile
        plydata = PlyData.read(self.files[idx])
        x = plydata['vertex'].data['x']
        y = plydata['vertex'].data['y']
        z = plydata['vertex'].data['z']
        points = np.stack((x, y, z), axis=1)

        # 2) Labels (ensure this matches your header!)
        #labels = np.array(plydata['vertex'].data['my_labels'])
        labels = np.array(plydata['vertex'].data['label'])

        # 3) Sample or pad to fixed num_points
        N = points.shape[0]
        if N >= self.num_points:
            inds = np.random.choice(N, self.num_points, replace=False)
        else:
            inds = np.random.choice(N, self.num_points, replace=True)
        points = points[inds]
        labels = labels[inds]

        # 4) To tensors
        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()

        # 5) Optional transform
        if self.transform:
            points_tensor, labels_tensor = self.transform(points_tensor, labels_tensor)

        return points_tensor, labels_tensor


class PosePointCloudDataset(Dataset):
    """
    Global‐pose regression dataset.
    Each sample: (points, quaternion, translation).
    Expects for every .ply in ply_paths a matching .npz in pose_paths.
    """
    def __init__(self, ply_paths, pose_paths, num_points=2048, transform=None):
        assert len(ply_paths) == len(pose_paths), "PLY and NPZ lists must match!"
        self.ply_paths = ply_paths
        self.pose_paths = pose_paths
        self.num_points = num_points
        self.transform = transform

    def __len__(self):
        return len(self.ply_paths)

    def __getitem__(self, idx):
        # 1) Load the point cloud (Open3D for speed and simplicity)
        pcd = o3d.io.read_point_cloud(self.ply_paths[idx])
        pts = np.asarray(pcd.points, dtype=np.float32)

        # 2) Sample or pad to fixed num_points
        N = pts.shape[0]
        if N >= self.num_points:
            inds = np.random.choice(N, self.num_points, replace=False)
        else:
            inds = np.random.choice(N, self.num_points, replace=True)
        pts = pts[inds]

        # 3) Load ground‐truth pose
        data = np.load(self.pose_paths[idx])
        quat = data["quaternion"].astype(np.float32)   # (4,) x,y,z,w
        trans = data["translation"].astype(np.float32) # (3,)

        # 4) Optional point‐only transform (e.g. jitter, rotate)
        if self.transform:
            pts = self.transform(pts)

        # 5) To tensors
        pts_tensor   = torch.from_numpy(pts)   # (num_points,3)
        quat_tensor  = torch.from_numpy(quat)  # (4,)
        trans_tensor = torch.from_numpy(trans) # (3,)

        return pts_tensor, quat_tensor, trans_tensor
