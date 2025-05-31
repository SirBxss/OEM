# src/train_pose.py

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import PosePointCloudDataset
from model import DGCNNPose
from glob import glob

def rotation_loss(q_pred, q_gt):
    """
    Geodesic loss between two unit quaternions.
    q_pred, q_gt: tensors of shape (B,4)
    """
    # dot product per sample (B,)
    dot = torch.abs((q_pred * q_gt).sum(dim=1))
    dot = torch.clamp(dot, -1.0, 1.0)
    # angle = arccos(dot)
    return torch.acos(dot).mean()

def train(args):
    # 1) Build dataset + loader
    ply_paths = sorted(glob(f"{args.data_dir}/*.ply"))
    npz_paths = [p.replace(".ply", ".npz") for p in ply_paths]
    dataset = PosePointCloudDataset(
        ply_paths, npz_paths,
        num_points=args.num_points,
        transform=None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 2) Model, optimizer, DGCNN
    device = torch.device(args.device)
    model = DGCNNPose(
        k=args.k,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3) Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        total_r_loss = 0.0
        total_t_loss = 0.0

        for pts, q_gt, t_gt in loader:
            pts   = pts.to(device)    # (B, N, 3)
            q_gt  = q_gt.to(device)   # (B, 4)
            t_gt  = t_gt.to(device)   # (B, 3)

            optimizer.zero_grad()
            q_pred, t_pred = model(pts)  # (B,4), (B,3)

            # compute losses
            loss_r = rotation_loss(q_pred, q_gt)
            loss_t = F.mse_loss(t_pred, t_gt)
            loss   = args.lambda_r * loss_r + args.lambda_t * loss_t

            loss.backward()
            optimizer.step()

            total_r_loss += loss_r.item() * pts.size(0)
            total_t_loss += loss_t.item() * pts.size(0)

        avg_r = total_r_loss / len(dataset)
        avg_t = total_t_loss / len(dataset)
        print(f"Epoch [{epoch}/{args.epochs}]  "
              f"RotLoss: {avg_r:.4f}  TransLoss: {avg_t:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train DGCNNPose for 6-DoF regression")
    p.add_argument("--data_dir",    type=str, required=True,
                   help="Folder containing .ply + .npz synthetic samples")
    p.add_argument("--k",           type=int,   default=20,
                   help="Number of neighbors for EdgeConv")
    p.add_argument("--num_points",  type=int,   default=2048,
                   help="Number of points sampled per cloud")
    p.add_argument("--dropout",     type=float, default=0.5,
                   help="Dropout rate in pose MLP")
    p.add_argument("--batch_size",  type=int,   default=8,
                   help="Batch size for training")
    p.add_argument("--num_workers", type=int,   default=4,
                   help="DataLoader worker processes")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate for optimizer")
    p.add_argument("--epochs",      type=int,   default=100,
                   help="Number of training epochs")
    p.add_argument("--lambda_r",    type=float, default=1.0,
                   help="Weight for rotation loss")
    p.add_argument("--lambda_t",    type=float, default=1.0,
                   help="Weight for translation loss")
    p.add_argument("--device",      type=str,
                   default="mps" if torch.backends.mps.is_available() else "cpu",
                   help="Device (mps or cpu)")

    args = p.parse_args()
    train(args)
