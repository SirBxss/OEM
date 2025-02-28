# src/train.py

import torch
import argparse
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
from model import DGCNN
import torch.nn.functional as F


def train(args):
    # 1. Load Dataset
    train_dataset = PointCloudDataset(args.train_files)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Initialize Model, Loss, Optimizer
    model = DGCNN(k=args.k, num_classes=args.num_classes).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    """for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for points, labels in train_loader:
            points, labels = points.to(args.device), labels.to(args.device)
            optimizer.zero_grad()

            outputs = model(points)  # (B, N, num_classes)
            loss = criterion(outputs.view(-1, args.num_classes), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {avg_loss:.4f}")"""

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for points, labels in train_loader:
            points, labels = points.to(args.device), labels.to(args.device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(points)  # shape: (B, N, num_classes)
            # Print output details for debugging
            print("Outputs shape:", outputs.shape)

            # Compute softmax probabilities and print their mean per sample
            softmax_out = torch.softmax(outputs, dim=-1)  # shape: (B, N, num_classes)
            # Average probability across points for each sample (should be near 0.5 for each class if not learning)
            print("Mean softmax per sample:", softmax_out.mean(dim=1))

            # Compute loss
            loss = criterion(outputs.view(-1, args.num_classes), labels.view(-1))
            print("Loss before backward:", loss.item())

            # Backward pass
            loss.backward()

            # Check gradient norms for each parameter
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"Gradient norm for {name}: {grad_norm}")
                else:
                    print(f"No gradient for {name}")

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_files", nargs="+", required=True,
                        help="List of .ply files for training.")
    parser.add_argument("--k", type=int, default=20, help="Number of neighbors for DGCNN.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of segmentation classes.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu).")

    args = parser.parse_args()
    train(args)
