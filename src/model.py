import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    # x: (B, C, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x, k=20):
    # x: (B, C, N)
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k=k)  # (B, N, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # Concatenate (neighbor - central, central)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (B, 2*num_dims, N, k)
    return feature


class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, num_classes=2):
        super(DGCNN, self).__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Fully connected layers for feature aggregation
        # After concatenation, channels: 64 + 64 + 128 + 256 = 512
        self.linear1 = nn.Linear(512, emb_dims)
        self.bn1 = nn.BatchNorm1d(emb_dims)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(emb_dims, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, N, 3) --> transpose to (B, 3, N)
        batch_size = x.size(0)
        x = x.transpose(2, 1)

        # First EdgeConv layer
        x1 = get_graph_feature(x, k=self.k)  # (B, 6, N, k)
        x1 = self.conv1(x1)  # (B, 64, N, k)
        x1 = x1.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # Second EdgeConv layer
        x2 = get_graph_feature(x1, k=self.k)  # (B, 64*2, N, k)
        x2 = self.conv2(x2)  # (B, 64, N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        # Third EdgeConv layer
        x3 = get_graph_feature(x2, k=self.k)  # (B, 64*2, N, k)
        x3 = self.conv3(x3)  # (B, 128, N, k)
        x3 = x3.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        # Fourth EdgeConv layer
        x4 = get_graph_feature(x3, k=self.k)  # (B, 128*2, N, k)
        x4 = self.conv4(x4)  # (B, 256, N, k)
        x4 = x4.max(dim=-1, keepdim=False)[0]  # (B, 256, N)

        # Concatenate features: (B, 64+64+128+256, N) = (B, 512, N)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        # Transpose to (B, N, 512) for the fully connected layers
        x_cat = x_cat.transpose(2, 1)

        # First fully connected layer
        x_global = self.linear1(x_cat)  # (B, N, emb_dims)
        # Transpose to (B, emb_dims, N) for BatchNorm1d
        x_global = x_global.transpose(1, 2)
        x_global = self.bn1(x_global)
        x_global = F.relu(x_global)
        x_global = self.dp1(x_global)
        # Transpose back to (B, N, emb_dims)
        x_global = x_global.transpose(1, 2)

        # Second fully connected layer
        x_global = self.linear2(x_global)  # (B, N, 256)
        x_global = x_global.transpose(1, 2)
        x_global = self.bn2(x_global)
        x_global = F.relu(x_global)
        x_global = self.dp2(x_global)
        x_global = x_global.transpose(1, 2)  # (B, N, 256)

        # Final classification layer (per point)
        x_out = self.linear3(x_global)  # (B, N, num_classes)
        return x_out
