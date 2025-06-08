# Surface Matching using DGCNN with Automated Point Cloud Labeling

## Overview
This project combines a Halcon-based labeling pipeline with a DGCNN architecture to perform surface matching in 3D point clouds. It automates label generation and trains a deep network for segmentation and pose estimation.

## Setup
```bash
conda env create -f environment.yml
conda activate OEM

python src/train.py --train_files data/*.ply --epochs 20 --batch_size 8