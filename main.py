# main.py

import argparse
import os

def train_segmentation():
    print("[INFO] Starting segmentation training...")
    os.system("python src/train.py --train_files data/raw/*.ply --epochs 20 --batch_size 8")

def train_pose():
    print("[INFO] Starting pose estimation training...")
    os.system("python src/train_pose.py --train_files data/synth_labeled/*.ply --epochs 30 --batch_size 4")

def run_test():
    print("[INFO] Running test script...")
    os.system("python src/test.py")

def generate_synthetic():
    print("[INFO] Generating synthetic data...")
    os.system("python src/synthetic_data_gen_with_labels.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface Matching Project Entry Point")
    parser.add_argument('--task', type=str, required=True, choices=['train_seg', 'train_pose', 'test', 'generate'],
                        help="Which task to run: train_seg, train_pose, test, generate")
    args = parser.parse_args()

    if args.task == 'train_seg':
        train_segmentation()
    elif args.task == 'train_pose':
        train_pose()
    elif args.task == 'test':
        run_test()
    elif args.task == 'generate':
        generate_synthetic()
