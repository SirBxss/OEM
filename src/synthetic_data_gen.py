import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def sample_pose(translation_range):
    # Random rotation as unit quaternion (x,y,z,w)
    quat = R.random().as_quat()
    # Random translation within the given 3-D box
    t = np.random.uniform(translation_range[0], translation_range[1])
    return quat, t

def apply_pose(points: np.ndarray, quat: np.ndarray, t: np.ndarray):
    # quat: (4,) in (x,y,z,w) format
    R_mat = R.from_quat(quat).as_matrix()  # (3,3)
    # rotate then translate
    return (R_mat @ points.T).T + t

def main(args):
    # 1) load template
    pcd = o3d.io.read_point_cloud(args.template)
    pts = np.asarray(pcd.points)  # (N,3)

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.num_samples):
        quat, t = sample_pose((args.trans_min, args.trans_max))
        pts_t = apply_pose(pts, quat, t)

        # 2) save transformed point cloud
        out_ply = os.path.join(args.out_dir, f"sample_{i:04d}.ply")
        pcd_new = o3d.geometry.PointCloud()
        pcd_new.points = o3d.utility.Vector3dVector(pts_t)
        #o3d.io.write_point_cloud(out_ply, pcd_new)
        o3d.io.write_point_cloud(out_ply, pcd_new, write_ascii=True)

        # 3) save pose metadata as .npz
        out_npz = os.path.join(args.out_dir, f"sample_{i:04d}.npz")
        np.savez(out_npz, quaternion=quat, translation=t)

        if i % 50 == 0:
            print(f"  • generated {i}/{args.num_samples}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template",    type=str, required=True,
                        help="Path to the template .ply")
    parser.add_argument("--out_dir",     type=str, required=True,
                        help="Folder to save synthetic samples")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="How many poses to generate")
    parser.add_argument("--trans_min",   type=float, nargs=3, default=[-0.1,-0.1,-0.1],
                        help="Min translation (x,y,z)")
    parser.add_argument("--trans_max",   type=float, nargs=3, default=[ 0.1, 0.1, 0.1],
                        help="Max translation (x,y,z)")
    args = parser.parse_args()
    main(args)
