import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement

def sample_pose(trans_min, trans_max):
    quat = R.random().as_quat()  # (x,y,z,w)
    t = np.random.uniform(trans_min, trans_max)
    return quat, t

def apply_pose(pts, quat, t):
    Rm = R.from_quat(quat).as_matrix()
    return (Rm @ pts.T).T + t

def main(args):
    # 1. Load the template PLY (with labels)
    ply = PlyData.read(args.template)
    xyz   = np.vstack([ply['vertex'].data['x'],
                       ply['vertex'].data['y'],
                       ply['vertex'].data['z']]).T       # (N,3)
    labels= np.array(ply['vertex'].data[args.label_field], dtype=np.uint8)  # (N,)

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.num_samples):
        quat, t = sample_pose(args.trans_min, args.trans_max)
        pts_t = apply_pose(xyz, quat, t)

        # 2. Build a structured array for plyfile
        vertex_data = np.zeros(pts_t.shape[0],
                               dtype=[('x','f4'),('y','f4'),('z','f4'),('label','u1')])
        vertex_data['x']      = pts_t[:,0]
        vertex_data['y']      = pts_t[:,1]
        vertex_data['z']      = pts_t[:,2]
        vertex_data['label']  = labels

        el = PlyElement.describe(vertex_data, 'vertex')
        out_ply = os.path.join(args.out_dir, f"sample_{i:04d}.ply")
        PlyData([el], text=args.ascii).write(out_ply)

        # 3. Save pose metadata
        out_npz = os.path.join(args.out_dir, f"sample_{i:04d}.npz")
        np.savez(out_npz, quaternion=quat, translation=t)

        if i % 50 == 0:
            print(f"Generated {i}/{args.num_samples}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--template",    type=str,       required=True,
                   help="Path to .ply with a label field")
    p.add_argument("--label_field", type=str,       default="label",
                   help="Name of the label property in the PLY header")
    p.add_argument("--out_dir",     type=str,       required=True,
                   help="Where to write new PLY+NPZ files")
    p.add_argument("--num_samples", type=int,       default=1000)
    p.add_argument("--trans_min",   type=float, nargs=3, default=[-0.1,-0.1,-0.1])
    p.add_argument("--trans_max",   type=float, nargs=3, default=[ 0.1, 0.1, 0.1])
    p.add_argument("--ascii",       action="store_true",
                   help="Write ASCII PLY (for easy inspection) otherwise binary")
    args = p.parse_args()
    main(args)
