import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

# Path to images.txt
images_txt_path = 'data/rotation_no_occlusion_converted/sparse/0/images.txt'
# images_txt_path = 'data/mustard/sparse/0/images.txt'

# Read images.txt
camera_poses = []
with open(images_txt_path, 'r') as f:
    lines = f.readlines()
    idx = 0  # Initialize index
    while idx < len(lines):
        line = lines[idx]
        if line.startswith('#') or line.strip() == '':
            idx += 1
            continue
        parts = line.strip().split()
        if len(parts) < 10:
            idx += 1
            continue  # Skip lines that don't contain pose data
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])

        # Convert quaternion to rotation matrix
        rot = R.from_quat([qx, qy, qz, qw])  # scipy expects [x, y, z, w]
        R_world_to_cam = rot.as_matrix()

        # Compute camera center in world coordinates
        t_world_to_cam = np.array([tx, ty, tz]).reshape(3, 1)
        R_cam_to_world = R_world_to_cam.T
        camera_center = -R_cam_to_world @ t_world_to_cam
        camera_center = camera_center.flatten()

        # Create a coordinate frame representing the camera pose
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Apply rotation and translation to the coordinate frame
        cam_frame.rotate(R_cam_to_world, center=(0, 0, 0))
        cam_frame.translate(camera_center)

        camera_poses.append(cam_frame)

        idx += 2  # Skip the next line (empty line after each image in images.txt)

# Visualize camera poses
o3d.visualization.draw_geometries(camera_poses)
