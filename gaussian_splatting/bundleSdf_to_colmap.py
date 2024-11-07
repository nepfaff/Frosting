import numpy as np
import os
import glob
from PIL import Image
from scipy.spatial.transform import Rotation as R
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to COLMAP format.')
    parser.add_argument('--data_dir', required=True, help='Path to the root directory of your dataset.')
    parser.add_argument('--output_dir', default=None, help='Output directory for COLMAP files. Defaults to data_dir.')
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir

    # Read intrinsic matrix
    cam_K_path = os.path.join(data_dir, 'cam_K.txt')
    if not os.path.exists(cam_K_path):
        raise FileNotFoundError(f'Intrinsic matrix file not found: {cam_K_path}')
    cam_K = np.loadtxt(cam_K_path)

    # Extract parameters
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]

    camera_id = 1  # Since you have only one camera
    model = 'PINHOLE'
    params = [fx, fy, cx, cy]

    # Get list of image files
    image_dir = os.path.join(data_dir, 'images_dino_sampled')
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if not image_files:
        raise FileNotFoundError(f'No images found in directory: {image_dir}')

    # Get image size from the first image
    with Image.open(image_files[0]) as img:
        width, height = img.size

    # Create output directories
    sparse_dir = os.path.join(output_dir, 'sparse', '0')
    os.makedirs(sparse_dir, exist_ok=True)

    # Write cameras.txt
    cameras_txt_path = os.path.join(sparse_dir, 'cameras.txt')
    with open(cameras_txt_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'{camera_id} {model} {int(width)} {int(height)} {" ".join(map(str, params))}\n')

    # Write images.txt
    images_txt_path = os.path.join(sparse_dir, 'images.txt')
    with open(images_txt_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')

        for idx, img_path in enumerate(image_files):
            image_id = idx + 1
            image_name = os.path.basename(img_path)
            
            # Read camera-to-world transformation
            pose_idx = os.path.splitext(image_name)[0]  # Get '000000' from '000000.png'
            pose_path = os.path.join(data_dir, 'ob_in_cam', f'{pose_idx}.txt')
            if not os.path.exists(pose_path):
                raise FileNotFoundError(f'Pose file not found: {pose_path}')
            cam_to_world = np.loadtxt(pose_path)  # Shape (4, 4)

            # Invert to get world-to-camera transformation
            world_to_cam = np.linalg.inv(cam_to_world)

            # Extract rotation matrix and translation vector
            rotation_matrix = world_to_cam[:3, :3]
            translation_vector = world_to_cam[:3, 3]

            # Convert rotation matrix to quaternion
            rot = R.from_matrix(rotation_matrix)
            quat = rot.as_quat()  # Returns [qx, qy, qz, qw]
            qx, qy, qz, qw = quat
            # COLMAP expects [qw, qx, qy, qz]
            f.write(f'{image_id} {qw} {qx} {qy} {qz} {translation_vector[0]} {translation_vector[1]} {translation_vector[2]} {camera_id} {image_name}\n')

            # Write empty line for 2D points (since we have none)
            f.write('\n')

    # Write empty points3D.txt
    points3D_txt_path = os.path.join(sparse_dir, 'points3D.txt')
    with open(points3D_txt_path, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n')
        f.write('\n')

    # Copy images to images directory
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_output_dir, exist_ok=True)

    for img_path in image_files:
        shutil.copy(img_path, images_output_dir)

    print('Conversion to COLMAP format completed successfully.')

if __name__ == '__main__':
    main()
