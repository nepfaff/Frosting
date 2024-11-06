import os
import logging
from argparse import ArgumentParser
import shutil
import numpy as np
import cv2
import struct

# Define COLMAP camera model constants
CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}

CAMERA_MODEL_NAMES = {v: k for k, v in CAMERA_MODEL_IDS.items()}

CAMERA_MODEL_PARAMS_NUM = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}

# Define Camera class
class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

# Functions to read and write cameras in binary format
def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            model_name = CAMERA_MODEL_NAMES[model_id]
            num_params = CAMERA_MODEL_PARAMS_NUM[model_name]
            params = struct.unpack("<" + "d" * num_params, fid.read(8 * num_params))
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
    return cameras

def write_cameras_binary(cameras, path):
    with open(path, 'wb') as fid:
        fid.write(struct.pack('<Q', len(cameras)))
        for camera in cameras.values():
            model_id = CAMERA_MODEL_IDS[camera.model]
            camera_id = camera.id
            width = camera.width
            height = camera.height
            num_params = len(camera.params)
            fid.write(struct.pack('<iiQQ', camera_id, model_id, width, height))
            fid.write(struct.pack('<' + 'd' * num_params, *camera.params))

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Custom Image Undistortion using OpenCV
## We need to undistort our images into ideal pinhole intrinsics while preserving the alpha channel.

# Path to the sparse model directory
sparse_model_dir = os.path.join(args.source_path, "distorted", "sparse", "0")

# Check if the directory exists
if not os.path.exists(sparse_model_dir):
    # If '0' doesn't exist, find any available model directory
    sparse_dirs = [d for d in os.listdir(os.path.join(args.source_path, "distorted", "sparse")) if os.path.isdir(os.path.join(args.source_path, "distorted", "sparse", d))]
    if not sparse_dirs:
        raise FileNotFoundError("No sparse model directories found.")
    sparse_model_dir = os.path.join(args.source_path, "distorted", "sparse", sparse_dirs[0])

# Read the binary cameras.bin file
cameras_bin_path = os.path.join(sparse_model_dir, "cameras.bin")
cameras = read_cameras_binary(cameras_bin_path)

# We assume a single camera
camera = next(iter(cameras.values()))

if camera.model != "OPENCV":
    raise ValueError(f"Unsupported camera model: {camera.model}")

# Extract original camera parameters
fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])
dist_coeffs = np.array([k1, k2, p1, p2])

# Create a directory for undistorted images
undistorted_dir = os.path.join(args.source_path, "images")
os.makedirs(undistorted_dir, exist_ok=True)

# Directory with the input images
input_dir = os.path.join(args.source_path, "input")

# List of images
image_list = os.listdir(input_dir)

# Prepare to update the camera parameters
# We'll compute the new camera matrix based on the first image
first_image_path = os.path.join(input_dir, image_list[0])
first_img = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
if first_img is None:
    raise ValueError(f"Failed to load image {first_image_path}")
h, w = first_img.shape[:2]

# Compute the new optimal camera matrix
alpha = 0  # Set to 0 to remove unwanted pixels
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha, (w, h))

# Update camera parameters: new_camera_matrix and zero distortion coefficients
fx_new = new_camera_matrix[0, 0]
fy_new = new_camera_matrix[1, 1]
cx_new = new_camera_matrix[0, 2]
cy_new = new_camera_matrix[1, 2]

# Change camera model to PINHOLE
camera.model = "PINHOLE"
camera.params = np.array([fx_new, fy_new, cx_new, cy_new])

# Prepare the new sparse model directory
new_sparse_model_dir = os.path.join(args.source_path, "sparse", "0")
os.makedirs(new_sparse_model_dir, exist_ok=True)

# Write the updated cameras.bin file
new_cameras_bin_path = os.path.join(new_sparse_model_dir, "cameras.bin")
write_cameras_binary({camera.id: camera}, new_cameras_bin_path)

# Copy images.bin and points3D.bin to the new sparse model directory
original_images_bin_path = os.path.join(sparse_model_dir, "images.bin")
original_points3D_bin_path = os.path.join(sparse_model_dir, "points3D.bin")
new_images_bin_path = os.path.join(new_sparse_model_dir, "images.bin")
new_points3D_bin_path = os.path.join(new_sparse_model_dir, "points3D.bin")

shutil.copy2(original_images_bin_path, new_images_bin_path)
shutil.copy2(original_points3D_bin_path, new_points3D_bin_path)

# Undistort each image
for image_name in image_list:
    image_path = os.path.join(input_dir, image_name)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load image {image_path}")
        continue

    # Undistort the image
    undistorted_img = cv2.undistort(img, K, dist_coeffs, None, new_camera_matrix)

    # Crop the image based on ROI
    x, y, w_new, h_new = roi
    undistorted_img = undistorted_img[y:y + h_new, x:x + w_new]

    # Save the undistorted image
    undistorted_image_path = os.path.join(undistorted_dir, image_name)
    cv2.imwrite(undistorted_image_path, undistorted_img)

if args.resize:
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directories
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        for scale, folder in zip([50, 25, 12.5], ["images_2", "images_4", "images_8"]):
            destination_file = os.path.join(args.source_path, folder, file)
            shutil.copy2(source_file, destination_file)
            exit_code = os.system(f"{magick_command} mogrify -resize {scale}% {destination_file}")
            if exit_code != 0:
                logging.error(f"{scale}% resize failed with code {exit_code}. Exiting.")
                exit(exit_code)

print("Done.")
