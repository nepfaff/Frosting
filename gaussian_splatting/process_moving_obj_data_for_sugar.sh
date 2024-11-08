#!/bin/bash

# Script for pre-processing the moving object/ static camera data for NerfStudio.
# The input dir should have the following structure:
# - input_dir/
#   - cam_K.txt         # Camera intrinsic parameters
#   - gripper_masks/    # Masks for the gripper
#   - masks/            # Masks for the object + gripper
#   - ob_in_cam/        # Poses of the object in the camera frame (X_CO)
#   - rgb/              # RGB images

# The output will contain the COLMAP data format for SuGAR.

INPUT_DIR=$1
OUTPUT_DIR=$2
NUM_IMAGES=${3:-600}

echo "Adding alpha channel to the RGB images."
python gaussian_splatting/add_alpha_channel.py \
    --images ${INPUT_DIR}/rgb \
    --masks ${INPUT_DIR}/masks/ \
    --out ${INPUT_DIR}/rgb_alpha

echo "Inverting the gripper masks."
python gaussian_splatting/invert_masks.py \
    ${INPUT_DIR}/gripper_masks \
    ${INPUT_DIR}/gripper_masks_inverted

echo "Subsampling the images with DINO."
python gaussian_splatting/sample_most_disimilar_images.py \
    --image_dir ${INPUT_DIR}/rgb_alpha/ \
    --output_dir ${INPUT_DIR}/images_dino_sampled \
    --K ${NUM_IMAGES} \
    --model dino

echo "Copying input dir to output dir."
cp -r ${INPUT_DIR} ${OUTPUT_DIR}

echo "Renaming the directories."
echo "gripper_masks_inverted -> gripper_masks"
rm -rf ${OUTPUT_DIR}/gripper_masks
mv ${OUTPUT_DIR}/gripper_masks_inverted ${OUTPUT_DIR}/gripper_masks
echo "images_dino_sampled -> images"
rm -rf ${OUTPUT_DIR}/images
mv ${OUTPUT_DIR}/images_dino_sampled ${OUTPUT_DIR}/images
echo "ob_in_cam -> poses"
rm -rf ${OUTPUT_DIR}/poses
mv ${OUTPUT_DIR}/ob_in_cam ${OUTPUT_DIR}/poses

echo "Creating the COLMAP data format."
python gaussian_splatting/bundleSdf_to_colmap.py \
    --data_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR}

echo "Computing an initial point cloud using COLMAP."
bash gaussian_splatting/compute_pcd_init.sh ${OUTPUT_DIR}

echo "Done."
