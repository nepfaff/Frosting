#!/bin/bash

# Paths (update these to match your dataset)
DATASET_PATH=$1
IMAGE_PATH="$DATASET_PATH/images"
DATABASE_PATH="$DATASET_PATH/database.db"
SPARSE_TEXT_PATH="$DATASET_PATH/sparse/0"
SPARSE_BIN_PATH="$DATASET_PATH/sparse/0_bin"
TRIANGULATED_PATH="$DATASET_PATH/sparse/triangulated"
TRIANGULATED_TEXT_PATH="$DATASET_PATH/sparse/triangulated_text"
MASKS_PATH="$DATASET_PATH/masks"  # Optional, if using masks

# Read camera intrinsics from cam_K.txt
cam_K=($(cat "$DATASET_PATH/cam_K.txt"))
fx=${cam_K[0]}
fy=${cam_K[4]}
cx=${cam_K[2]}
cy=${cam_K[5]}

# Step 1: Feature Extraction
colmap feature_extractor \
    --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_params "$fx,$fy,$cx,$cy" \
    --ImageReader.mask_path $MASKS_PATH  # Include this line if using masks

# Step 2: Feature Matching
colmap exhaustive_matcher \
    --database_path $DATABASE_PATH

# Create output directories
mkdir -p $SPARSE_BIN_PATH
mkdir -p $TRIANGULATED_PATH
mkdir -p $TRIANGULATED_TEXT_PATH

# Step 3: Convert Text Model to Binary
colmap model_converter \
    --input_path $SPARSE_TEXT_PATH \
    --output_path $SPARSE_BIN_PATH \
    --output_type BIN

# Step 4: Triangulate Points Using Known Poses
colmap point_triangulator \
    --database_path $DATABASE_PATH \
    --image_path $IMAGE_PATH \
    --input_path $SPARSE_BIN_PATH \
    --output_path $TRIANGULATED_PATH \
    --Mapper.fix_existing_images 1

# Step 5: Convert Triangulated Model to Text
colmap model_converter \
    --input_path $TRIANGULATED_PATH \
    --output_path $TRIANGULATED_TEXT_PATH \
    --output_type TXT

# Step 6: Save old model and rename new model.
mv $SPARSE_TEXT_PATH $SPARSE_TEXT_PATH"_old"
mv $TRIANGULATED_TEXT_PATH $SPARSE_TEXT_PATH
