import argparse
import concurrent.futures
import os

from PIL import Image
from tqdm import tqdm


def invert_binary_mask(input_path, output_path):
    # Open the image
    img = Image.open(input_path)

    # Convert the image to grayscale (if it's not already)
    img = img.convert("L")

    # Invert the binary mask
    inverted_img = Image.eval(img, lambda x: 255 - x)

    # Save the inverted image
    inverted_img.save(output_path)


def invert_masks_in_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all PNG files in the input directory
    mask_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]

    def process_mask(mask_file):
        input_path = os.path.join(input_dir, mask_file)
        output_path = os.path.join(output_dir, mask_file)
        invert_binary_mask(input_path, output_path)

    # Invert each mask and save to the output directory using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_mask, mask_files),
                total=len(mask_files),
                desc="Inverting masks",
            )
        )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Invert binary masks in PNG format")
    parser.add_argument(
        "input_dir", type=str, help="Path to the input directory containing PNG masks"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory where inverted masks will be saved",
    )

    args = parser.parse_args()

    # Run the inversion process
    invert_masks_in_directory(args.input_dir, args.output_dir)
