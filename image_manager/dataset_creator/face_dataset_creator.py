"""
Create face dataset.
"""

# pylint: disable=all

import os
import numpy as np
from PIL import Image
from image_manager.image_loader.image_loader import (
    load_images_from_folder,
    FACE_PATH,
    NOT_FACE_PATH,
)


def remove_frame(frame_size: int):
    """
    Remove a rectangular frame from the image.
    """
    for idx, image in enumerate(load_images_from_folder(FACE_PATH)):
        # Remove the frame
        image = image[frame_size:-frame_size, frame_size:-frame_size]

        # Resize to 22x22 grayscale
        image = Image.fromarray(image).resize((22, 22)).convert("L")

        # Create output directory if it doesn't exist on one level up as FACE_PATH
        output_dir = os.path.join(FACE_PATH, "cropped")
        os.makedirs(output_dir, exist_ok=True)

        # Save the processed image
        image.save(os.path.join(output_dir, f"face_{idx}.png"))

        if idx % 500 == 0:
            print(f"Processed {idx} images from the face dataset.")


if __name__ == "__main__":
    # Remove frame considering images are already 22x22
    remove_frame(2)
