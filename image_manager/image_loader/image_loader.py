"""
Load images from a folder using PIL and return them as a list of numpy arrays.
Eventually compute the integral images for each loaded image.
"""

import os
import numpy as np
from PIL import Image

FACE_PATH = "image_manager/training_set/face"
NOT_FACE_PATH = "image_manager/training_set/not_face"


def compute_integral_image(image):
    """
    Compute the integral image of a given image.

    Args:
        image (np.ndarray): Input image as a 2D numpy array.

    Returns:
        np.ndarray: Integral image as a 2D numpy array.
    """

    # Pad the image to handle borders
    padded = np.pad(image, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    # Compute the integral image
    intgr_image = np.cumsum(np.cumsum(padded.astype(np.int32), axis=0), axis=1)

    return intgr_image


def load_images_from_folder(folder):
    """
    Load images from a specified folder and return them as a list of numpy arrays.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        List[np.ndarray]: List of images as numpy arrays.
    """

    images = []

    for filename in os.listdir(folder):
        if (
            filename.endswith(".png")
            or filename.endswith(".jpg")
            or filename.endswith(".jpeg")
        ):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            img_array = np.array(img)
            images.append(img_array)

    return images


def compute_integral_images(images):
    """
    Compute integral images from a list of np images.

    Args:
        images: List of images as numpy arrays.

    Returns:
        List[np.ndarray]: List of integral images as numpy arrays.
    """

    integral_images = [compute_integral_image(img) for img in images]

    return integral_images


# Test
if __name__ == "__main__":
    # load images from the specified folder
    imgs = load_images_from_folder(FACE_PATH)

    # Compute integrals
    intgr_imgs = compute_integral_images(images=imgs)

    print(f"Loaded and computed {len(intgr_imgs)} integral images from {FACE_PATH}")
