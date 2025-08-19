"""
Load images from a folder using PIL and return them as a list of numpy arrays.
Eventually compute the integral images for each loaded image.
"""

import os
import numpy as np
from PIL import Image

FACE_PATH = "image_manager/training_set/face/cropped"
NOT_FACE_PATH = "image_manager/training_set/not_face"


def load_image_as_array(image_path, use_gray=True):
    """
    Load an image from a given path and convert it to a grayscale numpy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Grayscale image as a 2D numpy array.
    """
    try:
        img = Image.open(image_path)
        if use_gray:
            img = img.convert("L")  # Convert to grayscale
        img_array = np.array(img)
        return img_array
    except FileNotFoundError as e:
        print(f"Image {image_path} not found: {e}")
        return None


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

    print(f"📁 Loading images from folder: {folder}")
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

    print("✅ Done.\n")

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

    print(f"Computed {len(intgr_imgs)} integral images")
