"""
Preprocess training set
"""

import os
import numpy as np
from PIL import Image

FACE_PATH = "image_manager/training_set/face"
NOT_FACE_PATH = "image_manager/training_set/not_face"


# TODO


def add_random_images(n_images=1000):
    """
    Add random images to the non face training set.
    Format 22x22, grayscale, numpy array.

    Args:
        n_images (int): Number of random images to add.
    """

    not_face_path = NOT_FACE_PATH
    if not os.path.exists(not_face_path):
        os.makedirs(not_face_path)

    for i in range(n_images):
        random_image = np.random.randint(0, 256, (22, 22), dtype=np.uint8)
        image_path = os.path.join(not_face_path, f"random_image_{i}.png")
        Image.fromarray(random_image).save(image_path)


if __name__ == "__main__":
    add_random_images(1000)
    print(f"Added 1000 random images to {NOT_FACE_PATH}.\n")
