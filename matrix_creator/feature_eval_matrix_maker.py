"""
Load images, compute their integral images, and create a feature evaluation matrix.
"""

import numpy as np

from basic_elements.feature_gen.feature_gen import generate_all_haar_features
from image_manager.image_loader.image_loader import (
    load_images_from_folder,
    compute_integral_images,
    FACE_PATH,
    NOT_FACE_PATH,
)


def create_feature_eval_matrix(haar_features=None, integral_images=None):
    """
    Create a feature evaluation matrix.

    Returns:
        np.ndarray: Feature evaluation matrix where each row corresponds to a Haar feature and
                    each column corresponds to an image .
    """
    if haar_features is None:
        haar_features = generate_all_haar_features()[0:10]
    if integral_images is None:
        face_images = load_images_from_folder(FACE_PATH)
        not_face_images = load_images_from_folder(NOT_FACE_PATH)
        integral_images = compute_integral_images(face_images + not_face_images)[0:5]

    # Initialize the feature evaluation matrix
    num_features = len(haar_features)
    num_images = len(integral_images)
    feature_eval_matrix = np.zeros((num_features, num_images), dtype=np.float32)

    # Evaluate each Haar feature on each integral image
    for i, feature in enumerate(haar_features):
        for j, integral_image in enumerate(integral_images):
            # Evaluate the feature at position (0, 0) as all images are already 22x22
            feature_eval_matrix[i, j] = feature.evaluate(
                integral_image, shift_x=0, shift_y=0
            )

    return feature_eval_matrix


# Example usage
if __name__ == "__main__":
    # Create the feature evaluation matrix
    mat = create_feature_eval_matrix()

    # Print the shape of the matrix
    print(f"Feature evaluation matrix shape: {mat.shape}")

    # Optionally, print the first few rows and columns
    print("First 5 features for first 5 images:")
    print(mat[:5, :5])
