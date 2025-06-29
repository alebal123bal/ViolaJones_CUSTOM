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


def get_matrix_weights_labels(haar_features=None, integral_images=None):
    """
    Create a feature evaluation matrix, its weights and labels.

    Args:
        haar_features (list, optional): List of Haar features to evaluate.
          If None, generates all Haar features.
        integral_images (list, optional): List of integral images to evaluate the features on.
          If None, loads images from FACE_PATH and NOT_FACE_PATH.

    Returns:
        np.ndarray: Feature evaluation matrix where each row corresponds to a Haar feature and
                    each column corresponds to an image.
        np.ndarray: Weights for each sample in the dataset.
        np.ndarray: Labels for each sample, where 1 indicates a face and 0 indicates a non-face.
    """
    if haar_features is None:
        haar_features = generate_all_haar_features()[0:12]
    if integral_images is None:
        face_images = load_images_from_folder(FACE_PATH)[0:5]
        not_face_images = load_images_from_folder(NOT_FACE_PATH)[0:5]
        integral_images = compute_integral_images(face_images + not_face_images)

    # Initialize the feature evaluation matrix
    num_features = len(haar_features)
    num_images = len(integral_images)
    feature_eval_matrix = np.zeros((num_features, num_images), dtype=np.int32)

    # Evaluate each Haar feature on each integral image
    for i, feature in enumerate(haar_features):
        for j, integral_image in enumerate(integral_images):
            # Evaluate the feature at position (0, 0) as all images are already 22x22
            feature_eval_matrix[i, j] = feature.evaluate(
                integral_image, shift_x=0, shift_y=0
            )

    # Weights
    num_faces = len(face_images)
    num_non_faces = len(not_face_images)
    total_samples = num_faces + num_non_faces

    # Create weights array
    weights = np.ones(total_samples, dtype=np.float16)

    # Assign weights inversely proportional to class size
    weights[:num_faces] = 1.0 / num_faces  # Face weights
    weights[num_faces:] = 1.0 / num_non_faces  # Non-face weights

    # Normalize so weights sum to 1
    weights = weights / weights.sum()

    # Labels
    labels = np.zeros(total_samples, dtype=np.int8)
    labels[:num_faces] = 1  # Faces labeled as 1
    labels[num_faces:] = -1  # Non-faces labeled as -1

    return feature_eval_matrix, weights, labels


# Example usage
if __name__ == "__main__":
    # Create the feature evaluation matrix
    mat, w, lab = get_matrix_weights_labels()

    # Print the shape of the matrix
    print(f"Feature evaluation matrix shape: {mat.shape}")

    # Optionally, print the first few rows and columns
    print("First 5 features for first 5 images:")
    print(mat[:5, :5])
