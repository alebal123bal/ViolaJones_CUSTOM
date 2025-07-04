"""
Load images, compute their integral images, and create a feature evaluation matrix.
"""

import os
import time

import numpy as np

from basic_elements.feature_gen.feature_gen import generate_all_haar_features
from image_manager.image_loader.image_loader import (
    load_images_from_folder,
    compute_integral_images,
    FACE_PATH,
    NOT_FACE_PATH,
)

MATRIX_PATH = "matrix_creator/eval_matrix_weights_labels"
INT16_MIN, INT16_MAX = -32768, 32767


def get_matrix_weights_labels(
    haar_features=None, face_images=None, not_face_images=None
):
    """
    Create a feature evaluation matrix, its weights and labels.

    Args:
        haar_features (list, optional): List of Haar features to evaluate.
          If None, generates all Haar features.
        face_images (list, optional): List of face images to evaluate the features on.
          If None, loads images from FACE_PATH.
        not_face_images (list, optional): List of non-face images to evaluate the features on.
          If None, loads images from NOT_FACE_PATH.

    Returns:
        np.ndarray: Feature evaluation matrix where each row corresponds to a Haar feature and
                    each column corresponds to an image.
        np.ndarray: Weights for each sample in the dataset.
        np.ndarray: Labels for each sample, where 1 indicates a face and 0 indicates a non-face.
    """

    print("\nCreating feature evaluation matrix, weights, and labels...\n")

    # Load Haar features and integral images if not provided
    if haar_features is None:
        print("Haar features not provided. Generating all Haar features...\n")
        haar_features = generate_all_haar_features()[0:12]
    if face_images is None:
        print("Face images not provided. Loading images from folders...\n")
        face_images = load_images_from_folder(FACE_PATH)[0:6]
    if not_face_images is None:
        print("Not-Face images not provided. Loading images from folders...\n")
        not_face_images = load_images_from_folder(NOT_FACE_PATH)[0:4]

    # Compute integral images for the provided face and not-face images
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
        # Print every 10% of total features
        if i % (num_features // 10) == 0:
            print(f"Evaluated {i + 1}/{num_features} features.")

    # Weights
    num_faces = len(face_images)
    num_non_faces = len(not_face_images)
    total_samples = num_faces + num_non_faces

    # Create weights array
    weights = np.ones(total_samples, dtype=np.float16)

    # Assign weights inversely proportional to class size
    # TODO increase weights for faces
    weights[:num_faces] = 1.0 / num_faces  # Face weights
    weights[num_faces:] = 1.0 / num_non_faces  # Non-face weights

    # Normalize so weights sum to 1
    weights = weights / weights.sum()

    # Labels
    labels = np.zeros(total_samples, dtype=np.int8)
    labels[:num_faces] = 1  # Faces labeled as 1
    labels[num_faces:] = -1  # Non-faces labeled as -1

    print("Created feature evaluation matrix, weights, and labels.\n")
    print(f"Feature evaluation matrix shape: {feature_eval_matrix.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Labels shape: {labels.shape}\n")

    return feature_eval_matrix, weights, labels


def save_matrix_weights_labels(
    folder=MATRIX_PATH,
    matrix=None,
    weights=None,
    labels=None,
):
    """
    Save the feature evaluation matrix, weights, and labels to a file as .npy.

    Args:
        folder (str): Folder path where the data will be saved.
        haar_features (list, optional): List of Haar features to evaluate.
        integral_images (list, optional): List of integral images to evaluate the features on.
    """

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    # Save the matrix, weights, and labels
    np.save(os.path.join(folder, "feature_eval_matrix.npy"), matrix)
    np.save(os.path.join(folder, "weights.npy"), weights)
    np.save(os.path.join(folder, "labels.npy"), labels)

    print(f"\nSaved feature evaluation matrix, weights, and labels to {folder}.\n")


def load_matrix_weights_labels(folder=MATRIX_PATH):
    """
    Load the feature evaluation matrix, weights, and labels from a file.

    Args:
        folder (str): Folder path where the data is saved.

    Returns:
        tuple: (matrix, weights, labels)
    """
    matrix = np.load(os.path.join(folder, "feature_eval_matrix.npy"))
    weights = np.load(os.path.join(folder, "weights.npy"))
    labels = np.load(os.path.join(folder, "labels.npy"))

    print(f"Loaded feature evaluation matrix, weights, and labels from {folder}")

    return matrix, weights, labels


def exists_matrix_weights_labels(folder=MATRIX_PATH):
    """
    Check if the feature evaluation matrix, weights, and labels exist in the specified folder.

    Args:
        folder (str): Folder path where the data is saved.

    Returns:
        bool: True if the files exist, False otherwise.
    """
    return (
        os.path.exists(os.path.join(folder, "feature_eval_matrix.npy"))
        and os.path.exists(os.path.join(folder, "weights.npy"))
        and os.path.exists(os.path.join(folder, "labels.npy"))
    )


def _analyze_matrix(matrix):
    """
    Comprehensive analysis of the feature evaluation matrix to determine
    optimal data type and assess potential for memory optimization.

    Args:
        matrix (np.ndarray): The feature evaluation matrix to analyze.

    Returns:
        bool: True if int16 conversion is recommended, False otherwise.
    """

    # Basic matrix info
    print(f"Matrix shape: {matrix.shape}")
    print(f"Total elements: {matrix.size:,}")
    print(f"Current data type: {matrix.dtype}")
    print(f"Current memory usage: {matrix.nbytes / 1024**2:.1f} MB\n")

    # Value range analysis
    min_val, max_val = np.min(matrix), np.max(matrix)
    print(f"Value range: {min_val:,} to {max_val:,}")
    print(f"Range span: {max_val - min_val:,}")

    # Data type compatibility analysis
    print("\n=== Data Type Analysis ===")

    # int16 analysis
    too_high = np.sum(matrix > INT16_MAX)
    too_low = np.sum(matrix < INT16_MIN)
    total_clipped = too_high + too_low

    print(f"int16 range: {INT16_MIN:,} to {INT16_MAX:,}")
    print(f"Values exceeding int16 max: {too_high:,} ({too_high/matrix.size*100:.4f}%)")
    print(f"Values below int16 min: {too_low:,} ({too_low/matrix.size*100:.4f}%)")
    print(
        f"Total values clipped: {total_clipped:,} ({total_clipped/matrix.size*100:.4f}%)"
    )

    # Memory savings calculation
    current_memory = matrix.nbytes
    int16_memory = matrix.size * 2  # 2 bytes per int16
    memory_saved = current_memory - int16_memory

    print(f"\nMemory with int16: {int16_memory / 1024**2:.1f} MB")
    print(
        f"Memory savings: {memory_saved / 1024**2:.1f} MB ({memory_saved/current_memory*100:.1f}%)"
    )

    # Value distribution analysis
    print("\n=== Value Distribution ===")
    print(f"Mean: {np.mean(matrix):.2f}")
    print(f"Std deviation: {np.std(matrix):.2f}")
    print(f"Median: {np.median(matrix):.2f}")

    # Percentile analysis
    percentiles = [1, 5, 10, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(matrix, p)
        print(f"  {p:2d}th: {val:8.0f}")

    # Zero analysis
    zero_count = np.sum(matrix == 0)
    print(f"\nZero values: {zero_count:,} ({zero_count/matrix.size*100:.2f}%)")

    # Histogram of extreme values
    _ = max(abs(INT16_MIN), INT16_MAX)
    extreme_values = matrix[(matrix > INT16_MAX) | (matrix < INT16_MIN)]

    if len(extreme_values) > 0:
        print("\n=== Extreme Values Analysis ===")
        print(f"Extreme values (beyond int16): {len(extreme_values):,}")
        print(
            f"Extreme value range: {np.min(extreme_values):,} to {np.max(extreme_values):,}"
        )
        print(f"Most extreme positive: {np.max(matrix):,}")
        print(f"Most extreme negative: {np.min(matrix):,}")

        # Show distribution of extreme values
        if len(extreme_values) <= 20:
            print(f"All extreme values: {sorted(extreme_values)}")
        else:
            print(
                f"""Sample extreme values: {sorted(extreme_values)[:10]} ..."""
                f"""{sorted(extreme_values)[-10:]}"""
            )

    # Recommendation
    print("\n=== Recommendation ===")
    clipping_percentage = total_clipped / matrix.size * 100

    if clipping_percentage < 0.01:
        print(
            "✅ Very few values would be clipped (<0.01%) - int16 conversion recommended"
        )
        return True
    if clipping_percentage < 0.1:
        print(
            "⚠️  Small percentage would be clipped (<0.1%) - int16 conversion likely safe"
        )
        return True
    if clipping_percentage < 1.0:
        print("⚠️  Moderate clipping (<1%) - test int16 performance vs memory trade-off")
        return True
    print("❌ Significant clipping (>1%) - int16 conversion may hurt model performance")
    return False


def clip(matrix, weights, labels, dtype=np.int16, chunk_size=10000):
    """
    Clip the feature evaluation matrix to fit within the specified data type range.
    Uses chunked processing to minimize memory usage.

    Args:
        matrix (np.ndarray): The feature evaluation matrix to clip.
        weights (np.ndarray): Weights corresponding to the samples.
        labels (np.ndarray): Labels corresponding to the samples.
        dtype (type): Desired data type for the clipped matrix (default: np.int16).
        chunk_size (int): Number of rows to process in each chunk.
    """

    if dtype == np.int16:
        print("\nClipping matrix to fit within int16 range (-32768 to 32767)...")
        print(f"Processing in chunks of {chunk_size} rows...")

        # Create output array
        clipped_matrix = np.empty(matrix.shape, dtype=dtype)

        # Process in chunks
        for i in range(0, matrix.shape[0], chunk_size):
            end_idx = min(i + chunk_size, matrix.shape[0])

            # Process chunk
            chunk = matrix[i:end_idx]
            np.clip(chunk, INT16_MIN, INT16_MAX, out=chunk)  # In-place clip
            clipped_matrix[i:end_idx] = chunk.astype(dtype)

            if i % (chunk_size * 10) == 0:  # Progress update
                print(f"Processed {i}/{matrix.shape[0]} rows...")

        return clipped_matrix, weights, labels

    raise ValueError(f"Unsupported dtype: {dtype}")


def create(haar_features=None, face_images=None, not_face_images=None):
    """
    Create the feature evaluation matrix, weights, and labels if they do not exist.

    Args:
        haar_features (list, optional): List of Haar features to evaluate.
          If None, generates all Haar features.
        face_images (list, optional): List of face images to evaluate the features on.
        not_face_images (list, optional): List of non-face images to evaluate the features on.
    """

    start_time = time.time()

    # Check if the matrix, weights, and labels already exist
    if exists_matrix_weights_labels(folder=MATRIX_PATH):
        print("\nMatrix, weights, and labels already exist. Returning.\n")
        return

    # Create the feature evaluation matrix
    mat, w, l = get_matrix_weights_labels(
        haar_features=haar_features,
        face_images=face_images,
        not_face_images=not_face_images,
    )

    # Analyze the matrix to determine if clipping is safe
    do_clipping = _analyze_matrix(matrix=mat)

    # If clipping is safe, clip the matrix
    if do_clipping:
        mat, w, l = clip(mat, w, l, dtype=np.int16)
    else:
        print("Clipping unsafe - keeping original matrix.")

    # Save the matrix, weights, and labels to a file
    save_matrix_weights_labels(
        folder=MATRIX_PATH,
        matrix=mat,
        weights=w,
        labels=l,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nFeature evaluation matrix created in {elapsed_time:.2f} seconds.\n")


# Example usage
if __name__ == "__main__":
    # Create the feature evaluation matrix, weights, and labels
    create()
