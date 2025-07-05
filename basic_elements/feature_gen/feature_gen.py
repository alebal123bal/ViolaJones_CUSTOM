"""
Generate Haar features for object detection using Viola-Jones algorithm.
"""

from typing import List, Tuple
from basic_elements.rectangle.rectangle import Rectangle
from basic_elements.haar_feature.haar_feature import HaarFeature


def generate_two_rectangle_horizontal(
    window_size: Tuple[int, int] = (22, 22),
    x_start: int = 0,
    y_start: int = 0,
) -> List[HaarFeature]:
    """
    Generate two-rectangle horizontal features (left-right pattern).

    Args:
        window_size: Tuple of (width, height) for the detection window
        x_start: Starting x coordinate for feature generation
        y_start: Starting y coordinate for feature generation

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    for pol in (1, -1):
        # Try different positions and sizes
        for y in range(
            y_start, height - y_start - 1
        ):  # -1 to ensure minimum height of 2
            for x in range(
                x_start, width - x_start - 1
            ):  # -1 to ensure minimum width of 2
                for h in range(2, height - y + 1):  # Minimum height of 2
                    for w in range(2, width - x, 2):  # Even widths only, step by 2
                        if x + w <= width:
                            rect_width = w // 2
                            rect1 = Rectangle(x, y, rect_width, h, pol)
                            rect2 = Rectangle(x + rect_width, y, rect_width, h, -pol)
                            features.append(HaarFeature([rect1, rect2]))

    return features


def generate_two_rectangle_vertical(
    window_size: Tuple[int, int] = (22, 22),
    x_start: int = 0,
    y_start: int = 0,
) -> List[HaarFeature]:
    """
    Generate two-rectangle vertical features (top-bottom pattern).

    Args:
        window_size: Tuple of (width, height) for the detection window
        x_start: Starting x coordinate for feature generation
        y_start: Starting y coordinate for feature generation

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    for pol in (1, -1):
        for y in range(y_start, height - y_start - 1):
            for x in range(x_start, width - x_start - 1):
                for w in range(2, width - x + 1):  # Minimum width of 2
                    for h in range(2, height - y, 2):  # Even heights only, step by 2
                        if y + h <= height:
                            rect_height = h // 2
                            rect1 = Rectangle(x, y, w, rect_height, pol)
                            rect2 = Rectangle(x, y + rect_height, w, rect_height, -pol)
                            features.append(HaarFeature([rect1, rect2]))

    return features


def generate_three_rectangle_horizontal(
    window_size: Tuple[int, int] = (22, 22),
    x_start: int = 0,
    y_start: int = 0,
) -> List[HaarFeature]:
    """
    Generate three-rectangle horizontal features (left-center-right pattern).

    Args:
        window_size: Tuple of (width, height) for the detection window
        x_start: Starting x coordinate for feature generation
        y_start: Starting y coordinate for feature generation

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    for pol in (1, -1):
        for y in range(y_start, height - y_start - 1):
            for x in range(
                x_start, width - x_start - 2
            ):  # -2 to ensure minimum width of 3
                for h in range(2, height - y + 1):
                    for w in range(3, width - x, 3):  # Widths divisible by 3, step by 3
                        if x + w <= width:
                            rect_width = w // 3
                            rect1 = Rectangle(x, y, rect_width, h, pol)
                            rect2 = Rectangle(x + rect_width, y, rect_width, h, -pol)
                            rect3 = Rectangle(
                                x + 2 * rect_width, y, rect_width, h, +pol
                            )
                            features.append(HaarFeature([rect1, rect2, rect3]))

    return features


def generate_three_rectangle_vertical(
    window_size: Tuple[int, int] = (22, 22),
    x_start: int = 0,
    y_start: int = 0,
) -> List[HaarFeature]:
    """
    Generate three-rectangle vertical features (top-center-bottom pattern).

    Args:
        window_size: Tuple of (width, height) for the detection window
        x_start: Starting x coordinate for feature generation
        y_start: Starting y coordinate for feature generation

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    for pol in (1, -1):
        for y in range(
            y_start, height - y_start - 2
        ):  # -2 to ensure minimum height of 3
            for x in range(x_start, width - x_start - 1):
                for w in range(2, width - x + 1):
                    for h in range(
                        3, height - y, 3
                    ):  # Heights divisible by 3, step by 3
                        if y + h <= height:
                            rect_height = h // 3
                            rect1 = Rectangle(x, y, w, rect_height, pol)
                            rect2 = Rectangle(x, y + rect_height, w, rect_height, -pol)
                            rect3 = Rectangle(
                                x, y + 2 * rect_height, w, rect_height, pol
                            )
                            features.append(HaarFeature([rect1, rect2, rect3]))

    return features


def generate_four_rectangle_diagonal(
    window_size: Tuple[int, int] = (22, 22),
    x_start: int = 0,
    y_start: int = 0,
) -> List[HaarFeature]:
    """
    Generate four-rectangle diagonal features (2x2 checkerboard pattern).

    Args:
        window_size: Tuple of (width, height) for the detection window
        x_start: Starting x coordinate for feature generation
        y_start: Starting y coordinate for feature generation

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    for pol in (1, -1):
        for y in range(y_start, height - y_start - 1):
            for x in range(x_start, width - x_start - 1):
                for h in range(2, height - y, 2):  # Even heights only
                    for w in range(2, width - x, 2):  # Even widths only
                        if x + w <= width and y + h <= height:
                            rect_width = w // 2
                            rect_height = h // 2

                            rect1 = Rectangle(
                                x, y, rect_width, rect_height, pol
                            )  # Top-left
                            rect2 = Rectangle(
                                x + rect_width, y, rect_width, rect_height, -pol
                            )  # Top-right
                            rect3 = Rectangle(
                                x, y + rect_height, rect_width, rect_height, -pol
                            )  # Bottom-left
                            rect4 = Rectangle(
                                x + rect_width,
                                y + rect_height,
                                rect_width,
                                rect_height,
                                pol,
                            )  # Bottom-right

                            features.append(HaarFeature([rect1, rect2, rect3, rect4]))

    return features


def generate_eye_like_horizontal(
    window_size: Tuple[int, int] = (22, 22),
    x_start: int = 0,
    y_start: int = 0,
) -> List[HaarFeature]:
    """
    Generate eye-like horizontal features (dark-light-dark pattern).
    Useful for detecting eye regions with bright areas between dark eyebrows/eyes.

    Args:
        window_size: Tuple of (width, height) for the detection window
        x_start: Starting x coordinate for feature generation
        y_start: Starting y coordinate for feature generation

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    for pol in (1, -1):
        for y in range(y_start, height - y_start - 1):
            for x in range(x_start, width - x_start - 2):  # Need at least 3 rectangles
                for h in range(2, height - y + 1):
                    for w in range(3, width - x, 3):  # Widths divisible by 3
                        if x + w <= width:
                            rect_width = w // 3
                            rect1 = Rectangle(x, y, rect_width, h, -pol)  # Dark
                            rect2 = Rectangle(
                                x + rect_width, y, rect_width, h, pol
                            )  # Light
                            rect3 = Rectangle(
                                x + 2 * rect_width, y, rect_width, h, -pol
                            )  # Dark
                            features.append(HaarFeature([rect1, rect2, rect3]))

    return features


def generate_all_haar_features(
    window_size: Tuple[int, int] = (22, 22),
    feature_types: List[str] = None,
    x_start: int = 0,
    y_start: int = 0,
) -> List[HaarFeature]:
    """
    Generate all Haar features for the given window size.

    Args:
        window_size: Tuple of (width, height) for the detection window
        feature_types: List of feature types to generate. If None, generates all types.
        Valid types: ['horizontal_2', 'vertical_2', 'horizontal_3', 'vertical_3', 'diagonal_4']
        x_start: Starting x coordinate for feature generation
        y_start: Starting y coordinate for feature generation

    Returns:
        List of all generated HaarFeature objects
    """

    if feature_types is None:
        feature_types = [
            "horizontal_2",
            "vertical_2",
            "horizontal_3",
            "vertical_3",
            "diagonal_4",
            "eye_like_horizontal",
        ]

    all_features = []

    print(f"\nGenerating Haar features for {window_size[0]}x{window_size[1]} window...")

    if "horizontal_2" in feature_types:
        features = generate_two_rectangle_horizontal(window_size, x_start, y_start)
        all_features.extend(features)
        print(f"Generated {len(features)} two-rectangle horizontal features")

    if "vertical_2" in feature_types:
        features = generate_two_rectangle_vertical(window_size, x_start, y_start)
        all_features.extend(features)
        print(f"Generated {len(features)} two-rectangle vertical features")

    if "horizontal_3" in feature_types:
        features = generate_three_rectangle_horizontal(window_size, x_start, y_start)
        all_features.extend(features)
        print(f"Generated {len(features)} three-rectangle horizontal features")

    if "vertical_3" in feature_types:
        features = generate_three_rectangle_vertical(window_size, x_start, y_start)
        all_features.extend(features)
        print(f"Generated {len(features)} three-rectangle vertical features")

    if "diagonal_4" in feature_types:
        features = generate_four_rectangle_diagonal(window_size, x_start, y_start)
        all_features.extend(features)
        print(f"Generated {len(features)} four-rectangle diagonal features")

    if "eye_like_horizontal" in feature_types:
        features = generate_eye_like_horizontal(window_size, x_start, y_start)
        all_features.extend(features)
        print(f"Generated {len(features)} eye-like horizontal features")

    print(f"Total features generated: {len(all_features)}\n")
    return all_features


# Example
if __name__ == "__main__":
    my_features = generate_all_haar_features(
        feature_types=[
            "eye_like_horizontal",
        ],
        x_start=4,
        y_start=2,
    )

    # Import a grayscale image for testing
    import numpy as np

    # Uncomment the line below to load the precomputed npy array of the 1st image
    # image = np.load("basic_elements/feature_gen/test_image.npy")

    from image_manager.image_loader.image_loader import (
        load_images_from_folder,
        FACE_PATH,
    )

    # Load all
    face_images = load_images_from_folder(FACE_PATH)
    # Use the nth image for testing
    image = face_images[0]

    for i, feat in enumerate(my_features[0:10]):

        # Compute the integral image
        padded = np.pad(image, ((1, 0), (1, 0)), mode="constant", constant_values=0)
        intgr_image = np.cumsum(np.cumsum(padded.astype(np.int32), axis=0), axis=1)

        # Evaluate the feature with the window at position (0, 0)
        feat_val = feat.evaluate(intgr_image, shift_x=0, shift_y=0)

        print(f"Feature {i} value at (0, 0): {feat_val}")

        # Plot the feature
        feat.plot(grayscale_image=image)
