"""
Generate full-coverage Haar features for object detection using Viola-Jones algorithm.
These features cover the entire detection window with no ignored pixels.
"""

from typing import List, Tuple
from basic_elements.rectangle.rectangle import Rectangle
from basic_elements.haar_feature.haar_feature import HaarFeature


def generate_two_rectangle_horizontal_full(
    window_size: Tuple[int, int] = (22, 22),
) -> List[HaarFeature]:
    """
    Generate two-rectangle horizontal features covering the full window.
    Hardcoded for 22x22 windows. Loop from 1 to 21, and that is the width of the first rect.
    """
    features = []
    width, height = window_size

    # Horizontal splits (various ratios)
    for w in range(1, width):
        for pol in (1, -1):
            rect1 = Rectangle(0, 0, w, height, pol)
            rect2 = Rectangle(w, 0, width - w, height, -pol)
            features.append(HaarFeature([rect1, rect2]))

    return features


def generate_two_rectangle_vertical_full(
    window_size: Tuple[int, int] = (22, 22),
) -> List[HaarFeature]:
    """
    Generate two-rectangle vertical features covering the full window.
    Hardcoded for 22x22 windows. Loop from 1 to 21, and that is the height of the first rect.
    """
    features = []
    width, height = window_size

    # Vertical splits (various ratios)
    for h in range(1, height):
        for pol in (1, -1):
            rect1 = Rectangle(0, 0, width, h, pol)
            rect2 = Rectangle(0, h, width, height - h, -pol)
            features.append(HaarFeature([rect1, rect2]))

    return features


def generate_three_rectangle_horizontal_full(
    window_size: Tuple[int, int] = (22, 22),
) -> List[HaarFeature]:
    """
    Generate three-rectangle horizontal features covering the full window.
    Make all the possible combinations for a hardcoded 22x22 window.
    Example: (1, 1, 20), (1, 2, 19), (1, 3, 18), ..., (20, 1, 1)
    """

    features = []
    width, height = window_size

    for w1 in range(1, width - 1):
        for w2 in range(1, width - w1):
            w3 = width - w1 - w2
            if w3 <= 0:
                continue
            for pol in (1, -1):
                rect1 = Rectangle(0, 0, w1, height, pol)
                rect2 = Rectangle(w1, 0, w2, height, -pol)
                rect3 = Rectangle(w1 + w2, 0, w3, height, pol)
                features.append(HaarFeature([rect1, rect2, rect3]))

    return features


def generate_three_rectangle_vertical_full(
    window_size: Tuple[int, int] = (22, 22),
) -> List[HaarFeature]:
    """
    Generate three-rectangle vertical features covering the full window.
    Make all the possible combinations for a hardcoded 22x22 window.
    Example: (1, 1, 20), (1, 2, 19), (1, 3, 18), ..., (20, 1, 1)
    """

    features = []
    width, height = window_size

    for h1 in range(1, height - 1):
        for h2 in range(1, height - h1):
            h3 = height - h1 - h2
            if h3 <= 0:
                continue
            for pol in (1, -1):
                rect1 = Rectangle(0, 0, width, h1, pol)
                rect2 = Rectangle(0, h1, width, h2, -pol)
                rect3 = Rectangle(0, h1 + h2, width, h3, pol)
                features.append(HaarFeature([rect1, rect2, rect3]))

    return features


def generate_four_rectangle_grid_full(
    window_size: Tuple[int, int] = (22, 22),
) -> List[HaarFeature]:
    """
    Generate four-rectangle grid features covering the full window.
    Make all the possible combinations for a hardcoded 22x22 window.
    """

    features = []
    width, height = window_size

    # Split into 2x2 grid
    for w1 in range(1, width - 1):
        for h1 in range(1, height - 1):
            w2 = width - w1
            h2 = height - h1
            if w2 <= 0 or h2 <= 0:
                continue
            for pol in (1, -1):
                rect1 = Rectangle(0, 0, w1, h1, pol)
                rect2 = Rectangle(w1, 0, w2, h1, -pol)
                rect3 = Rectangle(0, h1, w1, h2, -pol)
                rect4 = Rectangle(w1, h1, w2, h2, pol)
                features.append(HaarFeature([rect1, rect2, rect3, rect4]))

    return features


def generate_four_rectangle_horizontal_full(
    window_size: Tuple[int, int] = (22, 22),
) -> List[HaarFeature]:
    """
    Generate four-rectangle horizontal features covering the full window.
    Make all the possible combinations for a hardcoded 22x22 window.
    Example: (1, 1, 1, 20), (1, 1, 2, 19), (1, 1, 3, 18), ..., (20, 1, 1, 1)
    """

    features = []
    width, height = window_size

    for w1 in range(1, width - 2):
        for w2 in range(1, width - w1 - 1):
            for w3 in range(1, width - w1 - w2):
                w4 = width - w1 - w2 - w3
                if w4 <= 0:
                    continue
                for pol in (1, -1):
                    rect1 = Rectangle(0, 0, w1, height, pol)
                    rect2 = Rectangle(w1, 0, w2, height, -pol)
                    rect3 = Rectangle(w1 + w2, 0, w3, height, pol)
                    rect4 = Rectangle(w1 + w2 + w3, 0, w4, height, -pol)
                    features.append(HaarFeature([rect1, rect2, rect3, rect4]))

    return features


def generate_four_rectangle_vertical_full(
    window_size: Tuple[int, int] = (22, 22),
) -> List[HaarFeature]:
    """
    Generate four-rectangle vertical features covering the full window.
    Make all the possible combinations for a hardcoded 22x22 window.
    Example: (1, 1, 1, 20), (1, 1, 2, 19), (1, 1, 3, 18), ..., (20, 1, 1, 1)
    """

    features = []
    width, height = window_size

    for h1 in range(1, height - 2):
        for h2 in range(1, height - h1 - 1):
            for h3 in range(1, height - h1 - h2):
                h4 = height - h1 - h2 - h3
                if h4 <= 0:
                    continue
                for pol in (1, -1):
                    rect1 = Rectangle(0, 0, width, h1, pol)
                    rect2 = Rectangle(0, h1, width, h2, -pol)
                    rect3 = Rectangle(0, h1 + h2, width, h3, pol)
                    rect4 = Rectangle(0, h1 + h2 + h3, width, h4, -pol)
                    features.append(HaarFeature([rect1, rect2, rect3, rect4]))

    return features


def generate_all_full_coverage_haar_features(
    window_size: Tuple[int, int] = (22, 22),
    feature_types: List[str] = None,
) -> List[HaarFeature]:
    """
    Generate all full-coverage Haar features for the given window size.

    Args:
        window_size: Tuple of (width, height) for the detection window
        feature_types: List of feature types to generate. If None, generates all types.
        Valid types: ['horizontal_2', 'vertical_2', 'horizontal_3', 'vertical_3',
                     'grid_4', 'strips_4', 'cross_5', 'grid_6', 'center_7', 'border_8', 'grid_9']

    Returns:
        List of all generated HaarFeature objects
    """

    if feature_types is None:
        feature_types = [
            "horizontal_2_full",
            "vertical_2_full",
            "horizontal_3_full",
            "vertical_3_full",
            "grid_4_full",
            "horizontal_4_full",
            "vertical_4_full",
        ]

    all_features = []

    print(
        f"\n🔄 Generating full Haar features for {window_size[0]}x{window_size[1]} window..."
    )

    if "horizontal_2_full" in feature_types:
        features = generate_two_rectangle_horizontal_full(window_size)
        all_features.extend(features)
        print(f"🔲 Generated {len(features)} two-rectangle horizontal features")

    if "vertical_2_full" in feature_types:
        features = generate_two_rectangle_vertical_full(window_size)
        all_features.extend(features)
        print(f"🔲 Generated {len(features)} two-rectangle vertical features")

    if "horizontal_3_full" in feature_types:
        features = generate_three_rectangle_horizontal_full(window_size)
        all_features.extend(features)
        print(f"👁️ Generated {len(features)} three-rectangle horizontal features")

    if "vertical_3_full" in feature_types:
        features = generate_three_rectangle_vertical_full(window_size)
        all_features.extend(features)
        print(f"👁️ Generated {len(features)} three-rectangle vertical features")

    if "grid_4_full" in feature_types:
        features = generate_four_rectangle_grid_full(window_size)
        all_features.extend(features)
        print(f"🎭 Generated {len(features)} four-rectangle grid features")

    if "horizontal_4_full" in feature_types:
        features = generate_four_rectangle_horizontal_full(window_size)
        all_features.extend(features)
        print(f"🎭 Generated {len(features)} four-rectangle horizontal features")

    if "vertical_4_full" in feature_types:
        features = generate_four_rectangle_vertical_full(window_size)
        all_features.extend(features)
        print(f"🎭 Generated {len(features)} four-rectangle vertical features")

    print(f"📊 Total full-coverage features generated: {len(all_features)}\n")
    return all_features


# Example
if __name__ == "__main__":

    import os
    import numpy as np

    from image_manager.image_loader.image_loader import load_image_as_array

    my_features = generate_all_full_coverage_haar_features(
        feature_types=[
            # "horizontal_2_full",
            # "vertical_2_full",
            # "horizontal_3_full",
            # "vertical_3_full",
            # "grid_4_full",
            # "horizontal_4_full"
            "vertical_4_full",
        ],
        window_size=(22, 22),
    )

    # Import a grayscale image for testing
    image = load_image_as_array(
        os.path.join(os.getcwd(), "basic_elements", "feature_gen", "test_image.png")
    )

    for i, feat in enumerate(my_features[0:]):
        # Compute the integral image
        padded = np.pad(image, ((1, 0), (1, 0)), mode="constant", constant_values=0)
        intgr_image = np.cumsum(np.cumsum(padded.astype(np.int32), axis=0), axis=1)

        # Evaluate the feature with the window at position (0, 0)
        feat_val = feat.evaluate(intgr_image, shift_x=0, shift_y=0)

        print(f"Feature {i} value at (0, 0): {feat_val}")

        # Plot the feature
        feat.plot(grayscale_image=image)
