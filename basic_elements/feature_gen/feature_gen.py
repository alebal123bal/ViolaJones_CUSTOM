"""
Generate Haar features for object detection using Viola-Jones algorithm.
"""

from typing import List, Tuple
from basic_elements.rectangle.rectangle import Rectangle
from basic_elements.haar_feature.haar_feature import HaarFeature


def generate_two_rectangle_horizontal(
    window_size: Tuple[int, int] = (22, 22),
    x_padding: int = 0,
    y_padding: int = 0,
    step: int = 1,
) -> List[HaarFeature]:
    """
    Generate all possible two-rectangle horizontal features (left-right pattern).

    Args:
        window_size: Tuple of (width, height) for the detection window
        x_padding: Padding on the x-axis
        y_padding: Padding on the y-axis

    Returns:
        List of HaarFeature objects
    """

    features = []
    width, height = window_size

    # Define minimum width and height for rectangles
    minimum_width = step
    minimum_height = step

    # Define maximum width and height
    maximum_width = width - x_padding
    maximum_height = height - y_padding

    # Define minimum x and y coordinates
    minimum_x = x_padding
    minimum_y = y_padding

    # Define maximum x and y coordinates
    maximum_x = width - x_padding - 1
    maximum_y = height - y_padding

    for pol in (1, -1):
        # Loop all possible x
        for x in range(minimum_x, maximum_x, step):
            # Loop all possible y
            for y in range(minimum_y, maximum_y, step):
                # Loop all possible widths
                for w1 in range(minimum_width, maximum_width - x, step):
                    # Loop the other rectangle's width
                    for w2 in range(minimum_width, maximum_width - x - w1 + 1, step):
                        # Loop all possible heights (they are the same)
                        for h in range(minimum_height, maximum_height - y + 1, step):
                            # Create the two rectangles
                            rect1 = Rectangle(x, y, w1, h, pol)
                            rect2 = Rectangle(x + w1, y, w2, h, -pol)
                            features.append(HaarFeature([rect1, rect2]))

    return features


def generate_two_rectangle_vertical(
    window_size: Tuple[int, int] = (22, 22),
    x_padding: int = 0,
    y_padding: int = 0,
    step: int = 1,
) -> List[HaarFeature]:
    """
    Generate all possible two-rectangle vertical features (left-right pattern).

    Args:
        window_size: Tuple of (width, height) for the detection window
        x_padding: Padding on the x-axis
        y_padding: Padding on the y-axis

    Returns:
        List of HaarFeature objects
    """

    features = []
    width, height = window_size

    # Define minimum width and height for rectangles
    minimum_width = step
    minimum_height = step

    # Define maximum width and height
    maximum_width = width - x_padding
    maximum_height = height - y_padding

    # Define minimum x and y coordinates
    minimum_x = x_padding
    minimum_y = y_padding

    # Define maximum x and y coordinates
    maximum_x = width - x_padding
    maximum_y = height - y_padding - 1

    for pol in (1, -1):
        # Loop all possible x
        for x in range(minimum_x, maximum_x, step):
            # Loop all possible y
            for y in range(minimum_y, maximum_y, step):
                # Loop all possible widths
                for h1 in range(minimum_height, maximum_height - y, step):
                    # Loop the other rectangle's height
                    for h2 in range(minimum_height, maximum_height - y - h1 + 1, step):
                        # Loop all possible widths (they are the same)
                        for w in range(minimum_width, maximum_width - x + 1, step):
                            # Create the two rectangles
                            rect1 = Rectangle(x, y, w, h1, pol)
                            rect2 = Rectangle(x, y + h1, w, h2, -pol)
                            features.append(HaarFeature([rect1, rect2]))

    return features


def generate_three_rectangle_horizontal(
    window_size: Tuple[int, int] = (22, 22),
    x_padding: int = 0,
    y_padding: int = 0,
    step: int = 1,
) -> List[HaarFeature]:
    """
    Generate all possible three-rectangle horizontal features (left-center-right pattern).
    Hardcoded for 22x22 window size.

    Args:
        x_padding: Padding on the x-axis
        y_padding: Padding on the y-axis
        step: Step size for iteration

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    # Define minimum width and height for rectangles
    minimum_width = step
    minimum_height = step

    # Define maximum width and height
    maximum_width = width - x_padding
    maximum_height = height - y_padding

    # Define minimum x and y coordinates
    minimum_x = x_padding
    minimum_y = y_padding

    # Define maximum x and y coordinates
    maximum_x = width - x_padding - 2  # -2 to ensure minimum total width of 3
    maximum_y = height - y_padding

    for pol in (1, -1):
        # Loop all possible x
        for x in range(minimum_x, maximum_x, step):
            # Loop all possible y
            for y in range(minimum_y, maximum_y, step):
                # Loop all possible widths for first rectangle
                for w1 in range(
                    minimum_width, maximum_width - x - 2 + 1, step
                ):  # -2 for other two rectangles
                    # Loop all possible widths for second rectangle
                    for w2 in range(
                        minimum_width, maximum_width - x - w1 - 1 + 1, step
                    ):  # -1 for third rectangle
                        # Loop all possible widths for third rectangle
                        for w3 in range(
                            minimum_width, maximum_width - x - w1 - w2 + 1, step
                        ):
                            # Loop all possible heights (they are the same for all three)
                            for h in range(
                                minimum_height, maximum_height - y + 1, step
                            ):
                                # Check if the three rectangles fit within the window
                                if x + w1 + w2 + w3 <= width:
                                    # Create the three rectangles
                                    rect1 = Rectangle(x, y, w1, h, pol)
                                    rect2 = Rectangle(x + w1, y, w2, h, -pol)
                                    rect3 = Rectangle(x + w1 + w2, y, w3, h, pol)
                                    features.append(HaarFeature([rect1, rect2, rect3]))

    return features


def generate_three_rectangle_vertical(
    window_size: Tuple[int, int] = (22, 22),
    x_padding: int = 0,
    y_padding: int = 0,
    step: int = 1,
) -> List[HaarFeature]:
    """
    Generate all possible three-rectangle vertical features (top-center-bottom pattern).
    Hardcoded for 22x22 window size.

    Args:
        x_padding: Padding on the x-axis
        y_padding: Padding on the y-axis
        step: Step size for iteration

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    # Define minimum width and height for rectangles
    minimum_width = step
    minimum_height = step

    # Define maximum width and height
    maximum_width = width - x_padding
    maximum_height = height - y_padding

    # Define minimum x and y coordinates
    minimum_x = x_padding
    minimum_y = y_padding

    # Define maximum x and y coordinates
    maximum_x = width - x_padding
    maximum_y = height - y_padding - 2  # -2 to ensure minimum total height of 3

    for pol in (1, -1):
        # Loop all possible x
        for x in range(minimum_x, maximum_x, step):
            # Loop all possible y
            for y in range(minimum_y, maximum_y, step):
                # Loop all possible heights for first rectangle
                for h1 in range(
                    minimum_height, maximum_height - y - 2 + 1, step
                ):  # -2 for other two rectangles
                    # Loop all possible heights for second rectangle
                    for h2 in range(
                        minimum_height, maximum_height - y - h1 - 1 + 1, step
                    ):  # -1 for third rectangle
                        # Loop all possible heights for third rectangle
                        for h3 in range(
                            minimum_height, maximum_height - y - h1 - h2 + 1, step
                        ):
                            # Loop all possible widths (they are the same for all three)
                            for w in range(minimum_width, maximum_width - x + 1, step):
                                # Check if the three rectangles fit within the window
                                if y + h1 + h2 + h3 <= height:
                                    # Create the three rectangles
                                    rect1 = Rectangle(x, y, w, h1, pol)
                                    rect2 = Rectangle(x, y + h1, w, h2, -pol)
                                    rect3 = Rectangle(x, y + h1 + h2, w, h3, pol)
                                    features.append(HaarFeature([rect1, rect2, rect3]))

    return features


def generate_four_rectangle_diagonal(
    window_size: Tuple[int, int] = (22, 22),
    x_padding: int = 0,
    y_padding: int = 0,
    step: int = 1,
) -> List[HaarFeature]:
    """
    Generate all possible four-rectangle diagonal features (2x2 checkerboard pattern).
    Hardcoded for 22x22 window size.

    Args:
        x_padding: Padding on the x-axis
        y_padding: Padding on the y-axis
        step: Step size for iteration

    Returns:
        List of HaarFeature objects
    """
    features = []
    width, height = window_size

    # Define minimum width and height for rectangles
    minimum_width = step
    minimum_height = step

    # Define maximum width and height
    maximum_width = width - x_padding
    maximum_height = height - y_padding

    # Define minimum x and y coordinates
    minimum_x = x_padding
    minimum_y = y_padding

    # Define maximum x and y coordinates
    maximum_x = width - x_padding - 1  # -1 to ensure minimum total width of 2
    maximum_y = height - y_padding - 1  # -1 to ensure minimum total height of 2

    for pol in (1, -1):
        # Loop all possible x
        for x in range(minimum_x, maximum_x, step):
            # Loop all possible y
            for y in range(minimum_y, maximum_y, step):
                # Loop all possible widths for first column of rectangles
                for w1 in range(
                    minimum_width, maximum_width - x - 1 + 1, step
                ):  # -1 for second column
                    # Loop all possible widths for second column of rectangles
                    for w2 in range(minimum_width, maximum_width - x - w1 + 1, step):
                        # Loop all possible heights for first row of rectangles
                        for h1 in range(
                            minimum_height, maximum_height - y - 1 + 1, step
                        ):  # -1 for second row
                            # Loop all possible heights for second row of rectangles
                            for h2 in range(
                                minimum_height, maximum_height - y - h1 + 1, step
                            ):
                                # Check if the four rectangles fit within the window
                                if x + w1 + w2 <= width and y + h1 + h2 <= height:
                                    # Create the four rectangles in checkerboard pattern
                                    rect1 = Rectangle(x, y, w1, h1, pol)  # Top-left
                                    rect2 = Rectangle(
                                        x + w1, y, w2, h1, -pol
                                    )  # Top-right
                                    rect3 = Rectangle(
                                        x, y + h1, w1, h2, -pol
                                    )  # Bottom-left
                                    rect4 = Rectangle(
                                        x + w1, y + h1, w2, h2, pol
                                    )  # Bottom-right
                                    features.append(
                                        HaarFeature([rect1, rect2, rect3, rect4])
                                    )

    return features


def generate_all_haar_features(
    window_size: Tuple[int, int] = (22, 22),
    feature_types: List[str] = None,
    x_padding: int = 0,
    y_padding: int = 0,
    step: int = 1,
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
        ]

    all_features = []

    print(
        f"\n🔄 Generating Haar features for {window_size[0]}x{window_size[1]} window..."
    )

    if "horizontal_2" in feature_types:
        features = generate_two_rectangle_horizontal(
            window_size, x_padding, y_padding, step
        )
        all_features.extend(features)
        print(f"🔲 Generated {len(features)} two-rectangle horizontal features")

    if "vertical_2" in feature_types:
        features = generate_two_rectangle_vertical(
            window_size, x_padding, y_padding, step
        )
        all_features.extend(features)
        print(f"🔲 Generated {len(features)} two-rectangle vertical features")

    if "horizontal_3" in feature_types:
        features = generate_three_rectangle_horizontal(
            window_size, x_padding, y_padding, step
        )
        all_features.extend(features)
        print(f"👁️ Generated {len(features)} three-rectangle horizontal features")

    if "vertical_3" in feature_types:
        features = generate_three_rectangle_vertical(
            window_size, x_padding, y_padding, step
        )
        all_features.extend(features)
        print(f"👁️ Generated {len(features)} three-rectangle vertical features")

    if "diagonal_4" in feature_types:
        features = generate_four_rectangle_diagonal(
            window_size, x_padding, y_padding, step
        )
        all_features.extend(features)
        print(f"🎭 Generated {len(features)} four-rectangle diagonal features")

    print(f"📊 Total features generated: {len(all_features)}\n")
    return all_features


# Example
if __name__ == "__main__":
    my_features = generate_all_haar_features(
        feature_types=[
            "horizontal_2",
            "vertical_2",
            "horizontal_3",
            "vertical_3",
            "diagonal_4",
        ],
        x_padding=2,
        y_padding=2,
        step=3,
    )

    import os
    from image_manager.image_loader.image_loader import load_image_as_array

    # Import a grayscale image for testing
    image = load_image_as_array(
        os.path.join(os.getcwd(), "basic_elements", "feature_gen", "test_image.png")
    )

    for i, feat in enumerate(my_features[0:]):
        # Plot the feature
        feat.plot(grayscale_image=image)
