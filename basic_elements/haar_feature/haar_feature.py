"""
Class to represent a Haar Feature for object detection.
A class representing a Haar Feature for object detection, consisting of multiple rectangles.

Raises:
    ValueError: If the feature has no rectangles.
"""

import numpy as np
import matplotlib.pyplot as plt

from basic_elements.rectangle.rectangle import Rectangle


class HaarFeature:
    """
    HaarFeature class for object detection using Viola-Jones algorithm.
    """

    def __init__(
        self,
        rectangles: list[Rectangle],
        window_size: tuple[int, int] = (22, 22),
        name="HaarFeature",
    ):
        """
        Initialize a Haar Feature.

        Args:
            rectangles (list[Rectangle]): List of Rectangle objects that make up the feature
            window_size (tuple[int, int]): Size of the detection window (width, height)
            name (str): Name of the feature for identification
        """

        if not rectangles:
            raise ValueError("Feature must have at least one rectangle")

        self.rectangles = rectangles
        self.window_size = window_size
        self.name = name

    def evaluate(self, integral_image, shift_x=0, shift_y=0):
        """
        Evaluate the feature at position (x, y) using an integral image.

        Args:
            integral_image: 2D array representing the integral image
            shift_x, shift_y: Window positional shift to evaluate the feature

        Returns:
            Feature value (sum of signed rectangle areas)
        """

        feature_value = 0

        for rect in self.rectangles:
            # Calculate rectangle bounds relative to feature position
            x1 = shift_x + rect.x
            y1 = shift_y + rect.y
            x2 = shift_x + rect.x2
            y2 = shift_y + rect.y2

            # Calculate sum using integral image
            a = integral_image[y2, x2]
            b = integral_image[y1, x2]
            c = integral_image[y2, x1]
            d = integral_image[y1, x1]

            rect_sum = a - b - c + d

            feature_value += rect.sign * rect_sum

        return feature_value

    def scale(self, scale):
        """
        Scale the Haar feature by a factor.
        Not only scale the rectangles, but of course also the window size.

        Args:
            scale (float): Scale factor to resize the rectangles

        Raises:
            ValueError: If the scale factor is not positive
        """

        if scale <= 0:
            raise ValueError("Scale factor must be positive")

        # Scale the window size
        self.window_size = (
            int(self.window_size[0] * scale),
            int(self.window_size[1] * scale),
        )

        # Scale each rectangle in the feature
        for rect in self.rectangles:
            rect.scale(scale)

    def plot(self, grayscale_image=None, shift_x=0, shift_y=0):
        """
        Plot the Haar feature rectangles on a grid.

        Args:
            grayscale_image: Optional grayscale image to overlay rectangles on
            shift_x: Horizontal offset for the Haar feature window (default: 0)
            shift_y: Vertical offset for the Haar feature window (default: 0)
        """
        _, ax = plt.subplots()

        # Determine the display size based on image or window size
        if grayscale_image is not None:
            height, width = grayscale_image.shape
            display_width, display_height = width, height

            # Check if Haar feature window (with shift) fits within image dimensions
            if (
                shift_x + self.window_size[0] > width
                or shift_y + self.window_size[1] > height
                or shift_x < 0
                or shift_y < 0
            ):
                raise ValueError(
                    f"Haar feature window at position ({shift_x}, {shift_y}) "
                    f"with size ({self.window_size[0]}x{self.window_size[1]}) "
                    f"doesn't fit within image dimensions ({width}x{height})"
                )
        else:
            # When no image, display area should accommodate window + shift
            display_width = self.window_size[0] + shift_x
            display_height = self.window_size[1] + shift_y

        # If a grayscale image is provided, display its numbers inside the grid squares
        if grayscale_image is not None:
            # Display the image as background using its original dimensions (STAYS FIXED)
            ax.imshow(
                grayscale_image,
                cmap="gray",
                extent=(0, display_width, display_height, 0),  # Flipped y-extent
                alpha=0.8,
                vmin=0,
                vmax=255,
            )

            # Add numeric values in each grid square (BACKGROUND STAYS FIXED)
            # for i in range(height):
            #     for j in range(width):
            #         ax.text(
            #             j + 0.5,
            #             i + 0.5,  # Use i directly (no more flipping)
            #             str(int(grayscale_image[i, j])),
            #             ha="center",
            #             va="center",
            #             fontsize=8,
            #             fontweight="bold",
            #             color="red",
            #         )

        # Draw the Haar feature window border in green (SHIFTED POSITION)
        border_thickness = 0.15  # How much to inset the border
        inner_border = plt.Rectangle(
            (shift_x + border_thickness, shift_y + border_thickness),  # Apply shift
            self.window_size[0] - 2 * border_thickness,
            self.window_size[1] - 2 * border_thickness,
            edgecolor="Green",
            facecolor="none",
            linewidth=3,
        )
        ax.add_patch(inner_border)

        # Draw each rectangle with only borders (SHIFTED POSITION)
        for rect in self.rectangles:
            rect_patch = plt.Rectangle(
                (
                    rect.x + shift_x,
                    rect.y + shift_y,
                ),  # Apply shift to rectangle position
                rect.width,
                rect.height,
                edgecolor=("Yellow" if rect.sign == 1 else "Orange"),
                facecolor="none",
                linewidth=4,
            )
            ax.add_patch(rect_patch)

        # Set limits and aspect using the display dimensions
        ax.set_xlim(0, display_width)
        ax.set_ylim(display_height, 0)  # Flipped y-limits (max to 0)
        ax.set_aspect("equal")
        ax.set_title(f"{self.name} (offset: {shift_x}, {shift_y})")

        # Configure grid to show 1 square per unit
        ax.set_xticks(range(display_width + 1))
        ax.set_yticks(range(display_height + 1))
        plt.grid()

        # If grayscale image is provided, compute integral and haar feature evaluation
        if grayscale_image is not None:
            # Compute integral image
            padded = np.pad(
                grayscale_image, ((1, 0), (1, 0)), mode="constant", constant_values=0
            )
            intgr_image = np.cumsum(np.cumsum(padded.astype(np.int32), axis=0), axis=1)

            # Evaluate the feature at the shifted position
            feat_val = self.evaluate(intgr_image, shift_x=shift_x, shift_y=shift_y)
            ax.set_title(
                f"{self.name} (offset: {shift_x}, {shift_y}) - Value: {feat_val:.2f}"
            )

        plt.show()

    def __str__(self):
        """
        String representation of the Haar feature.
        """
        return f"name={self.name}, window_size={self.window_size}, rectangles={self.rectangles}"


# Example
if __name__ == "__main__":
    # Create example rectangles
    rect1 = Rectangle(0, 0, 5, 4, +1)
    rect2 = Rectangle(5, 0, 5, 4, -1)

    # Create a Haar feature with these rectangles
    haar_feature = HaarFeature(
        [
            rect1,
            rect2,
        ],
        window_size=(22, 22),
        name="Test Feature",
    )

    # Import a grayscale image for testing

    image = np.load("basic_elements/feature_gen/test_image.npy")

    # Plot the feature
    haar_feature.plot(grayscale_image=image)

    # Scale and recenter the feature at (5, 5)
    haar_feature.scale(0.5)
    haar_feature.plot(grayscale_image=image, shift_x=5, shift_y=5)
