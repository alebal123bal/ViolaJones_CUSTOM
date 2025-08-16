"""
Face Detection Inference Module

This module performs face detection on test images using a trained Viola-Jones cascade classifier.
It implements multi-scale sliding window detection with Non-Maximum Suppression filtering.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from image_manager.image_loader.image_loader import (
    load_image_as_array,
    compute_integral_image,
)
from AdaBoost_smart.utils.io_operations import PickleUtils


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class DetectionConfig:
    """Configuration parameters for face detection."""

    # Test image paths (uncomment the one you want to use)
    IMAGE_PATHS = {
        "easy": "viola_jones/inference/test_image/easy.png",
        "medium": "viola_jones/inference/test_image/medium.png",
        "medium_1": "viola_jones/inference/test_image/medium_1.png",
        "medium_2": "viola_jones/inference/test_image/medium_2.png",
        "hard": "viola_jones/inference/test_image/hard.jpg",
        "very_hard": "viola_jones/inference/test_image/very_hard.jpg",
    }

    # Current test image
    current_image: str = "hard"

    # Classifier path
    classifier_path: str = "_pickle_folder/full_trained_classifier.pkl"

    # Detection parameters
    window_size: int = 22  # Base window size (22x22 pixels)
    step_size: int = 1  # Sliding window step size
    scales: List[float] = None  # Multi-scale detection scales
    iou_threshold: float = 0.5  # Non-Maximum Suppression threshold

    def __post_init__(self):
        """Set default scales if not provided."""
        if self.scales is None:
            self.scales = [6.0, 7.0, 8.0]  # Default scales

    @property
    def image_path(self) -> str:
        """Get the current image path."""
        return self.IMAGE_PATHS[self.current_image]


# ============================================================================
# CORE DETECTION CLASSES
# ============================================================================


class CascadeClassifier:
    """Wrapper for cascade classification operations."""

    @staticmethod
    def predict_window(
        classifier: List[Dict],
        grayscale_image: np.ndarray,
        integral_image: np.ndarray,
        x: int,
        y: int,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Perform cascade prediction on a single window.

        Args:
            classifier: Trained cascade classifier stages
            grayscale_image: Grayscale image window
            integral_image: Integral image of the window
            x, y: Window position (for logging)
            verbose: Whether to print detection details

        Returns:
            Tuple of (is_face_detected, stage_votes)
        """
        stage_votes = []

        for stage_idx, stage in enumerate(classifier):
            stage_vote = 0.0
            stage_features = stage["stage_features"]

            # Evaluate all features in this stage
            for feature_dict in stage_features:
                feature = feature_dict["feature"]
                feature_value = feature.evaluate(integral_image)

                # Apply weak classifier decision
                if feature_dict["direction"] == 0:  # less-equal threshold
                    vote = (
                        feature_dict["alpha"]
                        if feature_value <= feature_dict["threshold"]
                        else -feature_dict["alpha"]
                    )
                else:  # greater-than threshold
                    vote = (
                        feature_dict["alpha"]
                        if feature_value > feature_dict["threshold"]
                        else -feature_dict["alpha"]
                    )

                stage_vote += vote

            stage_votes.append(stage_vote)

            # Early rejection if stage threshold not met
            if stage_vote < stage["stage_thre"]:
                return False, None

        # All stages passed - face detected!
        stage_votes = np.round(np.array(stage_votes), 2)

        if verbose:
            print(f"Face detected at ({x}, {y}) with stage votes: {stage_votes}")

        return True, stage_votes


class MultiScaleDetector:
    """Handles multi-scale sliding window detection."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.classifier = None

    def load_classifier(self) -> None:
        """Load the trained classifier from file."""
        cwd = os.getcwd()
        classifier_path = os.path.join(cwd, self.config.classifier_path)
        self.classifier = PickleUtils.load_pickle_obj(classifier_path)
        print(f"Loaded classifier from: {classifier_path}")

    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Perform multi-scale face detection on an image.

        Args:
            image_path: Path to the input image

        Returns:
            List of detected face bounding boxes [(x, y, w, h), ...]
        """
        if self.classifier is None:
            self.load_classifier()

        # Load test image
        cwd = os.getcwd()
        test_image = load_image_as_array(os.path.join(cwd, image_path))
        print(f"Loaded image: {image_path}, shape: {test_image.shape}")

        all_detections = []

        # Multi-scale detection
        for scale in self.config.scales:
            print(f"Processing scale {scale}...")
            scale_detections = self._detect_at_scale(test_image, scale)
            all_detections.extend(scale_detections)
            print(f"  Found {len(scale_detections)} detections at scale {scale}")

        print(f"Total detections before filtering: {len(all_detections)}")
        return all_detections

    def _detect_at_scale(
        self, image: np.ndarray, scale: float
    ) -> List[Tuple[int, int, int, int]]:
        """
        Perform detection at a single scale.

        Args:
            image: Input image
            scale: Detection scale

        Returns:
            List of detections at this scale
        """
        detections = []
        window_size = int(self.config.window_size * scale)
        step_size = int(self.config.step_size * scale)

        # Sliding window over image
        for y in range(0, image.shape[0] - window_size + 1, step_size):
            for x in range(0, image.shape[1] - window_size + 1, step_size):

                # Extract and resize window
                window = image[y : y + window_size, x : x + window_size]
                resized_window = window[:: int(scale), :: int(scale)]

                # Ensure window is exactly 22x22
                if resized_window.shape != (
                    self.config.window_size,
                    self.config.window_size,
                ):
                    continue

                # Compute integral image
                integral_window = compute_integral_image(resized_window)

                # Classify window
                is_face, _ = CascadeClassifier.predict_window(
                    self.classifier, resized_window, integral_window, x, y
                )

                if is_face:
                    detections.append((x, y, window_size, window_size))

        return detections


class NonMaximumSuppression:
    """Handles Non-Maximum Suppression filtering."""

    @staticmethod
    def compute_iou(
        box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute Intersection over Union (IoU) of two bounding boxes.

        Args:
            box1, box2: Bounding boxes in format (x, y, width, height)

        Returns:
            IoU value between 0.0 and 1.0
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_left >= x_right or y_top >= y_bottom:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = (w1 * h1) + (w2 * h2) - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def filter_detections(
        detections: List[Tuple[int, int, int, int]], iou_threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """
        Apply Non-Maximum Suppression to filter overlapping detections.

        Args:
            detections: List of bounding boxes [(x, y, w, h), ...]
            iou_threshold: IoU threshold for suppression

        Returns:
            Filtered list of non-overlapping detections
        """
        if len(detections) <= 1:
            return detections

        # Sort by area (larger faces are typically better detections)
        detections_with_area = [
            (x * y * w * h, i, (x, y, w, h))
            for i, (x, y, w, h) in enumerate(detections)
        ]
        detections_with_area.sort(key=lambda x: x[0], reverse=True)

        kept_detections = []
        suppressed_indices = set()

        for area, idx, detection in detections_with_area:
            if idx in suppressed_indices:
                continue

            # Keep this detection
            kept_detections.append(detection)

            # Suppress overlapping detections
            for other_area, other_idx, other_detection in detections_with_area:
                if other_idx == idx or other_idx in suppressed_indices:
                    continue

                iou = NonMaximumSuppression.compute_iou(detection, other_detection)
                if iou > iou_threshold:
                    suppressed_indices.add(other_idx)

        return kept_detections


class ResultVisualizer:
    """Handles visualization of detection results."""

    @staticmethod
    def visualize_detections(
        image_path: str,
        detections: List[Tuple[int, int, int, int]],
        title: str = "Face Detection Results",
    ) -> None:
        """
        Visualize face detections on the original image.

        Args:
            image_path: Path to the original image
            detections: List of bounding boxes to visualize
            title: Plot title
        """
        # Load original image
        cwd = os.getcwd()
        image = load_image_as_array(os.path.join(cwd, image_path))

        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image, cmap="gray")

        # Draw bounding boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))

        for i, (x, y, w, h) in enumerate(detections):
            color = colors[i] if len(detections) > 1 else "red"

            # Draw rectangle
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                x,
                y - 5,
                f"Face {i+1}",
                color=color,
                fontweight="bold",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        ax.set_title(f"{title} - {len(detections)} faces detected", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        print(f"Visualization complete. {len(detections)} faces detected.")


# ============================================================================
# MAIN FACE DETECTION PIPELINE
# ============================================================================


class FaceDetectionPipeline:
    """Main pipeline for face detection."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.detector = MultiScaleDetector(config)

    def run_detection(self, image_key: str = None) -> List[Tuple[int, int, int, int]]:
        """
        Run the complete face detection pipeline.

        Args:
            image_key: Key for image selection (optional)

        Returns:
            List of final filtered detections
        """
        # Use provided image key or default from config
        if image_key:
            if image_key not in self.config.IMAGE_PATHS:
                raise ValueError(f"Unknown image key: {image_key}")
            image_path = self.config.IMAGE_PATHS[image_key]
        else:
            image_path = self.config.image_path

        print("=" * 60)
        print("FACE DETECTION PIPELINE")
        print("=" * 60)
        print(f"Image: {image_path}")
        print(f"Scales: {self.config.scales}")
        print(f"IoU threshold: {self.config.iou_threshold}")
        print("-" * 60)

        # Step 1: Multi-scale detection
        print("Step 1: Multi-scale sliding window detection...")
        raw_detections = self.detector.detect_faces(image_path)

        # Step 2: Non-Maximum Suppression
        print("Step 2: Applying Non-Maximum Suppression...")
        filtered_detections = NonMaximumSuppression.filter_detections(
            raw_detections, self.config.iou_threshold
        )
        print(f"Filtered detections: {len(filtered_detections)}")

        # Step 3: Visualization
        print("Step 3: Visualizing results...")
        ResultVisualizer.visualize_detections(image_path, filtered_detections)

        print("=" * 60)
        print("DETECTION COMPLETE")
        print("=" * 60)

        return filtered_detections


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main function to run face detection."""

    # Create configuration
    config = DetectionConfig(
        current_image="hard",  # Change this to test different images
        scales=[6.0, 7.0, 8.0],  # Adjust scales as needed
        iou_threshold=0.5,  # Adjust NMS threshold
        step_size=1,  # Adjust sliding window step
    )

    # Create and run pipeline
    pipeline = FaceDetectionPipeline(config)

    try:
        final_detections = pipeline.run_detection()
        print(f"\nFinal Results:")
        for i, (x, y, w, h) in enumerate(final_detections):
            print(f"  Face {i+1}: x={x}, y={y}, w={w}, h={h}")

    except Exception as e:
        print(f"Error during detection: {e}")
        raise


if __name__ == "__main__":
    main()
