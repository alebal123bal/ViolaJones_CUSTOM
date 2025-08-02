"""
Load the test image from the specified path, perform window sliding to detect faces using
 the trained classifier, and visualize the results.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from image_manager.image_loader.image_loader import (
    load_image_as_array,
    compute_integral_image,
)

from AdaBoost_smart.adaboost import load_pickle_obj


# IMAGE_PATH = "viola_jones/inference/test_image/easy.png"
IMAGE_PATH = "viola_jones/inference/test_image/medium.png"
# IMAGE_PATH = "viola_jones/inference/test_image/medium_1.png"
# IMAGE_PATH = "viola_jones/inference/test_image/medium_2.png"
IMAGE_PATH = "viola_jones/inference/test_image/hard.jpg"
# IMAGE_PATH = "viola_jones/inference/test_image/very_hard.jpg"
CLASSIFIER_PATH = "_pickle_folder/full_trained_classifier.pkl"

# Training sets
FACE_PATH = "image_manager/training_set/face"
NOT_FACE_PATH = "image_manager/training_set/not_face"


def visualize_detections(image, detections):
    """Visualize the final detections on the image."""

    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image, cmap="gray")

    for x, y, size in detections:
        # Pick a random color
        color = np.random.rand(
            3,
        )
        rect = patches.Rectangle(
            (x, y), size, size, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x, y, "Face", color=color, fontweight="bold")

    ax.set_title(f"Detected Faces: {len(detections)}")
    plt.show()


def cascade_prediction(
    classifier,
    grayscale_image,
    integral_image,
    x,
    y,
):
    """
    Perform cascade prediction using the trained classifier on the integral image.

    Args:
        classifier: The trained classifier.
        grayscale_image: The grayscale version of the input image.
        integral_image: The integral image of the input image.

    Returns:
        bool: True if a face is detected, False otherwise.
    """

    voted_sum_arr = []

    for s_idx, stage in enumerate(classifier):
        voted_sum = 0.0
        stage_features = stage["stage_features"]
        for feature_dict in stage_features:
            # Extract feature
            feature = feature_dict["feature"]

            # Show the Haar feature being evaluated
            # feature.plot(grayscale_image=grayscale_image)

            # Evaluate the Haar feature on the integral image
            feature_value = feature.evaluate(integral_image)

            # Check if the feature value meets the threshold
            if feature_dict["direction"] == 0:  # leq
                if feature_value <= feature_dict["threshold"]:
                    voted_sum += feature_dict["alpha"]
                else:
                    voted_sum -= feature_dict["alpha"]
            elif feature_dict["direction"] == 1:  # gt
                if feature_value > feature_dict["threshold"]:
                    voted_sum += feature_dict["alpha"]
                else:
                    voted_sum -= feature_dict["alpha"]

        # Append the voted sum for this stage
        voted_sum_arr.append(voted_sum)

        # If the sum of votes is positive, classify as a face
        if voted_sum < stage["stage_thre"]:
            # Quick rejection of this stage
            return False, None

    # At this point, all stages have been passed
    # Format to 2 decimal places for better readability
    voted_sum_arr = np.round(np.array(voted_sum_arr), 2)
    print(f"Face detected at {x}, {y} with voted sums:\n {voted_sum_arr}")

    return True, voted_sum_arr


def window_slide(image_path: str, classifier_path: str):
    """
    Perform window sliding on the input image to detect faces using the trained classifier.

    Args:
        image_path (str): Path to the input image.
        classifier_path (str): Path to the trained classifier.

    Returns:
        None
    """

    # Get cwd
    cwd = os.getcwd()

    # Load the test image as an array
    test_image = load_image_as_array(os.path.join(cwd, image_path))

    # Load the trained classifier
    adaboost_classifier = load_pickle_obj(os.path.join(cwd, classifier_path))

    # Preallocate a list to store the coordinates of detected faces
    detected_faces = []

    step = 2

    for scale in [
        1.0,
        # 2.0,
        # 4.0,
        # 5.0,
        # 8.0,
        # 9.0,
        # 10.0,
    ]:
        all_voted_sums = []

        # Perform window sliding to detect faces
        for y in range(0, test_image.shape[0] - int(22 * scale) + 1, step):
            for x in range(0, test_image.shape[1] - int(22 * scale) + 1, step):
                # Extract the scaled window from the image
                window = test_image[y : y + int(22 * scale), x : x + int(22 * scale)]

                # Compress the window to 22x22 by interpolating
                window = window[:: int(scale), :: int(scale)]

                # Compute the integral image of the window
                integral_window = compute_integral_image(window)

                # Classify the window using the trained classifier
                is_face, voted_sum_arr = cascade_prediction(
                    classifier=adaboost_classifier,
                    grayscale_image=window,
                    integral_image=integral_window,
                    x=x,
                    y=y,
                )

                # Store the voted sums for analysis
                all_voted_sums.append(
                    {
                        "x": x,
                        "y": y,
                        "voted_sum_arr": voted_sum_arr,
                    }
                )

                if is_face:
                    #     # Store the coordinates of the detected face
                    detected_faces.append((x, y, int(22 * scale)))

    print("Window sliding completed.")

    # Filter where the voted sums are not None
    all_voted_sums = [
        elem for elem in all_voted_sums if elem["voted_sum_arr"] is not None
    ]

    # Discard those voted sums where the first 3 elements are not all >=1
    good_pred = [
        elem
        for elem in all_voted_sums
        if np.all(np.array(elem["voted_sum_arr"][0:3]) >= 1.9)
    ]

    print("Performing non maximum suppression...")

    # Perform non-maximum suppression on the detected faces
    detected_faces = non_maximum_suppression(detected_faces)

    visualize_detections(test_image, detected_faces)

    return -1


def non_maximum_suppression(detected_faces):
    """
    Perform non-maximum suppression on the detected faces to remove overlapping detections.
    Need at least min_number detections to consider it a valid detection.
    Basically check the top left corner, and if the distance between two
    detections is less than 2 pixels, keep it and remove the other one.
    At the moment every detection window is fixed at 22x22 pixels.
    The detected_faces is a list of tuples (x, y) indicating the top left corner of the detection window.
    If the detected is alone, discard it as most likely it is a false positive.
    """

    MIN_DISTANCE = (
        3  # Minimum distance between two detections to consider them separate
    )

    # First check for the isolated faces and discard them, as they are false positives
    for i in range(len(detected_faces) - 1, -1, -1):
        x1, y1, size1 = detected_faces[i]
        isolated = True

        for j in range(len(detected_faces)):
            if i == j:
                continue
            x2, y2, size2 = detected_faces[j]

            # Check if the distance between the two detections is less than 2 pixels
            if abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE:
                isolated = False
                break

        if isolated:
            detected_faces.pop(i)

    # Sort the detected faces by their coordinates
    detected_faces = sorted(detected_faces, key=lambda x: (x[0], x[1]))

    # Initialize a list to store the final detections after non-maximum suppression
    final_detections = []

    # Iterate through the detected faces and apply non-maximum suppression
    for i in range(len(detected_faces)):
        x1, y1, size1 = detected_faces[i]
        keep = True

        for j in range(i + 1, len(detected_faces)):
            x2, y2, size2 = detected_faces[j]

            # Check if the distance between the two detections is less than 2 pixels
            if abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE:
                keep = False
                break

        if keep:
            final_detections.append((x1, y1, size1))

    return final_detections


if __name__ == "__main__":
    # Perform window sliding on the test image
    window_slide(IMAGE_PATH, CLASSIFIER_PATH)
