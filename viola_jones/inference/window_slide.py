"""
This module provides a function to slide a window over an image
"""

from adaboost import load_pickle_obj
from feature_gen.feature_gen import generate_all_haar_features

all_features = generate_all_haar_features()

best_classifier = load_pickle_obj("_pickle_folder/trained_classifier.pkl")

for s, stage in enumerate(best_classifier):
    print(f"Stage: {s}")
    for feature in stage:
        print(
            f"Feature: {feature['feature_idx']}, Threshold: {feature['threshold']}, Direction: {feature['direction']}, alpha: {feature['alpha']}"
        )
        # all_features[feature["feature_idx"]].plot()

# full image
IMAGE = "_full_image/free_him.jpeg"

# Load image in greyscale, resize by 1/9, compute integral
import cv2


def load_resized(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 9 * 2))
    print(f"Image shape: {img.shape}")
    # cv2.imshow("Original Image", img)
    # cv2.waitKey(0)
    # Save original image
    cv2.imwrite("_full_image/resized_image.jpeg", img)
    return img


resized_image = load_resized(IMAGE)


def slide_window(resized_image, window_size=(22, 22), step_size=5):
    """
    Slide a window over the image and yield each window.

    Args:
        resized_image: The image to slide the window over
        window_size: Size of the sliding window (width, height)
        step_size: Step size for sliding the window
    """
    h, w = resized_image.shape
    win_w, win_h = window_size

    for y in range(0, h - win_h + 1, step_size * 2):
        for x in range(0, w - win_w + 1, step_size):
            window = resized_image[y : y + win_h, x : x + win_w]
            # Show the window for debugging
            cv2.imshow("Window", window)
            cv2.waitKey(30)  # Display each window briefly
            # Calculate the integral image for the window
            integral_window = cv2.integral(window)
            is_face = apply_cascade(integral_window, best_classifier)
            if is_face:
                print(f"Face detected at ({x}, {y})")


def apply_cascade(window, classifier):
    """
    Apply the cascade classifier to a window.

    Args:
        window: The image window to classify
        classifier: The trained classifier

    Returns:
        True if the window is classified as a face, False otherwise
    """
    votes = []
    for stage in classifier:
        weighted_vote = 0.0
        for feature in stage:
            haar_feature = all_features[feature["feature_idx"]]
            evaluation = haar_feature.evaluate(window)
            if feature["direction"] == 0:  # Lower equal
                if evaluation <= feature["threshold"]:
                    weighted_vote += feature["alpha"]
                else:
                    weighted_vote -= feature["alpha"]
            else:  # Greater than
                if evaluation > feature["threshold"]:
                    weighted_vote += feature["alpha"]
                else:
                    weighted_vote -= feature["alpha"]

        if weighted_vote > 0:
            votes.append(1)
        else:
            return False  # Rejected by this stage
    return True  # Accepted by all stages


# for i in range(100):
#     # Load some faces from the original dataset
#     my_image = cv2.imread(
#         f"_training_set_harder/not_face/facelike_{i:04d}.png",
#         cv2.IMREAD_GRAYSCALE,
#         # f"_training_set_harder/face/person_{i:04d}.jpg",
#         # cv2.IMREAD_GRAYSCALE,
#     )

#     # Image is already a 22x22 window
#     # Compute integral image for the window
#     integral_my_image = cv2.integral(my_image)

#     # Apply the cascade classifier to the window
#     is_face = apply_cascade(integral_my_image, best_classifier)

#     print(f"Is the image a face? {'Yes' if is_face else 'No'}")


# Test with this other image
# my_image = cv2.imread("_full_image/person_0052.jpg", cv2.IMREAD_GRAYSCALE)

slide_window(resized_image, window_size=(22, 22), step_size=2)
# slide_window(my_image, window_size=(22, 22), step_size=2)
