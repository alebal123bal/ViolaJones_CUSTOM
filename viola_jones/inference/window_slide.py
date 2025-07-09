"""
Load the test image from the specified path, perform window sliding to detect faces using
 the trained classifier, and visualize the results.
"""

import os

import matplotlib.pyplot as plt

from image_manager.image_loader.image_loader import (
    load_image_as_array,
    compute_integral_image,
)

from AdaBoost_smart.adaboost import load_pickle_obj

IMAGE_PATH = "viola_jones/inference/test_image/scene.jpg"
CLASSIFIER_PATH = "_pickle_folder/full_trained_classifier.pkl"


def cascade_prediction(classifier, grayscale_image, integral_image):
    """
    Perform cascade prediction using the trained classifier on the integral image.

    Args:
        classifier: The trained classifier.
        grayscale_image: The grayscale version of the input image.
        integral_image: The integral image of the input image.

    Returns:
        None
    """
    for stage in classifier:
        voted_sum = 0.0
        for feature_dict in stage:
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

        # If the sum of votes is positive, classify as a face
        if voted_sum < 0.0:
            # Quick rejection of this stage
            return False

    # At this point, all stages have been passed
    print("Face detected!")
    plt.imshow(grayscale_image, cmap="gray")
    # Wait for user input to close the plot
    plt.show()

    return True


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

    # TODO: find a way to intelligently do non-maximum suppression

    # Perform window sliding to detect faces
    for y in range(0, test_image.shape[0] - 22, 3):
        for x in range(0, test_image.shape[1] - 22, 3):
            # Extract the window from the image
            window = test_image[y : y + 22, x : x + 22]

            # Compute the integral image of the window
            integral_window = compute_integral_image(window)

            # Classify the window using the trained classifier
            _ = cascade_prediction(
                classifier=adaboost_classifier,
                grayscale_image=window,
                integral_image=integral_window,
            )

    print("Window sliding completed.")


if __name__ == "__main__":
    # Perform window sliding on the test image
    window_slide(IMAGE_PATH, CLASSIFIER_PATH)
