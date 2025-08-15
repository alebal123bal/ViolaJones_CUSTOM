"""
Find best stage thresholds for cascade prediction using a trained classifier.
"""

# pylint: disable=unused-import, unused-argument,unused-variable, too-many-locals, too-many-statements

import numpy as np

from matrix_creator.feature_eval_matrix_maker import (
    create_matrix_weights_labels,
    generate_all_haar_features,
    # generate_all_full_coverage_haar_features,
    load_images_from_folder,
    compute_integral_images,
    load_matrix_weights_labels,
    FACE_PATH,
    NOT_FACE_PATH,
)

from AdaBoost_smart.classifiers.adaboost_trainer import AdaBoostTrainer
from AdaBoost_smart.classifiers.classifier_score_check import ClassifierScoreCheck
from AdaBoost_smart.utils.io_operations import PickleUtils


def cascade_prediction(
    classifier,
    grayscale_image,
    integral_image,
):
    """
    Perform cascade prediction using the trained classifier on the integral image.
    """

    voted_sum_arr = []

    for stage in classifier:
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

        # Assume a majority voting, so a zero threshold, as the AdaBoost
        # module is trained with a zero threshold.
        if voted_sum < 0.0:
            # Quick rejection of this stage
            return False, None

    return True, voted_sum_arr


def enrich_classifier(
    face_images=None,
    classifier=None,
    std_devs=1.5,
):
    """
    Enrich the classifier with the optimal stage thresholds.
    """
    print("\nEnriching classifier with optimal stage thresholds...")

    # Load face images if not provided
    if face_images is None:
        face_images = load_images_from_folder(FACE_PATH)

    # Compute integral images for face images
    integral_images = compute_integral_images(face_images)

    # Load the classifier if not provided
    if classifier is None:
        # Load the trained classifier from a pickle file
        classifier = PickleUtils.load_pickle_obj(
            "_pickle_folder/full_trained_classifier.pkl"
        )

    predictions = []
    for grayscale_image, integral_image in zip(face_images, integral_images):
        # Perform cascade prediction
        is_face, voted_sum_arr = cascade_prediction(
            classifier=classifier,
            grayscale_image=grayscale_image,
            integral_image=integral_image,
        )
        predictions.append(voted_sum_arr)

    # Clean the None values
    predictions = np.array([p for p in predictions if p is not None])

    # Optimal stage thresholds
    means = np.mean(predictions, axis=0)
    stds = np.std(predictions, axis=0)

    optim_stage_thre = means - std_devs * stds

    print(f"Optimal stage thresholds: {optim_stage_thre}")

    # Append them to the classifier
    for stage_idx, stage in enumerate(classifier):
        stage["stage_thre"] = optim_stage_thre[stage_idx]

    print("Classifier enriched with optimal stage thresholds.\n")

    # Save the updated classifier
    PickleUtils.save_pickle_obj(classifier, "full_trained_classifier.pkl")


if __name__ == "__main__":
    enrich_classifier()
