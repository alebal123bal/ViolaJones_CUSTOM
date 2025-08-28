"""
Train a Viola-Jones face detection model using the optimized AdaBoost.
"""

import time

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
from viola_jones.train.post_train import enrich_classifier


def train_model():
    """
    Train a Viola-Jones face detection model using the optimized AdaBoost.
    """

    start_time = time.time()

    # Generate Haar features
    haar_features_0 = generate_all_haar_features(
        feature_types=[
            "horizontal_2",
            "vertical_2",
            "diagonal_4",
        ],
        x_padding=1,
        y_padding=1,
        step=5,
    )

    haar_features_1 = generate_all_haar_features(
        feature_types=[
            "horizontal_2",
            "vertical_2",
            "horizontal_3",
            "vertical_3",
            # "diagonal_4",
        ],
        x_padding=2,
        y_padding=2,
        step=3,
    )

    haar_features_2 = generate_all_haar_features(
        feature_types=[
            "horizontal_2",
            "vertical_2",
            "horizontal_3",
            "vertical_3",
            # "diagonal_4",
        ],
        x_padding=5,
        y_padding=5,
        step=2,
    )

    haar_features = haar_features_0 + haar_features_1 + haar_features_2

    # Load face images from the dataset folder
    face_images = load_images_from_folder(FACE_PATH)

    # Load non-face images from the dataset folder
    not_face_images = load_images_from_folder(NOT_FACE_PATH)

    # Compute integral images for both face and non-face images
    integral_images = compute_integral_images(face_images + not_face_images)

    # Create the feature evaluation matrix, weights, and labels
    create_matrix_weights_labels(
        haar_features=haar_features,
        face_images=face_images,
        not_face_images=not_face_images,
    )

    # Load the feature evaluation matrix and corresponding weights and labels
    feature_matrix, weights, labels = load_matrix_weights_labels()

    # Create an AdaBoost classifier with the loaded feature matrix
    adaboost_classifier = AdaBoostTrainer(
        feature_eval_matrix=feature_matrix,
        sample_weights=weights,
        sample_labels=labels,
        n_stages=5,
        aggressivness=0.4,
        feature_per_stage=4,
    )

    # Train the classifier
    adaboost_classifier.train()

    end_time = time.time()
    print(f"⏱️ Training completed in {end_time - start_time:.2f} seconds.\n")

    # Reload the feature evaluation matrix and labels
    feature_matrix, _, labels = load_matrix_weights_labels()

    # Check the classifier score
    classifier_score = ClassifierScoreCheck(
        feature_eval_matrix=feature_matrix, sample_labels=labels
    )

    classifier_score.analyze()

    # Enrich the classifier with stage thresholds
    enrich_classifier(
        haar_features=haar_features,
        face_images=face_images,
        std_devs=1.75,
    )

    # Load the classifier for visualization
    trained_classifier = PickleUtils.load_pickle_obj(
        "_pickle_folder/full_trained_classifier.pkl"
    )

    # Visualize the top 4 best features
    i = 0
    for stage in trained_classifier:
        if i >= 4:
            break
        stage_features = stage["stage_features"]
        for feature_dict in stage_features:
            feature = feature_dict["feature"]
            feature.plot(grayscale_image=face_images[1])
            i += 1


if __name__ == "__main__":
    train_model()
