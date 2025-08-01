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
from AdaBoost_smart.adaboost import (
    AdaBoost,
    ClassifierScoreCheck,
    save_pickle_obj,
    load_pickle_obj,
)

if __name__ == "__main__":
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

    # Load face images from the dataset folder and limit to a images
    face_images = load_images_from_folder(FACE_PATH)

    # Load non-face images from the dataset folder and limit to b images
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
    adaboost_classifier = AdaBoost(
        feature_eval_matrix=feature_matrix,
        sample_weights=weights,
        sample_labels=labels,
        n_stages=8,
        aggressivness=0.4,
        feature_per_stage=3,
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

    # Save the trained classifier with the feature objects too
    my_classifier = []

    for stage in load_pickle_obj("_pickle_folder/trained_classifier.pkl"):
        stage_haar_features = []
        for feature_dict in stage:
            # Retrieve the Haar feature object using the index
            feature = haar_features[feature_dict["feature_idx"]]

            # Append the feature information along with the feature object
            stage_haar_features.append(
                {
                    "feature_idx": feature_dict["feature_idx"],
                    "threshold": feature_dict["threshold"],
                    "direction": feature_dict["direction"],
                    "error": feature_dict["error"],
                    "alpha": feature_dict["alpha"],
                    "feature": feature,
                }
            )
        my_classifier.append(stage_haar_features)

    save_pickle_obj(
        my_classifier,
        "full_trained_classifier.pkl",
    )

    # Visualize the best features
    for stage in load_pickle_obj("_pickle_folder/full_trained_classifier.pkl"):
        for feature_dict in stage:
            feature = feature_dict["feature"]
            feature.plot(grayscale_image=face_images[1])
