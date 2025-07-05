"""
Train a Viola-Jones face detection model using the optimized AdaBoost.
"""

import time
from matrix_creator.feature_eval_matrix_maker import (
    create_matrix_weights_labels,
    generate_all_haar_features,
    load_images_from_folder,
    compute_integral_images,
    load_matrix_weights_labels,
    FACE_PATH,
    NOT_FACE_PATH,
)
from AdaBoost_smart.adaboost import AdaBoost, ClassifierScoreCheck

if __name__ == "__main__":
    start_time = time.time()

    # Load Haar features eliminating the corner features
    haar_features = generate_all_haar_features(x_start=4, y_start=2)

    # Load face images from the dataset folder and limit to 2000 images
    face_images = load_images_from_folder(FACE_PATH)[0:2000]

    # Load non-face images from the dataset folder and limit to 5000 images
    not_face_images = load_images_from_folder(NOT_FACE_PATH)[0:5000]

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
        n_stages=3,
    )

    # Train the classifier
    adaboost_classifier.train()

    end_time = time.time()
    print(f"✅ Training completed in {end_time - start_time:.2f} seconds.\n")

    # Reload the feature evaluation matrix and labels
    feature_matrix, _, labels = load_matrix_weights_labels()

    # Check the classifier score
    classifier_score = ClassifierScoreCheck(
        feature_eval_matrix=feature_matrix, sample_labels=labels
    )

    classifier_score.analyze()

    # Visualize the best features
    for stage in adaboost_classifier.trained_classifier:
        for feature_dict in stage:
            feature = haar_features[feature_dict["feature_idx"]]
            feature.plot(grayscale_image=face_images[0])
