"""
Train a Viola-Jones face detection model using the optimized AdaBoost.
"""

import time
from matrix_creator.feature_eval_matrix_maker import (
    create,
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

    N = 100
    CLIP_TEST = False  # Set to True to limit the number of images for testing

    # Load Haar features
    haar_features = generate_all_haar_features()
    if CLIP_TEST:
        haar_features = haar_features[0 : 24 * N]

    # Load face images from the dataset folder
    face_images = load_images_from_folder(FACE_PATH)[0 : 10 * N]
    if CLIP_TEST:
        face_images = face_images[0 : 10 * N]

    # Load non-face images from the dataset folder
    not_face_images = load_images_from_folder(NOT_FACE_PATH)[0 : 20 * N]
    if CLIP_TEST:
        not_face_images = not_face_images[0 : 20 * N]

    integral_images = compute_integral_images(face_images + not_face_images)

    # Create the feature evaluation matrix, weights, and labels
    create(
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
