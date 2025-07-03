"""
Train a Viola-Jones face detection model using the optimized AdaBoost.
"""

import time
from matrix_creator.feature_eval_matrix_maker import (
    create,
    load_matrix_weights_labels,
    generate_all_haar_features,
    load_images_from_folder,
    compute_integral_images,
    FACE_PATH,
    NOT_FACE_PATH,
)
from AdaBoost_smart.adaboost import AdaBoost

if __name__ == "__main__":
    start_time = time.time()

    N = 2

    # Load Haar features
    haar_features = generate_all_haar_features()[0 : 12 * N]

    # Load images from the dataset folders
    face_images = load_images_from_folder(FACE_PATH)[0:10]
    not_face_images = load_images_from_folder(NOT_FACE_PATH)[0:10]
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
    print(f"Training completed in {end_time - start_time:.2f} seconds.\n")

    # Reload the matrix, weights, labels as the training clipped them
    feature_matrix, weights, labels = load_matrix_weights_labels()

    # Perform cascade prediction
    final_predictions = adaboost_classifier.cascade_predictions(
        matrix=feature_matrix,
        weights=weights,
        labels=labels,
    )
