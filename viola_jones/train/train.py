"""
Train a Viola-Jones face detection model using the optimized AdaBoost.
"""

import time
from matrix_creator.feature_eval_matrix_maker import create, load_matrix_weights_labels
from AdaBoost_smart.adaboost import AdaBoost

if __name__ == "__main__":
    start_time = time.time()

    # Create the feature evaluation matrix, weights, and labels
    create()

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
