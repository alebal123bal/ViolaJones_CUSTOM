# Viola-Jones Face Detection from Scratch

A complete implementation of the Viola-Jones face detection algorithm with a custom AdaBoost training pipeline.

![Demo](videos/demo_face_detection.gif)

## Features

- ✅ **Custom AdaBoost Implementation** - Built from scratch with a mathematical optimization trick that dramatically speeds up training
- ✅ **Haar Feature Generation** - Generates and evaluates thousands of Haar-like features
- ✅ **Integral Image Optimization** - Fast feature computation using integral images
- ✅ **Cascade Classifier** - Multi-stage cascade for efficient real-time detection
- ✅ **Complete Pipeline** - From dataset creation to inference

## Quick Start

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/alebal123bal/ViolaJones_CUSTOM.git
cd ViolaJones_CUSTOM

# Install dependencies
pip install numpy opencv-python pillow numba jupyter

# Run the demo notebook
jupyter notebook viola_jones_custom.ipynb
```

## What Makes This Special?

This project implements the complete Viola-Jones algorithm from the ground up, including a **custom AdaBoost trainer** that leverages a mathematical trick to significantly speed up the training process. While pre-trained classifiers are included in this repo (training not performed here), the AdaBoost implementation in `AdaBoost_smart/` demonstrates how to efficiently train weak classifiers on large feature sets.

The key optimization involves reformulating the weight update step to avoid redundant computations across iterations, making it feasible to train on datasets with thousands of features.

## Project Structure

```
ViolaJones_CUSTOM/
├── viola_jones_custom.ipynb        # Main demo notebook
├── basic_elements/                  # Core data structures
│   ├── haar_feature/               # Haar feature implementation
│   ├── rectangle/                  # Rectangle primitives
│   └── feature_gen/                # Feature generation logic
├── AdaBoost_smart/                 # Custom AdaBoost implementation
│   ├── classifiers/                # Weak classifier training
│   └── utils/                      # Numba-optimized utilities
├── image_manager/                  # Dataset handling
│   ├── dataset_creator/            # Face/non-face dataset creation
│   └── image_loader/               # Image loading utilities
├── matrix_creator/                 # Integral image & evaluation matrices
├── viola_jones/                    # Cascade classifier
│   ├── train/                      # Cascade training pipeline
│   └── inference/                  # Detection with sliding window
└── _pickle_folder/                 # Serialized trained models
```

## How It Works

1. **Feature Generation**: Generate ~160,000+ Haar-like features for 24×24 windows
2. **Integral Images**: Pre-compute integral images for O(1) feature evaluation
3. **AdaBoost Training**: Select the best features using custom AdaBoost (see `AdaBoost_smart/`)
4. **Cascade Construction**: Build a cascade of strong classifiers for fast rejection
5. **Detection**: Slide windows across images and run through the cascade

## Custom AdaBoost Implementation

The `AdaBoost_smart/` module contains the heart of this project - a custom implementation that:

- Trains weak classifiers on massive feature sets
- Uses weight vectorization for faster convergence
- Implements early stopping and convergence checks
- Provides detailed training statistics and metrics

Check out the [AdaBoost README](AdaBoost_smart/README.md) for implementation details.

## Usage Example

```python
# Load the trained cascade
from viola_jones.inference.window_slide import detect_faces

# Detect faces in an image
faces = detect_faces('path/to/image.jpg', scale_factor=1.25, min_neighbors=3)

# Faces will contain bounding boxes (x, y, width, height)
for (x, y, w, h) in faces:
    # Draw rectangle on image
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

## Dataset

Training requires:
- Face images (positive samples) in `image_manager/training_set/face/`
- Non-face images (negative samples) in `image_manager/training_set/not_face/`

The dataset creator modules can help build your training set from raw images.

## References

- Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*
- Freund, Y., & Schapire, R. E. (1997). *A Decision-Theoretic Generalization of On-Line Learning*

## License

MIT

## Acknowledgments

Built entirely from scratch as an educational deep-dive into classical computer vision techniques.