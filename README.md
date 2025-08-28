# ViolaJones_CUSTOM: A Custom Implementation of the Viola-Jones Algorithm

This repository provides a custom implementation of the Viola-Jones algorithm for object detection. The project is structured to allow for both training a new cascade classifier and performing inference to detect objects in images. The implementation is primarily in Python and is demonstrated in a Jupyter Notebook.

## About the Viola-Jones Algorithm

The Viola-Jones algorithm, proposed by Paul Viola and Michael Jones in 2001, is a robust and real-time object detection framework. [[1]](https://www.mygreatlearning.com/blog/viola-jones-algorithm/) It was one of the first to achieve high detection rates in real-time, making it a landmark in the field of computer vision. [[2]](https://pub.towardsai.net/the-magic-behind-face-detection-a-deep-dive-into-the-viola-jones-algorithm-51f323f16cd2) The algorithm breaks down the complex problem of object detection into four main stages:

### 1. Haar-like Features

Instead of using the pixel intensities directly, the Viola-Jones algorithm uses Haar-like features. These are simple rectangular features that represent the difference in intensity between adjacent regions of an image. [[3]](https://www.baeldung.com/cs/viola-jones-algorithm) For example, in face detection, the region of the eyes is often darker than the upper cheeks. [[4]](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework) By calculating the difference between the sum of pixel values in the black and white rectangular regions, a feature value is obtained. [[3]](https://www.baeldung.com/cs/viola-jones-algorithm) There are several types of Haar-like features, including edge, line, and four-sided features, which are effective at detecting different patterns in an image. [[1]](https://www.mygreatlearning.com/blog/viola-jones-algorithm/)

### 2. Integral Images

A key innovation of the Viola-Jones algorithm is the use of integral images (also known as summed-area tables). [[5]](https://medium.com/@Andrew_D./computer-vision-viola-jones-object-detection-d2a609527b7c)[[6]](https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/) An integral image is a data structure where the value at any point (x, y) is the sum of all the pixels above and to the left of that point. [[7]](https://towardsdatascience.com/viola-jones-algorithm-and-haar-cascade-classifier-ee3bfb19f7d8/) This representation allows for the rapid calculation of the sum of pixels within any rectangular area in constant time, regardless of the size of the rectangle. [[8]](https://pmc.ncbi.nlm.nih.gov/articles/PMC4541907/) This dramatically speeds up the process of calculating the thousands of Haar-like feature values for each sub-window of an image. [[9]](https://levelup.gitconnected.com/the-integral-image-4df3df5dce35)

### 3. AdaBoost for Feature Selection

A 24x24 pixel sub-window has over 160,000 possible Haar-like features, most of which are irrelevant for detecting a specific object. [[2]](https://pub.towardsai.net/the-magic-behind-face-detection-a-deep-dive-into-the-viola-jones-algorithm-51f323f16cd2) The AdaBoost (Adaptive Boosting) algorithm is used to select a small set of the most effective features and to train the classifiers that use them. [[3]](https://www.baeldung.com/cs/viola-jones-algorithm) AdaBoost combines many "weak" classifiers, which are just slightly better than random guessing, to form a single "strong" classifier. [[10]](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)[[11]](https://www.ipol.im/pub/art/2014/104/article_lr.pdf) During training, AdaBoost iteratively selects the feature that best separates the positive and negative training examples. [[12]](https://medium.com/@rohan.chaudhury.rc/adaboost-classifier-for-face-detection-using-viola-jones-algorithm-30246527db11)

### 4. Cascade of Classifiers

The final stage of the algorithm is a cascade of classifiers, which is designed to quickly discard non-object regions in an image. [[13]](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) The cascade is a series of stages, where each stage is a strong classifier trained by AdaBoost. [[4]](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework) When an image sub-window is evaluated, it is first passed to the initial, simpler stages of the cascade. If a sub-window fails to pass any stage, it is immediately rejected as not containing the object. [[13]](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) This allows the detector to focus computational effort on more promising regions, enabling real-time performance. [[4]](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)

## Repository Structure

The repository is organized into several directories, each with a specific purpose:

*   `.vscode/`: Contains Visual Studio Code editor settings for the project.
*   `AdaBoost_smart/`: This directory  holds the implementation of the AdaBoost algorithm, used for selecting the best features and training the weak classifiers.
*   `basic_elements/`: This contains fundamental classes and data structures used throughout the project, such as representations for features and classifiers.
*   `image_manager/`: This directory is  responsible for handling image-related tasks, such as loading, converting to grayscale, and other preprocessing steps.
*   `matrix_creator/`: This directory's name suggests that it contains the code for creating integral images from the input images.
*   `viola_jones/`: This is  the core directory containing the main implementation of the Viola-Jones algorithm, including the cascade classifier structure.
*   `.gitignore`: A standard Git file for specifying which files and directories to ignore.
*   `.gitmodules`: This file is used to manage submodules if the project depends on other Git repositories.
*   `viola_jones_custom.ipynb`: A Jupyter Notebook that provides a step-by-step guide and demonstration of the custom Viola-Jones implementation. This is  the main file to run to see the project in action.

## How to Use

To get started with this project, you can follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alebal123bal/ViolaJones_CUSTOM.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd ViolaJones_CUSTOM
    ```
3.  **Install dependencies:** Ensure you have Python and Jupyter Notebook installed. You  also need to install common data science libraries such as NumPy and OpenCV.
4.  **Run the Jupyter Notebook:** Launch Jupyter Notebook and open the `viola_jones_custom.ipynb` file to see a demonstration of the training and inference process.

## Languages Used

*   **Jupyter Notebook:** 94.9%
*   **Python:** 5.1%

---
Learn more:
1. [Face Detection using Viola Jones Algorithm - Great Learning](https://www.mygreatlearning.com/blog/viola-jones-algorithm/)
2. [The Magic Behind Face Detection: A Deep Dive into the Viola-Jones Algorithm - Towards AI](https://pub.towardsai.net/the-magic-behind-face-detection-a-deep-dive-into-the-viola-jones-algorithm-51f323f16cd2)
3. [The Viola-Jones Algorithm | Baeldung on Computer Science](https://www.baeldung.com/cs/viola-jones-algorithm)
4. [Viola–Jones object detection framework - Wikipedia](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)
5. [Computer Vision: Viola-Jones Object Detection | by Andrew D - Medium](https://medium.com/@Andrew_D./computer-vision-viola-jones-object-detection-d2a609527b7c)
6. [Computer Vision – The Integral Image | Computer Science: Source - WordPress.com](https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/)
7. [Viola Jones Algorithm and Haar Cascade Classifier | Towards Data Science](https://towardsdatascience.com/viola-jones-algorithm-and-haar-cascade-classifier-ee3bfb19f7d8/)
8. [Integral Images: Efficient Algorithms for Their Computation and Storage in Resource-Constrained Embedded Vision Systems - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4541907/)
9. [The Integral Image. Introduction to Computer Vision, Part 3 | by BenMauss | Level Up Coding](https://levelup.gitconnected.com/the-integral-image-4df3df5dce35)
10. [Rapid Object Detection using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
11. [An Analysis of the Viola-Jones Face Detection Algorithm - IPOL Journal](https://www.ipol.im/pub/art/2014/104/article_lr.pdf)
12. [Adaboost classifier for face detection using viola jones algorithm | by Rohan Chaudhury](https://medium.com/@rohan.chaudhury.rc/adaboost-classifier-for-face-detection-using-viola-jones-algorithm-30246527db11)
13. [Cascade Classifier - OpenCV Documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)