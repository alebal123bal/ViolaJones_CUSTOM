"""
Preprocess training set
"""

import os
import numpy as np
from PIL import Image
from image_manager.image_loader.image_loader import load_images_from_folder

FACE_PATH = "image_manager/training_set/face"
NOT_FACE_PATH = "image_manager/training_set/not_face"


# TODO


def add_random_images(n_images=1000):
    """
    Add random images to the non face training set.
    Format 22x22, grayscale, numpy array.

    Args:
        n_images (int): Number of random images to add.
    """

    not_face_path = NOT_FACE_PATH
    if not os.path.exists(not_face_path):
        os.makedirs(not_face_path)

    for i in range(n_images):
        random_image = np.random.randint(0, 256, (22, 22), dtype=np.uint8)
        image_path = os.path.join(not_face_path, f"random_image_{i}.png")
        Image.fromarray(random_image).save(image_path)


def remove_background_from_faces():
    """
    Zoom in from the face dataset.
    In particular, remove:
    - 5 pixels from left
    - 5 pixels from right
    - 2 pixels from top
    - 4 pixels from bottom

    Finally, reconvert to 22x22
    """
    face_path = FACE_PATH
    if not os.path.exists(face_path):
        print(f"Face path {face_path} does not exist.")
        return

    for filename in os.listdir(face_path):
        if not filename.lower().endswith((".jpg", ".jpeg")):
            continue

        image_path = os.path.join(face_path, filename)

        try:
            image = Image.open(image_path).convert("L")
            image_array = np.array(image, dtype=np.uint8)

            # Crop the image
            cropped_image = image_array[2:-4, 5:-5]

            # Resize to 22x22
            resized_image = Image.fromarray(cropped_image).resize(
                (22, 22), Image.Resampling.LANCZOS
            )

            # Save as PNG
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(face_path, png_filename)
            resized_image.save(png_path, "PNG")

            # Remove original JPG
            os.remove(image_path)

            print(f"Processed {filename} and saved as {png_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def create_hard_negatives():
    """
    Create hard negatives for the non-face training set.
    They are derived from the faces, with some transformations.
    """

    face_path = FACE_PATH
    not_face_path = NOT_FACE_PATH

    face = load_images_from_folder(face_path)[0]  # Load the first face image

    # Ensure face is 22x22
    if face.shape != (22, 22):
        face = np.array(
            Image.fromarray(face).resize((22, 22), Image.Resampling.LANCZOS)
        )

    print(f"Original face shape: {face.shape}")
    print(f"Total shifts will be: {8 * 8} = 64")
    print(f"Monochromatic images: 64 shifts × 52 colors = {64 * 52}")
    print(f"Random noise images: 64 shifts × 50 iterations = {64 * 50}")
    print(f"Total images: {64 * 52 + 64 * 50}")

    count = 0

    # Monochromatic padding
    for shift_x in (-5, -4, -3, -2, 2, 3, 4, 5):
        for shift_y in (-5, -4, -3, -2, 2, 3, 4, 5):
            for padding_value in range(0, 260, 5):  # 0, 5, 10, ..., 255 (52 values)
                # Create padded canvas
                canvas_width = 22 + abs(shift_x)
                canvas_height = 22 + abs(shift_y)
                padded_image = np.full(
                    (canvas_height, canvas_width), padding_value, dtype=np.uint8
                )

                # Calculate where to place the face on the canvas
                if shift_x >= 0:
                    face_x_start = shift_x
                    face_x_slice = slice(0, 22)
                else:
                    face_x_start = 0
                    face_x_slice = slice(abs(shift_x), 22)

                if shift_y >= 0:
                    face_y_start = shift_y
                    face_y_slice = slice(0, 22)
                else:
                    face_y_start = 0
                    face_y_slice = slice(abs(shift_y), 22)

                # Place the (potentially cropped) face
                face_portion = face[face_y_slice, face_x_slice]
                face_height, face_width = face_portion.shape

                padded_image[
                    face_y_start : face_y_start + face_height,
                    face_x_start : face_x_start + face_width,
                ] = face_portion

                # Resize back to 22x22
                resized_image = Image.fromarray(padded_image).resize(
                    (22, 22), Image.Resampling.LANCZOS
                )

                # Save as PNG
                image_name = (
                    f"mono_shift_{shift_x:+d}_{shift_y:+d}_pad_{padding_value:03d}.png"
                )
                image_path = os.path.join(not_face_path, image_name)
                resized_image.save(image_path, "PNG")
                count += 1

                if count % 100 == 0:
                    print(f"Saved {count} monochromatic images...")

    print(f"Completed monochromatic images: {count}")

    # Random noise padding
    for shift_x in (-5, -4, -3, -2, 2, 3, 4, 5):
        for shift_y in (-5, -4, -3, -2, 2, 3, 4, 5):
            for noise_iter in range(50):
                # Create padded canvas with random noise
                canvas_width = 22 + abs(shift_x)
                canvas_height = 22 + abs(shift_y)
                padded_image = np.random.randint(
                    0, 256, (canvas_height, canvas_width), dtype=np.uint8
                )

                # Calculate where to place the face (same logic as above)
                if shift_x >= 0:
                    face_x_start = shift_x
                    face_x_slice = slice(0, 22)
                else:
                    face_x_start = 0
                    face_x_slice = slice(abs(shift_x), 22)

                if shift_y >= 0:
                    face_y_start = shift_y
                    face_y_slice = slice(0, 22)
                else:
                    face_y_start = 0
                    face_y_slice = slice(abs(shift_y), 22)

                # Place the (potentially cropped) face
                face_portion = face[face_y_slice, face_x_slice]
                face_height, face_width = face_portion.shape

                padded_image[
                    face_y_start : face_y_start + face_height,
                    face_x_start : face_x_start + face_width,
                ] = face_portion

                # Resize back to 22x22
                resized_image = Image.fromarray(padded_image).resize(
                    (22, 22), Image.Resampling.LANCZOS
                )

                # Save as PNG with unique filename
                image_name = (
                    f"noise_shift_{shift_x:+d}_{shift_y:+d}_iter_{noise_iter:02d}.png"
                )
                image_path = os.path.join(not_face_path, image_name)
                resized_image.save(image_path, "PNG")
                count += 1

                if count % 100 == 0:
                    print(f"Saved {count} total images...")

    print(f"✅ Completed! Total images created: {count}")
    print(f"📁 Saved to: {not_face_path}")


if __name__ == "__main__":
    create_hard_negatives()
    pass
