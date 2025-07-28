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


def create_central_crop_hard_negatives():
    """
    Create hard negatives by cropping the central part of faces with variable frame sizes.
    This helps the detector learn that partial faces (missing edges) are not complete faces.

    The crops focus on the central region (eyes, nose, cheeks) which the detector
    might be over-relying on for classification.
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

    # Define crop sizes (these will be the central regions we keep)
    crop_sizes = [
        (16, 16),  # Keep 16x16 center, crop out 3px border all around
        (14, 14),  # Keep 14x14 center, crop out 4px border all around
        (12, 12),  # Keep 12x12 center, crop out 5px border all around
        (10, 10),  # Keep 10x10 center, crop out 6px border all around
        (18, 16),  # Keep 18x16 center (rectangular crops)
        (16, 18),  # Keep 16x18 center
        (20, 14),  # Keep 20x14 center (wide crop)
        (14, 20),  # Keep 14x20 center (tall crop)
    ]

    # Padding configurations
    mono_padding_values = list(range(0, 260, 10))  # 26 values: 0, 10, 20, ..., 250
    noise_iterations = 25  # Number of random noise variations per crop

    total_crops = len(crop_sizes)
    total_mono = total_crops * len(mono_padding_values)
    total_noise = total_crops * noise_iterations

    print(f"Crop sizes: {total_crops}")
    print(
        f"Monochromatic images: {total_crops} crops × {len(mono_padding_values)} colors = {total_mono}"
    )
    print(
        f"Random noise images: {total_crops} crops × {noise_iterations} iterations = {total_noise}"
    )
    print(f"Total images: {total_mono + total_noise}")

    count = 0

    # Monochromatic padding
    for crop_width, crop_height in crop_sizes:
        # Calculate crop coordinates (center crop)
        start_x = (22 - crop_width) // 2
        start_y = (22 - crop_height) // 2
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        # Extract central crop
        central_crop = face[start_y:end_y, start_x:end_x]

        for padding_value in mono_padding_values:
            # Create 22x22 canvas with padding
            padded_image = np.full((22, 22), padding_value, dtype=np.uint8)

            # Place the central crop back in the center
            padded_image[start_y:end_y, start_x:end_x] = central_crop

            # Convert to PIL and save
            resized_image = Image.fromarray(padded_image)

            # Save as PNG
            image_name = f"central_crop_{crop_width}x{crop_height}_mono_pad_{padding_value:03d}.png"
            image_path = os.path.join(not_face_path, image_name)
            resized_image.save(image_path, "PNG")
            count += 1

            if count % 50 == 0:
                print(f"Saved {count} central crop monochromatic images...")

    print(f"Completed central crop monochromatic images: {count}")

    # Random noise padding
    for crop_width, crop_height in crop_sizes:
        # Calculate crop coordinates (center crop)
        start_x = (22 - crop_width) // 2
        start_y = (22 - crop_height) // 2
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        # Extract central crop
        central_crop = face[start_y:end_y, start_x:end_x]

        for noise_iter in range(noise_iterations):
            # Create 22x22 canvas with random noise
            padded_image = np.random.randint(0, 256, (22, 22), dtype=np.uint8)

            # Place the central crop back in the center
            padded_image[start_y:end_y, start_x:end_x] = central_crop

            # Convert to PIL and save
            resized_image = Image.fromarray(padded_image)

            # Save as PNG with unique filename
            image_name = f"central_crop_{crop_width}x{crop_height}_noise_iter_{noise_iter:02d}.png"
            image_path = os.path.join(not_face_path, image_name)
            resized_image.save(image_path, "PNG")
            count += 1

            if count % 50 == 0:
                print(f"Saved {count} total central crop images...")

    print(f"✅ Completed central crop hard negatives! Total images created: {count}")
    print(f"📁 Saved to: {not_face_path}")


def create_advanced_central_crop_hard_negatives():
    """
    Advanced version with more sophisticated cropping patterns and gradual padding.
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

    # More sophisticated crop patterns
    crop_configs = [
        # (crop_width, crop_height, offset_x, offset_y, description)
        (16, 16, 0, 0, "center_large"),
        (14, 14, 0, 0, "center_medium"),
        (12, 12, 0, 0, "center_small"),
        (10, 10, 0, 0, "center_tiny"),
        # Slightly off-center crops (eyes region focus)
        (16, 12, 0, -2, "eyes_focus_wide"),
        (14, 10, 0, -3, "eyes_focus_narrow"),
        (12, 14, 0, 1, "nose_mouth_focus"),
        # Asymmetric crops
        (18, 10, -1, -2, "upper_face_left"),
        (18, 10, 1, -2, "upper_face_right"),
        (10, 18, 0, 0, "vertical_strip"),
        (18, 8, 0, -3, "horizontal_strip_upper"),
        (18, 8, 0, 3, "horizontal_strip_lower"),
    ]

    total_crops = len(crop_configs)
    mono_values = 20  # Fewer padding values but more crop variations
    noise_iterations = 15

    total_mono = total_crops * mono_values
    total_noise = total_crops * noise_iterations

    print(f"Advanced crop configs: {total_crops}")
    print(
        f"Monochromatic images: {total_crops} crops × {mono_values} colors = {total_mono}"
    )
    print(
        f"Random noise images: {total_crops} crops × {noise_iterations} iterations = {total_noise}"
    )
    print(f"Total images: {total_mono + total_noise}")

    count = 0

    # Gradual padding values (creating smoother transitions)
    padding_values = [int(x) for x in np.linspace(0, 255, mono_values)]

    # Monochromatic padding with gradual transitions
    for crop_width, crop_height, offset_x, offset_y, description in crop_configs:
        # Calculate crop coordinates with offset
        center_x, center_y = 11, 11  # Center of 22x22 image
        start_x = max(0, min(22 - crop_width, center_x - crop_width // 2 + offset_x))
        start_y = max(0, min(22 - crop_height, center_y - crop_height // 2 + offset_y))
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        # Extract crop
        central_crop = face[start_y:end_y, start_x:end_x]

        for padding_value in padding_values:
            # Create 22x22 canvas with padding
            padded_image = np.full((22, 22), padding_value, dtype=np.uint8)

            # Place the crop
            padded_image[start_y:end_y, start_x:end_x] = central_crop

            # Convert to PIL and save
            resized_image = Image.fromarray(padded_image)

            # Save as PNG
            image_name = f"adv_crop_{description}_mono_{padding_value:03d}.png"
            image_path = os.path.join(not_face_path, image_name)
            resized_image.save(image_path, "PNG")
            count += 1

            if count % 50 == 0:
                print(f"Saved {count} advanced crop monochromatic images...")

    print(f"Completed advanced crop monochromatic images: {count}")

    # Random noise padding
    for crop_width, crop_height, offset_x, offset_y, description in crop_configs:
        # Calculate crop coordinates with offset
        center_x, center_y = 11, 11
        start_x = max(0, min(22 - crop_width, center_x - crop_width // 2 + offset_x))
        start_y = max(0, min(22 - crop_height, center_y - crop_height // 2 + offset_y))
        end_x = start_x + crop_width
        end_y = start_y + crop_height

        # Extract crop
        central_crop = face[start_y:end_y, start_x:end_x]

        for noise_iter in range(noise_iterations):
            # Create 22x22 canvas with random noise
            padded_image = np.random.randint(0, 256, (22, 22), dtype=np.uint8)

            # Place the crop
            padded_image[start_y:end_y, start_x:end_x] = central_crop

            # Convert to PIL and save
            resized_image = Image.fromarray(padded_image)

            # Save as PNG
            image_name = f"adv_crop_{description}_noise_{noise_iter:02d}.png"
            image_path = os.path.join(not_face_path, image_name)
            resized_image.save(image_path, "PNG")
            count += 1

            if count % 50 == 0:
                print(f"Saved {count} total advanced crop images...")

    print(
        f"✅ Completed advanced central crop hard negatives! Total images created: {count}"
    )
    print(f"📁 Saved to: {not_face_path}")


if __name__ == "__main__":
    pass
