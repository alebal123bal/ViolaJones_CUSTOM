"""
Create nonface dataset.
"""

import os
import numpy as np
from PIL import Image
from image_manager.image_loader.image_loader import load_images_from_folder

FACE_PATH = "image_manager/training_set/face"
NOT_FACE_PATH = "image_manager/training_set/not_face"


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


def create_face_like_objects():
    """
    Create objects that have face-like patterns but are definitely not faces.
    Target: ~2000 samples
    """
    not_face_path = NOT_FACE_PATH
    count = 0

    def create_face_like_object(obj_type, variation):
        image = np.zeros((22, 22), dtype=np.uint8)

        if obj_type == "car_front":
            # Car with headlights, grille, license plate
            # Headlights (eye-like)
            image[6:10, 4:8] = 200  # Left headlight
            image[6:10, 14:18] = 200  # Right headlight
            # Grille (nose-like)
            image[10:14, 8:14] = 120
            # License plate (mouth-like)
            image[16:19, 6:16] = 180
            # Car outline
            image[4:20, 2:20] = np.maximum(image[4:20, 2:20], 80)

        elif obj_type == "house_front":
            # House with windows and door
            # Windows (eye-like)
            image[6:10, 5:9] = 150  # Left window
            image[6:10, 13:17] = 150  # Right window
            # Door (mouth-like)
            image[14:20, 9:13] = 100
            # Roof triangle
            for i in range(11):
                image[i, 11 - i : 11 + i + 1] = 120

        elif obj_type == "electrical_outlet":
            # Electrical outlet with two holes and ground
            image[7:11, 6:10] = 0  # Left socket (dark)
            image[7:11, 12:16] = 0  # Right socket (dark)
            image[15:17, 10:12] = 0  # Ground hole
            image[5:18, 4:18] = 200  # Outlet plate

        elif obj_type == "clock":
            # Clock face with hour markers
            center_y, center_x = 11, 11
            radius = 8
            # Clock face
            y, x = np.mgrid[0:22, 0:22]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
            image[mask] = 220
            # 12, 3, 6, 9 o'clock markers (face-like pattern)
            image[3, 11] = 0  # 12 o'clock
            image[11, 19] = 0  # 3 o'clock
            image[19, 11] = 0  # 6 o'clock
            image[11, 3] = 0  # 9 o'clock
            # Clock hands
            image[11, 6:11] = 0  # Hour hand
            image[6:11, 11] = 0  # Minute hand

        elif obj_type == "butterfly":
            # Butterfly with symmetric wing patterns
            # Body
            image[8:14, 10:12] = 100
            # Wings (symmetric eye-like patterns)
            image[6:12, 4:10] = 180  # Left wing
            image[6:12, 12:18] = 180  # Right wing
            image[8:10, 6:8] = 50  # Wing spots
            image[8:10, 14:16] = 50

        elif obj_type == "owl_toy":
            # Toy owl with large round eyes
            # Large eyes
            y, x = np.mgrid[0:22, 0:22]
            left_eye = (x - 7) ** 2 + (y - 8) ** 2 <= 9
            right_eye = (x - 15) ** 2 + (y - 8) ** 2 <= 9
            image[left_eye] = 200
            image[right_eye] = 200
            # Eye pupils
            image[7:10, 6:8] = 0
            image[7:10, 14:16] = 0
            # Beak
            image[12:15, 10:12] = 150

        elif obj_type == "robot_head":
            # Robot with LED eyes and speaker grille
            # Head outline
            image[3:19, 3:19] = 120
            # LED eyes
            image[7:10, 6:9] = 255
            image[7:10, 13:16] = 255
            # Speaker grille (mouth area)
            for i in range(14, 18):
                image[i, 6:16] = 80 if i % 2 == 0 else 160

        elif obj_type == "traffic_light":
            # Traffic light with three circles
            image[4:18, 8:14] = 100  # Housing
            image[5:8, 9:13] = 255  # Top light
            image[9:12, 9:13] = 255  # Middle light
            image[13:16, 9:13] = 255  # Bottom light

        elif obj_type == "emoticon":
            # Simple emoticon drawn pattern
            # Face circle
            y, x = np.mgrid[0:22, 0:22]
            face = (x - 11) ** 2 + (y - 11) ** 2 <= 81
            image[face] = 200
            # Eyes
            image[8:10, 8:10] = 0
            image[8:10, 12:14] = 0
            # Smile
            image[14:16, 9:13] = 0

        # Add noise variation
        if variation > 0:
            noise = np.random.normal(0, variation * 10, (22, 22))
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

        return image

    print("Creating face-like objects (target: ~2000)...")

    object_types = [
        "car_front",
        "house_front",
        "electrical_outlet",
        "clock",
        "butterfly",
        "owl_toy",
        "robot_head",
        "traffic_light",
        "emoticon",
    ]

    for obj_type in object_types:
        for variation in range(25):  # 25 variations per object type
            for noise_level in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:  # 10 noise levels
                image = create_face_like_object(obj_type, noise_level)

                image_name = (
                    f"object_{obj_type}_var{variation:02d}_noise{noise_level}.png"
                )
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(image).save(image_path, "PNG")
                count += 1

                if count % 200 == 0:
                    print(f"Created {count} face-like objects...")

    print(f"✅ Created {count} face-like object negatives")
    return count


def create_partial_face_fragments():
    """
    Create fragments that contain parts of faces but are incomplete.
    Target: ~1500 samples
    """
    face_path = FACE_PATH
    not_face_path = NOT_FACE_PATH

    faces = load_images_from_folder(face_path)[:10]
    faces = [
        (
            np.array(Image.fromarray(face).resize((22, 22), Image.Resampling.LANCZOS))
            if face.shape != (22, 22)
            else face
        )
        for face in faces
    ]

    count = 0

    print("Creating partial face fragments (target: ~1500)...")

    # Fragment definitions: what parts to keep, rest filled with noise/patterns
    fragments = [
        ("eyes_only", [(6, 14, 0, 22)]),  # Keep only eye region
        ("mouth_only", [(16, 22, 4, 18)]),  # Keep only mouth region
        ("left_half_face", [(0, 22, 0, 11)]),  # Keep only left half
        ("upper_third", [(0, 7, 0, 22)]),  # Keep only forehead
        ("center_strip", [(8, 14, 0, 22)]),  # Keep only center horizontal strip
        ("single_eye", [(6, 14, 2, 12)]),  # Keep only one eye
        ("nose_area", [(10, 16, 8, 14)]),  # Keep only nose
        ("corner_fragment", [(0, 11, 0, 11)]),  # Keep only one corner
    ]

    fill_patterns = ["noise", "gradient", "solid_dark", "solid_light", "stripes"]

    for face_idx, face in enumerate(faces):
        for fragment_name, keep_regions in fragments:
            for fill_pattern in fill_patterns:
                for variation in range(4):  # 4 variations per combination
                    # Start with background pattern
                    if fill_pattern == "noise":
                        background = np.random.randint(0, 256, (22, 22), dtype=np.uint8)
                    elif fill_pattern == "gradient":
                        background = np.tile(np.linspace(0, 255, 22), (22, 1)).astype(
                            np.uint8
                        )
                    elif fill_pattern == "solid_dark":
                        background = np.full((22, 22), 50, dtype=np.uint8)
                    elif fill_pattern == "solid_light":
                        background = np.full((22, 22), 200, dtype=np.uint8)
                    elif fill_pattern == "stripes":
                        background = np.zeros((22, 22), dtype=np.uint8)
                        background[::2, :] = 150

                    fragment_image = background.copy()

                    # Place face fragments
                    for y1, y2, x1, x2 in keep_regions:
                        fragment_image[y1:y2, x1:x2] = face[y1:y2, x1:x2]

                    # Add some random modifications
                    if variation > 0:
                        noise = np.random.normal(0, variation * 5, (22, 22))
                        fragment_image = np.clip(
                            fragment_image.astype(float) + noise, 0, 255
                        ).astype(np.uint8)

                    image_name = f"fragment_f{face_idx:02d}_{fragment_name}_{fill_pattern}_v{variation}.png"
                    image_path = os.path.join(not_face_path, image_name)
                    Image.fromarray(fragment_image).save(image_path, "PNG")
                    count += 1

                    if count % 150 == 0:
                        print(f"Created {count} face fragments...")

    print(f"✅ Created {count} partial face fragment negatives")
    return count


def create_natural_non_face_patterns():
    """
    Create natural patterns that could appear in real-world scenarios.
    Target: ~2000 samples
    """
    not_face_path = NOT_FACE_PATH
    count = 0

    def create_natural_pattern(pattern_type, variation):
        image = np.zeros((22, 22), dtype=np.uint8)

        if pattern_type == "tree_bark":
            # Vertical bark-like patterns
            base = np.random.randint(80, 120, (22, 22))
            for i in range(0, 22, 2):
                base[:, i] += np.random.randint(-20, 20, 22)
            image = np.clip(base, 0, 255).astype(np.uint8)

        elif pattern_type == "brick_wall":
            # Brick pattern
            image.fill(150)
            for row in range(0, 22, 4):
                offset = 2 if (row // 4) % 2 else 0
                for col in range(offset, 22, 6):
                    if row < 22:
                        image[row, :] = 100
                    if col < 22:
                        image[:, col] = 100

        elif pattern_type == "fabric_texture":
            # Woven fabric pattern
            y, x = np.mgrid[0:22, 0:22]
            pattern = 128 + 50 * np.sin(x * 0.8) * np.cos(y * 0.8)
            image = np.clip(pattern, 0, 255).astype(np.uint8)

        elif pattern_type == "marble_veins":
            # Marble-like veining
            y, x = np.mgrid[0:22, 0:22]
            base = 180 + 30 * np.sin(x * 0.3 + y * 0.2)
            veins = 50 * np.sin(x * 0.1 + 3 * np.sin(y * 0.15))
            image = np.clip(base + veins, 0, 255).astype(np.uint8)

        elif pattern_type == "wood_grain":
            # Wood grain pattern
            y, x = np.mgrid[0:22, 0:22]
            grain = 120 + 40 * np.sin(x * 0.2 + 2 * np.sin(y * 0.1))
            image = np.clip(grain, 0, 255).astype(np.uint8)

        elif pattern_type == "water_ripples":
            # Water ripple pattern
            y, x = np.mgrid[0:22, 0:22]
            center_x, center_y = 11, 11
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            ripples = 128 + 60 * np.sin(r * 0.8)
            image = np.clip(ripples, 0, 255).astype(np.uint8)

        elif pattern_type == "sand_dunes":
            # Sandy texture
            base = np.random.randint(120, 180, (22, 22))
            y, x = np.mgrid[0:22, 0:22]
            waves = 20 * np.sin(x * 0.3) * np.cos(y * 0.5)
            image = np.clip(base + waves, 0, 255).astype(np.uint8)

        elif pattern_type == "concrete":
            # Concrete texture
            base = np.random.randint(100, 140, (22, 22))
            # Add aggregate spots
            for _ in range(15):
                y, x = np.random.randint(0, 22, 2)
                size = np.random.randint(1, 3)
                base[
                    max(0, y - size) : min(22, y + size),
                    max(0, x - size) : min(22, x + size),
                ] += np.random.randint(-30, 30)
            image = np.clip(base, 0, 255).astype(np.uint8)

        elif pattern_type == "metal_brushed":
            # Brushed metal texture
            base = np.full((22, 22), 140)
            for i in range(22):
                base[i, :] += np.random.randint(-20, 20)
            image = np.clip(base, 0, 255).astype(np.uint8)

        elif pattern_type == "cloud_pattern":
            # Cloud-like pattern
            y, x = np.mgrid[0:22, 0:22]
            clouds = (
                128
                + 60 * np.sin(x * 0.2) * np.cos(y * 0.3)
                + 40 * np.sin(x * 0.5) * np.cos(y * 0.7)
            )
            image = np.clip(clouds, 0, 255).astype(np.uint8)

        # Add variation
        if variation > 0:
            noise = np.random.normal(0, variation * 3, (22, 22))
            image = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)

        return image

    print("Creating natural non-face patterns (target: ~2000)...")

    pattern_types = [
        "tree_bark",
        "brick_wall",
        "fabric_texture",
        "marble_veins",
        "wood_grain",
        "water_ripples",
        "sand_dunes",
        "concrete",
        "metal_brushed",
        "cloud_pattern",
    ]

    for pattern_type in pattern_types:
        for seed in range(20):  # 20 different random seeds
            np.random.seed(seed)
            for variation in range(10):  # 10 variation levels
                image = create_natural_pattern(pattern_type, variation)

                image_name = f"natural_{pattern_type}_seed{seed:02d}_var{variation}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(image).save(image_path, "PNG")
                count += 1

                if count % 200 == 0:
                    print(f"Created {count} natural patterns...")

    print(f"✅ Created {count} natural non-face pattern negatives")
    return count


def create_geometric_shapes():
    """
    Create geometric shapes that might have symmetry but are clearly not faces.
    Target: ~1500 samples
    """
    not_face_path = NOT_FACE_PATH
    count = 0

    def create_shape(shape_type, size_var, fill_var):
        image = np.zeros((22, 22), dtype=np.uint8)

        if shape_type == "circles":
            # Multiple circles
            centers = [(7, 7), (15, 7), (11, 15)]
            radii = [3 + size_var, 3 + size_var, 4 + size_var]
            for (cy, cx), r in zip(centers, radii):
                y, x = np.mgrid[0:22, 0:22]
                mask = (x - cx) ** 2 + (y - cy) ** 2 <= r**2
                image[mask] = 150 + fill_var

        elif shape_type == "rectangles":
            # Overlapping rectangles
            image[4:8, 6:16] = 120 + fill_var
            image[10:14, 4:18] = 180 + fill_var
            image[16:20, 8:14] = 100 + fill_var

        elif shape_type == "triangles":
            # Triangle pattern
            for i in range(10):
                image[5 + i, 11 - i : 11 + i + 1] = 140 + fill_var
            image[18:21, 6:16] = 100 + fill_var

        elif shape_type == "hexagon":
            # Hexagonal pattern
            center_y, center_x = 11, 11
            radius = 8 + size_var
            y, x = np.mgrid[0:22, 0:22]
            # Approximate hexagon with conditions
            mask = (np.abs(x - center_x) <= radius / 2) & (
                np.abs(y - center_y) <= radius
            )
            image[mask] = 160 + fill_var

        elif shape_type == "diamond":
            # Diamond shape
            center_y, center_x = 11, 11
            y, x = np.mgrid[0:22, 0:22]
            mask = np.abs(x - center_x) + np.abs(y - center_y) <= 8 + size_var
            image[mask] = 140 + fill_var

        elif shape_type == "cross":
            # Cross shape
            image[8:14, 2:20] = 130 + fill_var  # Horizontal bar
            image[2:20, 8:14] = 130 + fill_var  # Vertical bar

        elif shape_type == "star":
            # Star-like pattern
            center_y, center_x = 11, 11
            y, x = np.mgrid[0:22, 0:22]
            # Create star points
            angle = np.arctan2(y - center_y, x - center_x)
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            star_mask = r <= (6 + size_var) * (1 + 0.5 * np.sin(5 * angle))
            image[star_mask] = 150 + fill_var

        return np.clip(image, 0, 255).astype(np.uint8)

    print("Creating geometric shapes (target: ~1500)...")

    shape_types = [
        "circles",
        "rectangles",
        "triangles",
        "hexagon",
        "diamond",
        "cross",
        "star",
    ]

    for shape_type in shape_types:
        for size_var in range(-2, 3):  # Size variations
            for fill_var in range(-30, 31, 10):  # Fill intensity variations
                for rotation in range(15):  # Different orientations/positions
                    np.random.seed(rotation)  # Reproducible randomness
                    image = create_shape(shape_type, size_var, fill_var)

                    # Add some random positioning offset
                    if rotation > 0:
                        shift_y = np.random.randint(-2, 3)
                        shift_x = np.random.randint(-2, 3)
                        shifted = np.zeros_like(image)
                        y_slice = slice(max(0, shift_y), min(22, 22 + shift_y))
                        x_slice = slice(max(0, shift_x), min(22, 22 + shift_x))
                        orig_y = slice(max(0, -shift_y), min(22, 22 - shift_y))
                        orig_x = slice(max(0, -shift_x), min(22, 22 - shift_x))
                        shifted[y_slice, x_slice] = image[orig_y, orig_x]
                        image = shifted

                    image_name = f"shape_{shape_type}_size{size_var:+d}_fill{fill_var:+d}_rot{rotation:02d}.png"
                    image_path = os.path.join(not_face_path, image_name)
                    Image.fromarray(image).save(image_path, "PNG")
                    count += 1

                    if count % 150 == 0:
                        print(f"Created {count} geometric shapes...")

    print(f"✅ Created {count} geometric shape negatives")
    return count


def create_random_noise_patterns():
    """
    Create various noise patterns and random textures.
    Target: ~3000 samples
    """
    not_face_path = NOT_FACE_PATH
    count = 0

    print("Creating random noise patterns (target: ~3000)...")

    noise_types = [
        ("uniform", lambda: np.random.randint(0, 256, (22, 22))),
        ("gaussian", lambda: np.clip(np.random.normal(128, 50, (22, 22)), 0, 255)),
        ("salt_pepper", lambda: np.random.choice([0, 255], (22, 22))),
        ("speckle", lambda: np.clip(128 + np.random.normal(0, 30, (22, 22)), 0, 255)),
    ]

    for noise_name, noise_func in noise_types:
        for seed in range(750):  # 750 samples per noise type = 3000 total
            np.random.seed(seed)
            noise_image = noise_func().astype(np.uint8)

            image_name = f"noise_{noise_name}_seed{seed:04d}.png"
            image_path = os.path.join(not_face_path, image_name)
            Image.fromarray(noise_image).save(image_path, "PNG")
            count += 1

            if count % 300 == 0:
                print(f"Created {count} noise patterns...")

    print(f"✅ Created {count} noise pattern negatives")
    return count


def create_monochromatic_negatives():
    """
    Create monochromatic (single color) negative samples.
    These are important for teaching the detector that uniform color patches are not faces.
    Target: ~1000 samples
    """
    not_face_path = NOT_FACE_PATH
    count = 0

    print("Creating monochromatic negative samples (target: ~1000)...")

    # Pure monochromatic samples
    print("Creating pure monochromatic samples...")
    for intensity in range(0, 256, 5):  # Every 5th intensity level (52 samples)
        mono_image = np.full((22, 22), intensity, dtype=np.uint8)

        image_name = f"mono_pure_intensity_{intensity:03d}.png"
        image_path = os.path.join(not_face_path, image_name)
        Image.fromarray(mono_image).save(image_path, "PNG")
        count += 1

    # Monochromatic with slight noise
    print("Creating monochromatic with noise...")
    base_intensities = [0, 32, 64, 96, 128, 160, 192, 224, 255]  # 9 base levels
    noise_levels = [1, 2, 3, 5, 8, 10, 15, 20]  # 8 noise levels

    for base_intensity in base_intensities:
        for noise_level in noise_levels:
            for seed in range(5):  # 5 random variations per combination
                np.random.seed(seed)

                # Create base monochromatic image
                mono_image = np.full((22, 22), base_intensity, dtype=np.float32)

                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, (22, 22))
                noisy_mono = mono_image + noise

                # Clip to valid range
                noisy_mono = np.clip(noisy_mono, 0, 255).astype(np.uint8)

                image_name = f"mono_noise_base{base_intensity:03d}_noise{noise_level:02d}_seed{seed}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(noisy_mono).save(image_path, "PNG")
                count += 1

    # Monochromatic gradients (still mostly uniform)
    print("Creating monochromatic gradients...")
    gradient_types = ["horizontal", "vertical", "radial", "diagonal"]
    base_intensities = [32, 64, 96, 128, 160, 192, 224]  # 7 base levels
    gradient_strengths = [5, 10, 15, 20, 25]  # 5 gradient strengths

    for gradient_type in gradient_types:
        for base_intensity in base_intensities:
            for gradient_strength in gradient_strengths:
                y, x = np.mgrid[0:22, 0:22]

                if gradient_type == "horizontal":
                    gradient = base_intensity + gradient_strength * (x - 11) / 11
                elif gradient_type == "vertical":
                    gradient = base_intensity + gradient_strength * (y - 11) / 11
                elif gradient_type == "diagonal":
                    gradient = base_intensity + gradient_strength * ((x + y) - 22) / 22
                elif gradient_type == "radial":
                    center_dist = np.sqrt((x - 11) ** 2 + (y - 11) ** 2)
                    max_dist = np.sqrt(11**2 + 11**2)
                    gradient = base_intensity + gradient_strength * (
                        center_dist - max_dist / 2
                    ) / (max_dist / 2)

                gradient = np.clip(gradient, 0, 255).astype(np.uint8)

                image_name = f"mono_grad_{gradient_type}_base{base_intensity:03d}_str{gradient_strength:02d}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(gradient).save(image_path, "PNG")
                count += 1

    # Monochromatic with simple patterns (still mostly uniform)
    print("Creating monochromatic with simple patterns...")
    pattern_types = ["dots", "lines_h", "lines_v", "checker", "border"]
    base_intensities = [64, 128, 192]  # 3 base levels
    pattern_intensities = [32, 64, 96]  # 3 pattern contrast levels

    for pattern_type in pattern_types:
        for base_intensity in base_intensities:
            for pattern_intensity in pattern_intensities:
                for spacing in [3, 4, 5, 6]:  # 4 different spacings
                    pattern_image = np.full((22, 22), base_intensity, dtype=np.uint8)

                    if pattern_type == "dots":
                        # Dot pattern
                        for i in range(0, 22, spacing):
                            for j in range(0, 22, spacing):
                                if i < 22 and j < 22:
                                    pattern_image[i, j] = pattern_intensity

                    elif pattern_type == "lines_h":
                        # Horizontal lines
                        for i in range(0, 22, spacing):
                            if i < 22:
                                pattern_image[i, :] = pattern_intensity

                    elif pattern_type == "lines_v":
                        # Vertical lines
                        for j in range(0, 22, spacing):
                            if j < 22:
                                pattern_image[:, j] = pattern_intensity

                    elif pattern_type == "checker":
                        # Checkerboard pattern
                        for i in range(0, 22, spacing):
                            for j in range(0, 22, spacing):
                                if (i // spacing + j // spacing) % 2 == 0:
                                    end_i = min(i + spacing, 22)
                                    end_j = min(j + spacing, 22)
                                    pattern_image[i:end_i, j:end_j] = pattern_intensity

                    elif pattern_type == "border":
                        # Border pattern
                        thickness = spacing // 2 + 1
                        pattern_image[:thickness, :] = pattern_intensity
                        pattern_image[-thickness:, :] = pattern_intensity
                        pattern_image[:, :thickness] = pattern_intensity
                        pattern_image[:, -thickness:] = pattern_intensity

                    image_name = f"mono_pattern_{pattern_type}_base{base_intensity:03d}_pat{pattern_intensity:03d}_sp{spacing}.png"
                    image_path = os.path.join(not_face_path, image_name)
                    Image.fromarray(pattern_image).save(image_path, "PNG")
                    count += 1

    # Monochromatic blocks (different regions with different intensities)
    print("Creating monochromatic blocks...")
    block_configs = [
        ("halves_h", [(slice(0, 11), slice(0, 22)), (slice(11, 22), slice(0, 22))]),
        ("halves_v", [(slice(0, 22), slice(0, 11)), (slice(0, 22), slice(11, 22))]),
        (
            "quarters",
            [
                (slice(0, 11), slice(0, 11)),
                (slice(0, 11), slice(11, 22)),
                (slice(11, 22), slice(0, 11)),
                (slice(11, 22), slice(11, 22)),
            ],
        ),
        (
            "thirds_h",
            [
                (slice(0, 7), slice(0, 22)),
                (slice(7, 15), slice(0, 22)),
                (slice(15, 22), slice(0, 22)),
            ],
        ),
        ("center_border", [(slice(6, 16), slice(6, 16)), (slice(0, 22), slice(0, 22))]),
    ]

    for block_name, regions in block_configs:
        for contrast_level in [32, 64, 96, 128]:  # 4 contrast levels
            for variation in range(8):  # 8 variations per config
                np.random.seed(variation)
                block_image = np.zeros((22, 22), dtype=np.uint8)

                # Assign random intensities to each region
                if block_name == "center_border":
                    # Special case: center and border
                    border_intensity = np.random.randint(0, 256 - contrast_level)
                    center_intensity = border_intensity + contrast_level

                    block_image.fill(border_intensity)  # Border (everything)
                    y_slice, x_slice = regions[0]  # Center region
                    block_image[y_slice, x_slice] = center_intensity
                else:
                    # Regular regions
                    base_intensity = np.random.randint(
                        contrast_level - 1, 256 - contrast_level
                    )
                    for i, (y_slice, x_slice) in enumerate(regions):
                        region_intensity = base_intensity + i * contrast_level
                        block_image[y_slice, x_slice] = region_intensity

                image_name = f"mono_blocks_{block_name}_contrast{contrast_level:03d}_var{variation}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(block_image).save(image_path, "PNG")
                count += 1

    if count % 100 == 0 or count < 100:
        print(f"Created {count} monochromatic samples...")

    print(f"✅ Created {count} monochromatic negative samples")
    return count


def create_extended_monochromatic_negatives():
    """
    Extended version with more sophisticated monochromatic patterns.
    Target: ~2000 samples
    """
    not_face_path = NOT_FACE_PATH
    count = 0

    print("Creating extended monochromatic negatives (target: ~2000)...")

    # First create the basic set
    count += create_monochromatic_negatives()

    # Add more sophisticated monochromatic patterns
    print("Creating textured monochromatic samples...")

    # Textured monochromatic (single color with texture)
    base_colors = [0, 30, 60, 90, 120, 150, 180, 210, 240, 255]
    texture_types = ["fabric", "wood", "metal", "paper", "concrete"]

    for base_color in base_colors:
        for texture_type in texture_types:
            for intensity in [5, 10, 15, 20]:  # Texture intensity
                for seed in range(4):  # 4 variations
                    np.random.seed(seed)

                    # Create base color
                    textured_mono = np.full((22, 22), base_color, dtype=np.float32)

                    if texture_type == "fabric":
                        y, x = np.mgrid[0:22, 0:22]
                        texture = intensity * np.sin(x * 0.8) * np.cos(y * 0.8)
                    elif texture_type == "wood":
                        y, x = np.mgrid[0:22, 0:22]
                        texture = intensity * np.sin(x * 0.3 + 2 * np.sin(y * 0.1))
                    elif texture_type == "metal":
                        texture = intensity * np.random.normal(0, 1, (22, 22))
                        # Make it more directional (brushed metal)
                        for i in range(22):
                            texture[i, :] = np.mean(texture[i, :])
                    elif texture_type == "paper":
                        texture = intensity * np.random.normal(0, 0.5, (22, 22))
                    elif texture_type == "concrete":
                        texture = intensity * np.random.normal(0, 1, (22, 22))
                        # Add some speckles
                        speckles = np.random.choice(
                            [0, intensity * 2], (22, 22), p=[0.95, 0.05]
                        )
                        texture += speckles

                    textured_mono += texture
                    textured_mono = np.clip(textured_mono, 0, 255).astype(np.uint8)

                    image_name = f"mono_textured_{texture_type}_base{base_color:03d}_int{intensity:02d}_seed{seed}.png"
                    image_path = os.path.join(not_face_path, image_name)
                    Image.fromarray(textured_mono).save(image_path, "PNG")
                    count += 1

    # Monochromatic with geometric overlays
    print("Creating monochromatic with geometric overlays...")
    overlay_shapes = ["circle", "square", "triangle", "cross", "diamond"]

    for base_color in [32, 96, 160, 224]:  # 4 base colors
        for overlay_shape in overlay_shapes:
            for overlay_color in [0, 64, 128, 192, 255]:  # 5 overlay colors
                for size_var in [0, 1, 2]:  # 3 size variations
                    mono_image = np.full((22, 22), base_color, dtype=np.uint8)

                    center_y, center_x = 11, 11
                    size = 6 + size_var

                    if overlay_shape == "circle":
                        y, x = np.mgrid[0:22, 0:22]
                        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= size**2
                        mono_image[mask] = overlay_color

                    elif overlay_shape == "square":
                        y1, y2 = center_y - size, center_y + size
                        x1, x2 = center_x - size, center_x + size
                        y1, y2 = max(0, y1), min(22, y2)
                        x1, x2 = max(0, x1), min(22, x2)
                        mono_image[y1:y2, x1:x2] = overlay_color

                    elif overlay_shape == "triangle":
                        for i in range(size):
                            y = center_y - size + i
                            if 0 <= y < 22:
                                x1 = max(0, center_x - i)
                                x2 = min(22, center_x + i + 1)
                                mono_image[y, x1:x2] = overlay_color

                    elif overlay_shape == "cross":
                        # Horizontal bar
                        y1, y2 = center_y - 2, center_y + 3
                        x1, x2 = center_x - size, center_x + size
                        y1, y2 = max(0, y1), min(22, y2)
                        x1, x2 = max(0, x1), min(22, x2)
                        mono_image[y1:y2, x1:x2] = overlay_color
                        # Vertical bar
                        y1, y2 = center_y - size, center_y + size
                        x1, x2 = center_x - 2, center_x + 3
                        y1, y2 = max(0, y1), min(22, y2)
                        x1, x2 = max(0, x1), min(22, x2)
                        mono_image[y1:y2, x1:x2] = overlay_color

                    elif overlay_shape == "diamond":
                        y, x = np.mgrid[0:22, 0:22]
                        mask = np.abs(x - center_x) + np.abs(y - center_y) <= size
                        mono_image[mask] = overlay_color

                    image_name = f"mono_overlay_{overlay_shape}_base{base_color:03d}_over{overlay_color:03d}_size{size}.png"
                    image_path = os.path.join(not_face_path, image_name)
                    Image.fromarray(mono_image).save(image_path, "PNG")
                    count += 1

    print(f"✅ Created {count} total monochromatic negative samples")
    return count


def create_quick_monochromatic_negatives():
    """
    Quick version for generating essential monochromatic negatives.
    Target: ~500 samples
    """
    not_face_path = NOT_FACE_PATH
    count = 0

    print("Creating quick monochromatic negatives (target: ~500)...")

    # Essential pure monochromatic samples
    for intensity in range(0, 256, 10):  # Every 10th intensity (26 samples)
        mono_image = np.full((22, 22), intensity, dtype=np.uint8)

        image_name = f"mono_quick_pure_{intensity:03d}.png"
        image_path = os.path.join(not_face_path, image_name)
        Image.fromarray(mono_image).save(image_path, "PNG")
        count += 1

    # Monochromatic with minimal noise (essential for robustness)
    base_intensities = [0, 64, 128, 192, 255]  # 5 key intensities
    for base_intensity in base_intensities:
        for noise_level in [2, 5, 10, 15]:  # 4 noise levels
            for seed in range(5):  # 5 variations
                np.random.seed(seed)

                mono_image = np.full((22, 22), base_intensity, dtype=np.float32)
                noise = np.random.normal(0, noise_level, (22, 22))
                noisy_mono = np.clip(mono_image + noise, 0, 255).astype(np.uint8)

                image_name = f"mono_quick_noise_base{base_intensity:03d}_n{noise_level:02d}_s{seed}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(noisy_mono).save(image_path, "PNG")
                count += 1

    # Simple gradients
    for gradient_type in ["horizontal", "vertical"]:
        for base_intensity in [64, 128, 192]:
            for gradient_strength in [20, 40, 60]:
                y, x = np.mgrid[0:22, 0:22]

                if gradient_type == "horizontal":
                    gradient = base_intensity + gradient_strength * (x - 11) / 11
                else:  # vertical
                    gradient = base_intensity + gradient_strength * (y - 11) / 11

                gradient = np.clip(gradient, 0, 255).astype(np.uint8)

                image_name = f"mono_quick_grad_{gradient_type}_b{base_intensity:03d}_s{gradient_strength:02d}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(gradient).save(image_path, "PNG")
                count += 1

    # Two-tone blocks (most important for face detection)
    for base_color in [32, 96, 160, 224]:
        for contrast in [32, 64, 96]:
            second_color = min(255, base_color + contrast)

            # Horizontal split
            split_image = np.full((22, 22), base_color, dtype=np.uint8)
            split_image[11:, :] = second_color

            image_name = f"mono_quick_split_h_b{base_color:03d}_c{contrast:02d}.png"
            image_path = os.path.join(not_face_path, image_name)
            Image.fromarray(split_image).save(image_path, "PNG")
            count += 1

            # Vertical split
            split_image = np.full((22, 22), base_color, dtype=np.uint8)
            split_image[:, 11:] = second_color

            image_name = f"mono_quick_split_v_b{base_color:03d}_c{contrast:02d}.png"
            image_path = os.path.join(not_face_path, image_name)
            Image.fromarray(split_image).save(image_path, "PNG")
            count += 1

    print(f"✅ Created {count} quick monochromatic negative samples")
    return count


def create_shifted_positive_samples():
    """
    Take positive face samples, shift them by at least 4 pixels in x/y directions,
    and pad the frame with monochrome backgrounds or noise.
    This creates challenging negative samples where faces are off-center.
    Target: ~3000 samples
    """
    face_path = FACE_PATH
    not_face_path = NOT_FACE_PATH

    # Load positive face samples
    faces = load_images_from_folder(face_path)[:20]  # Use up to 20 different faces
    faces = [
        (
            np.array(Image.fromarray(face).resize((22, 22), Image.Resampling.LANCZOS))
            if face.shape != (22, 22)
            else face
        )
        for face in faces
    ]

    count = 0

    # Define shift patterns (minimum 4 pixels in at least one direction)
    shift_patterns = [
        # Single direction shifts
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),  # Right shifts
        (-4, 0),
        (-5, 0),
        (-6, 0),
        (-7, 0),
        (-8, 0),  # Left shifts
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),  # Down shifts
        (0, -4),
        (0, -5),
        (0, -6),
        (0, -7),
        (0, -8),  # Up shifts
        # Diagonal shifts
        (4, 4),
        (5, 5),
        (6, 6),
        (4, -4),
        (5, -5),
        (6, -6),
        (-4, 4),
        (-5, 5),
        (-6, 6),
        (-4, -4),
        (-5, -5),
        (-6, -6),
        # Asymmetric shifts
        (4, 2),
        (6, 3),
        (8, 2),
        (4, -2),
        (6, -3),
        (8, -2),
        (-4, 2),
        (-6, 3),
        (-8, 2),
        (-4, -2),
        (-6, -3),
        (-8, -2),
        (2, 4),
        (3, 6),
        (2, 8),
        (-2, 4),
        (-3, 6),
        (-2, 8),
        (2, -4),
        (3, -6),
        (2, -8),
        (-2, -4),
        (-3, -6),
        (-2, -8),
    ]

    # Define background fill types
    background_types = [
        ("mono_black", lambda: 0),
        ("mono_dark", lambda: 32),
        ("mono_mid_dark", lambda: 64),
        ("mono_mid", lambda: 128),
        ("mono_mid_light", lambda: 160),
        ("mono_light", lambda: 192),
        ("mono_white", lambda: 255),
        ("mono_random", lambda: np.random.randint(0, 256)),
        ("noise_uniform", lambda: np.random.randint(0, 256, (22, 22))),
        (
            "noise_gaussian_low",
            lambda: np.clip(np.random.normal(128, 30, (22, 22)), 0, 255),
        ),
        (
            "noise_gaussian_high",
            lambda: np.clip(np.random.normal(128, 60, (22, 22)), 0, 255),
        ),
        (
            "noise_salt_pepper",
            lambda: np.random.choice([0, 255], (22, 22), p=[0.5, 0.5]),
        ),
        ("gradient_h", lambda: np.tile(np.linspace(0, 255, 22), (22, 1))),
        (
            "gradient_v",
            lambda: np.tile(np.linspace(0, 255, 22).reshape(-1, 1), (1, 22)),
        ),
        ("checker_fine", lambda: create_checkerboard(22, 22, 2)),
        ("checker_coarse", lambda: create_checkerboard(22, 22, 4)),
    ]

    def create_checkerboard(h, w, size):
        """Helper function to create checkerboard pattern"""
        pattern = np.zeros((h, w))
        for i in range(0, h, size):
            for j in range(0, w, size):
                if (i // size + j // size) % 2 == 0:
                    pattern[i : i + size, j : j + size] = 255
        return pattern

    print("Creating shifted positive samples as negatives (target: ~3000)...")
    print(f"Using {len(faces)} source faces")
    print(f"Shift patterns: {len(shift_patterns)}")
    print(f"Background types: {len(background_types)}")

    # Create samples with different combinations
    for face_idx, face in enumerate(faces):
        for shift_x, shift_y in shift_patterns:
            for bg_name, bg_func in background_types:
                # Set random seed for reproducible backgrounds where applicable
                np.random.seed(
                    (face_idx * 1000 + shift_x * 100 + shift_y * 10 + hash(bg_name))
                    % 10000
                )

                # Create background
                if bg_name.startswith("mono_") and not bg_name == "mono_random":
                    # Monochrome background
                    background = np.full((22, 22), bg_func(), dtype=np.uint8)
                else:
                    # Pattern or noise background
                    background = bg_func()
                    background = np.full((22, 22), background, dtype=np.uint8)

                # Create shifted image
                shifted_image = background.copy()

                # Calculate the region where the face will be placed
                face_h, face_w = face.shape

                # Calculate destination coordinates (with bounds checking)
                dst_y1 = max(0, shift_y)
                dst_y2 = min(22, shift_y + face_h)
                dst_x1 = max(0, shift_x)
                dst_x2 = min(22, shift_x + face_w)

                # Calculate source coordinates
                src_y1 = max(0, -shift_y)
                src_y2 = src_y1 + (dst_y2 - dst_y1)
                src_x1 = max(0, -shift_x)
                src_x2 = src_x1 + (dst_x2 - dst_x1)

                # Only place face if there's a valid region
                if (
                    dst_y2 > dst_y1
                    and dst_x2 > dst_x1
                    and src_y2 > src_y1
                    and src_x2 > src_x1
                ):
                    shifted_image[dst_y1:dst_y2, dst_x1:dst_x2] = face[
                        src_y1:src_y2, src_x1:src_x2
                    ]

                # Save the shifted sample
                image_name = f"shifted_f{face_idx:02d}_x{shift_x:+03d}_y{shift_y:+03d}_{bg_name}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(shifted_image).save(image_path, "PNG")
                count += 1

                if count % 500 == 0:
                    print(f"Created {count} shifted samples...")

    print(f"✅ Created {count} shifted positive samples as negatives")
    return count


def create_partial_shifted_faces():
    """
    Create samples where faces are partially visible due to shifting.
    These are particularly challenging as they contain face features but are incomplete.
    Target: ~2000 samples
    """
    face_path = FACE_PATH
    not_face_path = NOT_FACE_PATH

    faces = load_images_from_folder(face_path)[:15]  # Use 15 different faces
    faces = [
        (
            np.array(Image.fromarray(face).resize((22, 22), Image.Resampling.LANCZOS))
            if face.shape != (22, 22)
            else face
        )
        for face in faces
    ]

    count = 0

    # Extreme shifts that leave only partial faces visible
    extreme_shifts = [
        # Heavy shifts that cut off significant portions
        (10, 0),
        (12, 0),
        (14, 0),
        (16, 0),  # Heavy right shift
        (-10, 0),
        (-12, 0),
        (-14, 0),
        (-16, 0),  # Heavy left shift
        (0, 10),
        (0, 12),
        (0, 14),
        (0, 16),  # Heavy down shift
        (0, -10),
        (0, -12),
        (0, -14),
        (0, -16),  # Heavy up shift
        # Diagonal extreme shifts
        (10, 10),
        (12, 12),
        (8, -8),
        (10, -10),
        (-10, 10),
        (-12, 12),
        (-8, -8),
        (-10, -10),
        # Asymmetric extreme shifts
        (14, 4),
        (16, 6),
        (14, -4),
        (16, -6),
        (-14, 4),
        (-16, 6),
        (-14, -4),
        (-16, -6),
        (4, 14),
        (6, 16),
        (-4, 14),
        (-6, 16),
        (4, -14),
        (6, -16),
        (-4, -14),
        (-6, -16),
    ]

    # Background types optimized for partial faces
    partial_backgrounds = [
        ("mono_black", lambda: 0),
        ("mono_white", lambda: 255),
        ("mono_gray", lambda: 128),
        ("noise_low", lambda: np.clip(np.random.normal(100, 20, (22, 22)), 0, 255)),
        ("noise_high", lambda: np.clip(np.random.normal(150, 20, (22, 22)), 0, 255)),
        ("gradient_fade", lambda: create_fade_gradient()),
    ]

    def create_fade_gradient():
        """Create a fading gradient for more natural partial occlusion"""
        y, x = np.mgrid[0:22, 0:22]
        # Random direction gradient
        direction = np.random.choice(["h", "v", "d1", "d2"])
        if direction == "h":
            grad = x / 21 * 255
        elif direction == "v":
            grad = y / 21 * 255
        elif direction == "d1":
            grad = (x + y) / 42 * 255
        else:  # d2
            grad = (21 - x + y) / 42 * 255
        return grad

    print(f"Creating partial shifted faces (target: ~2000)...")
    print(f"Using {len(faces)} source faces")

    for face_idx, face in enumerate(faces):
        for shift_x, shift_y in extreme_shifts:
            for bg_name, bg_func in partial_backgrounds:
                # Set random seed for reproducible results
                np.random.seed(
                    (face_idx * 1000 + shift_x * 100 + shift_y * 10 + hash(bg_name))
                    % 10000
                )

                # Create background
                background = bg_func()
                background = np.full((22, 22), background, dtype=np.uint8)

                # Create shifted image with partial face
                shifted_image = background.copy()

                # Calculate placement with bounds checking
                face_h, face_w = face.shape

                dst_y1 = max(0, shift_y)
                dst_y2 = min(22, shift_y + face_h)
                dst_x1 = max(0, shift_x)
                dst_x2 = min(22, shift_x + face_w)

                src_y1 = max(0, -shift_y)
                src_y2 = src_y1 + (dst_y2 - dst_y1)
                src_x1 = max(0, -shift_x)
                src_x2 = src_x1 + (dst_x2 - dst_x1)

                # Place partial face
                if dst_y2 > dst_y1 and dst_x2 > dst_x1:
                    shifted_image[dst_y1:dst_y2, dst_x1:dst_x2] = face[
                        src_y1:src_y2, src_x1:src_x2
                    ]

                # Save the partial face sample
                image_name = f"partial_f{face_idx:02d}_x{shift_x:+03d}_y{shift_y:+03d}_{bg_name}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(shifted_image).save(image_path, "PNG")
                count += 1

                if count % 300 == 0:
                    print(f"Created {count} partial face samples...")

    print(f"✅ Created {count} partial shifted face negatives")
    return count


def create_multi_face_fragments():
    """
    Create samples with multiple face fragments in different positions.
    These are very challenging as they contain multiple face-like regions.
    Target: ~1000 samples
    """
    face_path = FACE_PATH
    not_face_path = NOT_FACE_PATH

    faces = load_images_from_folder(face_path)[:8]  # Use 8 different faces
    faces = [
        (
            np.array(Image.fromarray(face).resize((22, 22), Image.Resampling.LANCZOS))
            if face.shape != (22, 22)
            else face
        )
        for face in faces
    ]

    count = 0

    print("Creating multi-face fragments (target: ~1000)...")

    # Define fragment positions and sizes for multiple fragments
    fragment_configs = [
        # Two fragments
        [{"pos": (0, 0), "size": (12, 12)}, {"pos": (10, 10), "size": (12, 12)}],
        [{"pos": (0, 10), "size": (12, 12)}, {"pos": (10, 0), "size": (12, 12)}],
        [{"pos": (5, 0), "size": (10, 22)}, {"pos": (12, 0), "size": (10, 22)}],
        [{"pos": (0, 5), "size": (22, 10)}, {"pos": (0, 12), "size": (22, 10)}],
        # Three fragments
        [
            {"pos": (0, 0), "size": (8, 8)},
            {"pos": (0, 14), "size": (8, 8)},
            {"pos": (14, 7), "size": (8, 8)},
        ],
        [
            {"pos": (7, 0), "size": (8, 8)},
            {"pos": (0, 7), "size": (8, 8)},
            {"pos": (14, 14), "size": (8, 8)},
        ],
        # Four fragments (corners)
        [
            {"pos": (0, 0), "size": (6, 6)},
            {"pos": (0, 16), "size": (6, 6)},
            {"pos": (16, 0), "size": (6, 6)},
            {"pos": (16, 16), "size": (6, 6)},
        ],
    ]

    background_types = ["black", "white", "gray", "noise"]

    for config_idx, fragment_config in enumerate(fragment_configs):
        for bg_type in background_types:
            for face_combo in range(20):  # 20 different face combinations
                np.random.seed(face_combo)

                # Create background
                if bg_type == "black":
                    multi_image = np.zeros((22, 22), dtype=np.uint8)
                elif bg_type == "white":
                    multi_image = np.full((22, 22), 255, dtype=np.uint8)
                elif bg_type == "gray":
                    multi_image = np.full((22, 22), 128, dtype=np.uint8)
                else:  # noise
                    multi_image = np.random.randint(0, 256, (22, 22), dtype=np.uint8)

                # Place face fragments
                for frag_idx, fragment in enumerate(fragment_config):
                    # Choose a random face
                    face_idx = np.random.randint(0, len(faces))
                    face = faces[face_idx]

                    # Extract and place fragment
                    pos_y, pos_x = fragment["pos"]
                    size_h, size_w = fragment["size"]

                    # Random source position in the face
                    src_y = np.random.randint(0, max(1, 22 - size_h + 1))
                    src_x = np.random.randint(0, max(1, 22 - size_w + 1))

                    # Calculate actual placement bounds
                    dst_y1 = pos_y
                    dst_y2 = min(22, pos_y + size_h)
                    dst_x1 = pos_x
                    dst_x2 = min(22, pos_x + size_w)

                    src_y2 = src_y + (dst_y2 - dst_y1)
                    src_x2 = src_x + (dst_x2 - dst_x1)

                    if dst_y2 > dst_y1 and dst_x2 > dst_x1:
                        multi_image[dst_y1:dst_y2, dst_x1:dst_x2] = face[
                            src_y:src_y2, src_x:src_x2
                        ]

                # Save multi-fragment sample
                image_name = f"multifrag_config{config_idx}_bg{bg_type}_combo{face_combo:02d}.png"
                image_path = os.path.join(not_face_path, image_name)
                Image.fromarray(multi_image).save(image_path, "PNG")
                count += 1

                if count % 100 == 0:
                    print(f"Created {count} multi-fragment samples...")

    print(f"✅ Created {count} multi-face fragment negatives")
    return count


def create_all_nonface_negatives():
    """
    Calls all negative sample generators in sequence.
    Does not collect or print per-category counts, only progress messages.
    """
    print("=== STARTING FULL NEGATIVE SET GENERATION ===\n")

    print("[1/12] Adding random noise images...")
    add_random_images(1000)

    print("[2/12] Creating face-like objects...")
    create_face_like_objects()

    print("[3/12] Creating partial face fragments...")
    create_partial_face_fragments()

    print("[4/12] Creating natural non-face patterns...")
    create_natural_non_face_patterns()

    print("[5/12] Creating geometric shapes...")
    create_geometric_shapes()

    print("[6/12] Creating random noise patterns...")
    create_random_noise_patterns()

    print("[7/12] Creating monochromatic negatives...")
    create_monochromatic_negatives()

    print("[8/12] Creating extended monochromatic negatives...")
    create_extended_monochromatic_negatives()

    print("[9/12] Creating quick monochromatic negatives...")
    create_quick_monochromatic_negatives()

    print("[10/12] Creating shifted positive samples as negatives...")
    create_shifted_positive_samples()

    print("[11/12] Creating partial shifted faces...")
    create_partial_shifted_faces()

    print("[12/12] Creating multi-face fragments...")
    create_multi_face_fragments()

    print("\n=== NEGATIVE SET GENERATION COMPLETE ===")
    print("All negative sample generators have finished.\n")
    print("========================================")


if __name__ == "__main__":
    print("Starting negative sample generation...")
    # create_all_nonface_negatives()
    print("Done!")
