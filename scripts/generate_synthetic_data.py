"""
Generate a lightweight synthetic retinal dataset for testing the pipeline end-to-end.
Creates circular retinal masks with synthetic optic disks and lesions.
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image

CLASS_NAMES = ["0_No_DR", "1_Mild", "2_Moderate", "3_Severe", "4_Proliferative_DR"]


def create_retinal_fundus_sample(stage: int, size: int = 299) -> Image.Image:
    """Create a synthetic retinal fundus image representing a specific DR severity stage."""
    # 1. Base dark background
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # 2. Orange-red retinal fundus circle
    center = (size // 2, size // 2)
    radius = int(size * 0.44)
    cv2.circle(img, center, radius, (20, 70, 190), -1)  # BGR

    # Add subtle gradient / texture
    noise = np.random.normal(0, 10, (size, size, 3)).astype(np.int16)
    fundus_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(fundus_mask, center, radius, 255, -1)

    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[fundus_mask == 0] = 0

    # 3. Optic disc (yellowish circle)
    disc_center = (int(size * 0.35), int(size * 0.5))
    disc_radius = int(size * 0.08)
    cv2.circle(img, disc_center, disc_radius, (100, 220, 240), -1)

    # 4. Stage-specific lesions
    if stage >= 1:
        # Microaneurysms (tiny red dots)
        num_ma = stage * 8
        for _ in range(num_ma):
            pt = (
                int(center[0] + np.random.uniform(-0.3, 0.3) * size),
                int(center[1] + np.random.uniform(-0.3, 0.3) * size),
            )
            cv2.circle(img, pt, 2, (0, 0, 160), -1)

    if stage >= 2:
        # Hard exudates (yellow patches)
        num_exudates = stage * 4
        for _ in range(num_exudates):
            pt = (
                int(center[0] + np.random.uniform(-0.25, 0.25) * size),
                int(center[1] + np.random.uniform(-0.25, 0.25) * size),
            )
            cv2.circle(img, pt, np.random.randint(3, 7), (120, 240, 255), -1)

    if stage >= 3:
        # Cotton wool spots (fluffy white areas)
        for _ in range(stage * 2):
            pt = (
                int(center[0] + np.random.uniform(-0.3, 0.3) * size),
                int(center[1] + np.random.uniform(-0.3, 0.3) * size),
            )
            cv2.circle(img, pt, np.random.randint(6, 12), (220, 220, 220), -1)

    if stage == 4:
        # Neovascularization (irregular dark red branching)
        for _ in range(4):
            pts = np.array([
                [center[0] + np.random.randint(-40, 40), center[1] + np.random.randint(-40, 40)],
                [center[0] + np.random.randint(-60, 60), center[1] + np.random.randint(-60, 60)],
                [center[0] + np.random.randint(-80, 80), center[1] + np.random.randint(-80, 80)],
            ], np.int32)
            cv2.polylines(img, [pts], False, (0, 0, 130), 2)

    # Convert BGR back to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def generate_dataset(output_dir: str = "data", samples_per_class: int = 10) -> None:
    print(f"Generating synthetic retinal dataset in '{output_dir}'...")
    splits = {
        "train": samples_per_class,
        "validation": max(2, samples_per_class // 4),
        "test": max(2, samples_per_class // 4),
    }

    for split_name, count in splits.items():
        for stage_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)

            for i in range(count):
                img = create_retinal_fundus_sample(stage=stage_idx)
                img.save(os.path.join(class_dir, f"sample_{stage_idx}_{i:03d}.png"))

    print("Synthetic dataset successfully created!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic retinal dataset")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--samples_per_class", type=int, default=10)
    args = parser.parse_args()
    generate_dataset(args.output_dir, args.samples_per_class)
