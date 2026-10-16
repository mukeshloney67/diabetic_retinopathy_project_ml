"""
Dataset preparation script for Diabetic Retinopathy.
Splits labeled image datasets (e.g., APTOS 2019, EyePACS) into stratified
train, validation, and test subdirectories organized by class.
"""

import os
import shutil
import argparse
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    0: "0_No_DR",
    1: "1_Mild",
    2: "2_Moderate",
    3: "3_Severe",
    4: "4_Proliferative_DR",
}


def prepare_dataset(
    csv_path: str,
    images_dir: str,
    output_dir: str = "data",
    val_split: float = 0.15,
    test_split: float = 0.15,
    image_col: str = "id_code",
    label_col: str = "diagnosis",
    img_extension: str = ".png",
    seed: int = 42,
) -> None:
    logger.info(f"Reading labels from: {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df)} samples.")
    logger.info(f"Class distribution:\n{df[label_col].value_counts().sort_index()}")

    # Stratified split into train_val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_split, stratify=df[label_col], random_state=seed
    )

    # Stratified split into train and validation
    effective_val_ratio = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val_df, test_size=effective_val_ratio, stratify=train_val_df[label_col], random_state=seed
    )

    splits = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    for split_name, split_df in splits.items():
        logger.info(f"Organizing {split_name} split ({len(split_df)} images)...")
        for cls_id, cls_dir in CLASS_MAPPING.items():
            os.makedirs(os.path.join(output_dir, split_name, cls_dir), exist_ok=True)

        copied = 0
        for _, row in split_df.iterrows():
            img_id = str(row[image_col])
            label = int(row[label_col])
            dest_dir = os.path.join(output_dir, split_name, CLASS_MAPPING[label])

            if not img_id.endswith(img_extension):
                fname = f"{img_id}{img_extension}"
            else:
                fname = img_id

            src = os.path.join(images_dir, fname)
            dst = os.path.join(dest_dir, fname)

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                # Try finding alternative extension
                found = False
                for alt_ext in [".png", ".jpg", ".jpeg"]:
                    alt_src = os.path.join(images_dir, f"{os.path.splitext(img_id)[0]}{alt_ext}")
                    if os.path.exists(alt_src):
                        shutil.copy2(alt_src, os.path.join(dest_dir, os.path.basename(alt_src)))
                        copied += 1
                        found = True
                        break

        logger.info(f"Finished {split_name}: {copied} files copied.")


def main():
    parser = argparse.ArgumentParser(description="Prepare and split retinal image dataset.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing raw images")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    args = parser.parse_args()

    prepare_dataset(
        args.csv_path,
        args.images_dir,
        args.output_dir,
        args.val_split,
        args.test_split,
    )


if __name__ == "__main__":
    main()
