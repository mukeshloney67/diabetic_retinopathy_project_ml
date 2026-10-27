"""Command line entry points for data preparation, training, evaluation, and prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import create_splits
from .evaluation import evaluate
from .inference import predict_image
from .training import train
from .utils import write_json


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Path to an experiment YAML file.")


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Validate the dataset and create reproducible splits.")
    _config_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    splits = create_splits(config)
    summary = splits["split"].value_counts().to_dict()
    print(f"Wrote {config.data.split_file}")
    print("Split counts:", dict(sorted(summary.items())))


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train the InceptionV3 diabetic retinopathy classifier.")
    _config_argument(parser)
    args = parser.parse_args()
    run_dir = train(load_config(args.config))
    print(f"Training complete. Best model: {run_dir / 'best_model.keras'}")


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved model on the test split.")
    _config_argument(parser)
    parser.add_argument("--model", required=True, type=Path, help="Saved .keras model path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional evaluation output directory.")
    args = parser.parse_args()
    metrics = evaluate(args.model, load_config(args.config), args.output_dir)
    print(
        "Evaluation complete: "
        f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
        f"kappa={metrics['quadratic_weighted_kappa']:.4f}"
    )


def predict_main() -> None:
    parser = argparse.ArgumentParser(description="Predict diabetic retinopathy severity for one fundus image.")
    _config_argument(parser)
    parser.add_argument("--model", required=True, type=Path, help="Saved .keras model path.")
    parser.add_argument("--image", required=True, type=Path, help="Fundus image path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file path.")
    args = parser.parse_args()
    result = predict_image(args.model, args.image, load_config(args.config))
    if args.output:
        write_json(args.output, result)
        print(f"Wrote {args.output}")
    else:
        import json
        print(json.dumps(result, indent=2))
