# Diabetic Retinopathy Detection with InceptionV3

A reproducible TensorFlow/Keras pipeline for classifying retinal fundus images
into the five ICDR diabetic-retinopathy grades: **No DR**, **Mild**,
**Moderate**, **Severe**, and **Proliferative DR**. It uses transfer learning
with an ImageNet-pretrained InceptionV3 backbone, augmentation, two-stage
fine-tuning, class weighting, and clinically useful evaluation metrics.

> **Clinical safety notice:** this repository is for research and education.
> It is not a diagnostic device and must not be used to make patient-care
> decisions without rigorous external validation, regulatory review, and
> qualified clinical oversight.

## Features

- InceptionV3 transfer learning for 299 x 299 fundus images
- Five-class ICDR/aptos-compatible labels (`0` through `4`)
- Reproducible train/validation/test splits, with patient-level isolation when
  a patient ID is supplied
- Frozen-backbone training followed by selective fine-tuning
- Accuracy, macro F1, quadratic weighted kappa, per-class recall, confusion
  matrix, and a saved classification report
- Command-line workflows for data preparation, training, evaluation, and
  single-image prediction
- GitHub Actions CI and focused tests

## Requirements

- Python 3.11, 3.12, or 3.13
- A GPU is recommended for training but is not required
- A labeled retinal-image dataset that you have permission to use

The project deliberately does **not** include images, labels, checkpoints, or
other patient-related artifacts.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For NVIDIA GPU acceleration, install the TensorFlow build and CUDA/cuDNN stack
that match your operating system and TensorFlow release. See the official
TensorFlow installation guide for the supported combinations.

## Dataset layout

Place data locally, never in Git:

```text
data/
├── images/
│   ├── 0001.png
│   └── 0002.png
└── labels.csv
```

The minimum `labels.csv` schema is:

```csv
image_id,diagnosis
0001,0
0002,3
```

The default configuration is compatible with the APTOS-style `image_id` /
`diagnosis` schema. `image_id` can include an extension or omit it; when it is
omitted, `.png` is appended. For clinical datasets with multiple eye images per
person, add a patient column and set `data.patient_column` in
[`configs/inception_v3.yaml`](configs/inception_v3.yaml). This is important to
prevent patient leakage across splits.

## Run the pipeline

All paths in the config are resolved relative to the config file.

```bash
# Validate labels and generate data/splits.csv.
dr-prepare --config configs/inception_v3.yaml

# Train frozen head, then fine-tune the last part of InceptionV3.
dr-train --config configs/inception_v3.yaml

# Evaluate the best model from a run directory.
dr-evaluate \
  --config configs/inception_v3.yaml \
  --model artifacts/runs/<run-id>/best_model.keras

# Predict one image.
dr-predict \
  --config configs/inception_v3.yaml \
  --model artifacts/runs/<run-id>/best_model.keras \
  --image path/to/fundus.png
```

Scripts in [`scripts/`](scripts) offer the same commands if you prefer
`python scripts/train.py ...` after installing the project.

## Outputs

Each run gets a timestamped directory under `artifacts/runs/` containing:

```text
best_model.keras            Best validation-loss checkpoint
class_names.json            Ordered label-to-class mapping
run_config.yaml             Exact training configuration
training_history.json       Combined phase-one and fine-tuning history
training_summary.json       Dataset sizes and class weights
logs/                       TensorBoard and CSV logs
evaluation/                 Metrics, report, and confusion matrix
```

These outputs are ignored by Git to protect data and keep the repository
lightweight. Start TensorBoard with:

```bash
tensorboard --logdir artifacts/runs
```

## Evaluation guidance

Accuracy can look strong even when rare, sight-threatening grades are missed.
Review macro F1, quadratic weighted kappa, the confusion matrix, and recall
for severe/proliferative classes. Before any real-world use, evaluate on an
independent, demographically representative external cohort; assess image
quality, calibration, subgroup performance, and referral thresholds with
clinical experts.

## Project structure

```text
configs/             Reproducible experiment configuration
data/                Local-only dataset location (ignored)
scripts/             Command wrappers
src/dr_classifier/   Training, data, model, and evaluation package
tests/               Automated tests
artifacts/           Local-only training outputs (ignored)
```

## License

No license has been granted yet. Add an explicit license before redistributing
or accepting outside contributions.
