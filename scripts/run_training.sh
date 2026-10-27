#!/usr/bin/env bash
# ==============================================================================
# Diabetic Retinopathy Detection - Training Pipeline Runner
# ==============================================================================

set -e

CONFIG_FILE="config/config.yaml"
PHASE="both"

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --phase)
      PHASE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./scripts/run_training.sh [--config config_path] [--phase both|phase1|phase2]"
      exit 1
      ;;
  esac
done

echo "=========================================================="
echo " Starting Diabetic Retinopathy Training with InceptionV3"
echo "=========================================================="
echo " Config: $CONFIG_FILE"
echo " Phase:  $PHASE"
echo " Time:   $(date)"
echo "=========================================================="

# Check Python environment
python3 -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"

# Execute training
python3 -m src.train --config "$CONFIG_FILE" --phase "$PHASE"

echo "=========================================================="
echo " Training finished successfully!"
echo " Checkpoints saved in: models/"
echo " Logs available in: logs/"
echo "=========================================================="
