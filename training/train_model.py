from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.ml_model import train_and_save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the custom logistic regression sentiment model.")
    parser.add_argument("--sample-size", type=int, default=6000, help="Number of Amazon review rows used for training.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of the sample reserved for testing.")
    args = parser.parse_args()

    report = train_and_save_model(sample_size=args.sample_size, test_size=args.test_size)
    print("Training complete.")
    print(f"Dataset size: {report['dataset_size']}")
    print(f"Training samples: {report['training_samples']}")
    print(f"Testing samples: {report['testing_samples']}")
    print(f"TF-IDF features: {report['tfidf_features']}")
    print("Logistic Regression metrics:", report["model_metrics"]["logistic_regression"])


if __name__ == "__main__":
    main()
