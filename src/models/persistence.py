from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
BUNDLE_NAME = "trained_bundle.joblib"


def save_bundle(bundle, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / BUNDLE_NAME
    joblib.dump(bundle, path)
    return path


def load_bundle(model_dir=MODEL_DIR):
    path = Path(model_dir) / BUNDLE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no trained model bundle at {path} - run `python -m src.pipeline` first"
        )
    return joblib.load(path)
