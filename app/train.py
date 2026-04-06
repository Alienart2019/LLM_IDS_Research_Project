import sys
from pathlib import Path
from app.trainer import train_model

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else project_root / "data"

    if not Path(dataset_path).is_absolute():
        dataset_path = project_root / dataset_path

    train_model(str(dataset_path))