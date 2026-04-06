import sys
from pathlib import Path
from app.trainer import train_model

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else base_dir / "data"
    train_model(str(dataset_path))