import os
import pandas as pd

from src.config import PROJECT_ROOT
from src.data.dataset_utkface import load_full_dataset, train_test_split_utkface


def main(test_size: float = 0.3, random_state: int = 42) -> None:
    """
    Charge l'ensemble du dataset UTKFace, effectue un split train/test
    et enregistre les fichiers dans data/processed/train.csv et test.csv.
    """
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("[make_dataset] Loading full UTKFace dataset...")
    df = load_full_dataset()
    print(f"[make_dataset] Full dataset shape: {df.shape}")

    train_df, test_df = train_test_split_utkface(df, test_size=test_size, seed=random_state)

    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[make_dataset] Saved train split to: {train_path} (shape={train_df.shape})")
    print(f"[make_dataset] Saved test split to:  {test_path} (shape={test_df.shape})")


if __name__ == "__main__":
    main()
