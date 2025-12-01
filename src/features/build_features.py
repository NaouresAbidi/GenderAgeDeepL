import os
import pandas as pd
import tensorflow as tf

from src.config import PROJECT_ROOT, BATCH_SIZE, TRAIN_CACHE_FILE, TEST_CACHE_FILE
from src.features.preprocessing import make_supervised_dataset


def load_processed_splits():
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Processed splits not found. Expected:\n  {train_path}\n  {test_path}\n"
            "Run `py -3.11 -m src.data.make_dataset` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def build_datasets(batch_size: int = BATCH_SIZE):
    train_df, test_df = load_processed_splits()

    train_paths = train_df["image_path"].values
    train_ages = train_df["age"].values
    train_genders = train_df["gender"].values

    test_paths = test_df["image_path"].values
    test_ages = test_df["age"].values
    test_genders = test_df["gender"].values

    ds_train = make_supervised_dataset(
        train_paths,
        train_ages,
        train_genders,
        shuffle=True,
        batch_size=batch_size,
    )

    ds_test = make_supervised_dataset(
        test_paths,
        test_ages,
        test_genders,
        shuffle=False,
        batch_size=batch_size,
    )

    autotune = tf.data.AUTOTUNE
    ds_train = ds_train.cache(TRAIN_CACHE_FILE).prefetch(autotune)
    ds_test = ds_test.cache(TEST_CACHE_FILE).prefetch(autotune)

    return ds_train, ds_test, train_df, test_df


if __name__ == "__main__":
    ds_train, ds_test, train_df, test_df = build_datasets()
    print("[build_features] Train df shape:", train_df.shape)
    print("[build_features] Test df shape:", test_df.shape)

    for batch_imgs, (batch_ages, batch_genders) in ds_train.take(1):
        print("[build_features] One batch images shape:", batch_imgs.shape)
        print("[build_features] One batch ages shape:", batch_ages.shape)
        print("[build_features] One batch genders shape:", batch_genders.shape)
