import os
import random
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import BASE_DIR, CROP_DIR

def load_full_dataset(shuffle_seed: int = 42) -> pd.DataFrame:
    image_filenames = os.listdir(BASE_DIR)
    random.seed(shuffle_seed)
    random.shuffle(image_filenames)

    age_labels = []
    gender_labels = []
    image_paths = []

    for image in tqdm(image_filenames, desc="Loading UTKFace filenames"):
        image_path = os.path.join(BASE_DIR, image)
        age_gender = image.split('_')
        age_label = int(age_gender[0])
        gender_label = int(age_gender[1])

        age_labels.append(age_label)
        gender_labels.append(gender_label)
        image_paths.append(image_path)

    data = pd.DataFrame({
        "image_path": image_paths,
        "age": age_labels,
        "gender": gender_labels
    })
    return data


def train_test_split_utkface(data: pd.DataFrame, test_size: float = 0.30, seed: int = 42):
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=seed
    )
    return train_df, test_df


def load_children_subset():
    """Subset of 0–17 y from crop_part1, as you did at the end of the notebook."""
    crop_files = [p for p in glob.glob(os.path.join(CROP_DIR, "*.jpg*"))]

    c_paths, c_ages, c_genders = [], [], []
    for p in crop_files:
        base = os.path.basename(p).split('.')[0]
        age, gender = base.split('_')[:2]
        age, gender = int(age), int(gender)
        if 0 <= age <= 17:
            c_paths.append(p)
            c_ages.append(age)
            c_genders.append(gender)

    return c_paths, c_ages, c_genders
