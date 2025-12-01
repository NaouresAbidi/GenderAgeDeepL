import os

# Project root = folder that contains `src/`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data/raw folder inside the project
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# UTKFace full dataset
BASE_DIR = os.getenv("UTKFACE_BASE_DIR", os.path.join(DATA_RAW_DIR, "UTKFace"))

# Cropped children dataset
CROP_DIR = os.getenv("UTKFACE_CROP_DIR", os.path.join(DATA_RAW_DIR, "crop_part1"))

IMAGE_SIZE = (360, 360)
NUM_CHANNELS = 1

BUFFER_SIZE = 1000
BATCH_SIZE = 32
AUTOTUNE = "auto"

TRAIN_CACHE_FILE = "./train_cache.tf-data"
TEST_CACHE_FILE = "./test_cache.tf-data"

LEARNING_RATE = 1e-4
EPOCHS = 2

MODELS_DIR = os.path.join(PROJECT_ROOT, "src")

BASE_MODEL_PATH = os.path.join(MODELS_DIR, "best_age_gender_model.h5")
CHILD_TUNED_MODEL_PATH = os.path.join(MODELS_DIR, "best_age_gender_model_children_tuned.h5")

gender_mapping = {
    0: "Male",
    1: "Female"
}
