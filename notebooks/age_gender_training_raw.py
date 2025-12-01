import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf

from src.config import gender_mapping, CROP_DIR
from src.data.dataset_utkface import load_full_dataset
from src.training.train_base import train
from src.training.fine_tune_children import fine_tune_children
from src.inference.predict import (
    load_final_model,
    predict_single_image,
    predict_child_tta,
)

warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


Data = load_full_dataset()
print("Data shape:", Data.shape)
Data.head()

print(Data[["age", "gender"]].describe())

plt.figure(figsize=(8, 4))
sns.histplot(Data["age"], kde=True, bins=40)
plt.title("Distribution des âges dans UTKFace")
plt.xlabel("Âge")
plt.ylabel("Nombre d'images")
plt.show()

from sklearn.model_selection import train_test_split

Data_train, Data_test = train_test_split(
    Data, test_size=0.30, random_state=42
)

print(f"Total images: {len(Data)}")
print(f"Training Images (70%): {len(Data_train)}")
print(f"Testing Images (30%): {len(Data_test)}")

num_images_to_show = 4

fig, axes = plt.subplots(1, num_images_to_show, figsize=(16, 4))
plt.suptitle("Sample Images (brutes)", fontsize=16)

sample_rows = Data_train.sample(num_images_to_show, random_state=0)

for ax, (_, row) in zip(axes, sample_rows.iterrows()):
    path = row["image_path"]
    img = Image.open(path)
    gender = row["gender"]
    ax.imshow(img)
    ax.set_title(f"Age: {row['age']}  Gender: {gender_mapping[gender]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
base_model, base_history, base_eval = train()
print("\nBase model evaluation on test set:")
print(base_eval)

hist_df = pd.DataFrame(base_history.history)
epochs_ran = hist_df.shape[0]
hist_df.head()

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(hist_df["loss"], label="Train Loss")
plt.plot(hist_df["val_loss"], label="Val Loss")
plt.title("Base Model - Total Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(hist_df["age_output_mae"], label="Train Age MAE")
plt.plot(hist_df["val_age_output_mae"], label="Val Age MAE")
plt.title("Base Model - Age Prediction (MAE)")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error (years)")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(hist_df["gender_output_accuracy"], label="Train Gender Acc")
plt.plot(hist_df["val_gender_output_accuracy"], label="Val Gender Acc")
plt.title("Base Model - Gender Prediction (Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

tuned_model, hist_child, hist_mixed, eval_children = fine_tune_children()
print("\nFine-tuned model evaluation on children validation set:")
print(eval_children)

hist_child_df = pd.DataFrame(hist_child.history)
hist_child_df.head()

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(hist_child_df["loss"], label="Train Loss")
plt.plot(hist_child_df["val_loss"], label="Val Loss")
plt.title("Fine-tuning Stage 1 (Children) - Total Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(hist_child_df["age_output_mae"], label="Train Age MAE")
plt.plot(hist_child_df["val_age_output_mae"], label="Val Age MAE")
plt.title("Fine-tuning Stage 1 - Age MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE (years)")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(hist_child_df["gender_output_accuracy"], label="Train Gender Acc")
plt.plot(hist_child_df["val_gender_output_accuracy"], label="Val Gender Acc")
plt.title("Fine-tuning Stage 1 - Gender Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
hist_mixed_df = pd.DataFrame(hist_mixed.history)
hist_mixed_df.head()

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(hist_mixed_df["loss"], label="Train Loss")
plt.plot(hist_mixed_df["val_loss"], label="Val Loss")
plt.title("Fine-tuning Stage 2 (Mixed) - Total Loss")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(hist_mixed_df["age_output_mae"], label="Train Age MAE")
plt.plot(hist_mixed_df["val_age_output_mae"], label="Val Age MAE")
plt.title("Fine-tuning Stage 2 - Age MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE (years)")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(hist_mixed_df["gender_output_accuracy"], label="Train Gender Acc")
plt.plot(hist_mixed_df["val_gender_output_accuracy"], label="Val Gender Acc")
plt.title("Fine-tuning Stage 2 - Gender Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

final_model = load_final_model()

TEST_IMAGE_FILE = os.path.join(
    CROP_DIR,
    "10_0_0_20170110224238891.jpg.chip.jpg"
)

if not os.path.exists(TEST_IMAGE_FILE):
    print(f"Error: Test image not found at {TEST_IMAGE_FILE}")
    print("Please ensure the crop_part1 dataset is available.")
else:
    results = predict_single_image(TEST_IMAGE_FILE, final_model)
    print("Prediction on single test image:")
    print(results)

    original_img = Image.open(TEST_IMAGE_FILE)

    plt.figure(figsize=(6, 6))
    plt.imshow(original_img, cmap="gray")
    pred_text = (
        f"Age: {results['age']} yrs\n"
        f"Gender: {results['gender']} ({results['gender_probability']:.2f})"
    )
    plt.title(pred_text, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()
age_groups = {
    "Children (Age 0-12)": (0, 12, 3),
    "Teens (Age 13-17)": (13, 17, 3),
    "Young Adults (Age 18-21)": (18, 21, 3),
    "Adults (Age 22+)": (22, 116, 3),
}

final_test_paths = []
final_test_info = []

for group_name, (min_age, max_age, num_samples) in age_groups.items():
    group_data = Data[(Data["age"] >= min_age) & (Data["age"] <= max_age)]

    if len(group_data) < num_samples:
        print(f"Warning: Only {len(group_data)} images available for {group_name}. Sampling all.")
        sampled_data = group_data
    else:
        sampled_data = group_data.sample(n=num_samples, random_state=42)

    for _, row in sampled_data.iterrows():
        final_test_paths.append(row["image_path"])
        final_test_info.append(
            {
                "path": row["image_path"],
                "true_age": row["age"],
                "true_gender": gender_mapping[row["gender"]],
                "group": group_name,
            }
        )

print(f"Collected {len(final_test_paths)} images for final testing.")

N_SAMPLES = len(final_test_paths)
N_COLS = 3
N_ROWS = int(np.ceil(N_SAMPLES / N_COLS))

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(5 * N_COLS, 6 * N_ROWS))
axes = axes.flatten()

for i, info in enumerate(final_test_info):
    results = predict_single_image(info["path"], final_model)

    original_img = Image.open(info["path"])

    gender_correct = results["gender"] == info["true_gender"]
    age_diff = abs(results["age"] - info["true_age"])

    gender_status = "CORRECT" if gender_correct else "WRONG"
    age_status = f"Diff: {age_diff} yrs"

    ax = axes[i]
    ax.imshow(original_img)

    title_text = (
        f"Group: {info['group']}\n"
        f"True: Age {info['true_age']}, Gender {info['true_gender']}\n"
        f"Pred: Age {results['age']} ({age_status})\n"
        f"Pred: Gender {results['gender']} ({results['gender_probability']:.2f}) - {gender_status}"
    )

    ax.set_title(title_text, fontsize=9, fontweight="bold")
    ax.axis("off")

for j in range(len(final_test_paths), N_ROWS * N_COLS):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
