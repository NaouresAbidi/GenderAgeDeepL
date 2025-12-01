import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.config import (
    CHILD_TUNED_MODEL_PATH,
    BASE_MODEL_PATH,
    BATCH_SIZE,
)
from src.data.dataset_utkface import load_full_dataset, load_children_subset
from src.features.preprocessing import make_supervised_dataset
from src.models.age_gender_model import build_classification_model


def configure_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_base_model():
    """
    Charge le modèle de base entraîné sur tout UTKFace.
    """
    custom_objects = {
        "mae": tf.keras.metrics.MeanAbsoluteError,
        "accuracy": tf.keras.metrics.binary_accuracy,
        "age_output_mae": tf.keras.metrics.MeanAbsoluteError,
        "gender_output_accuracy": tf.keras.metrics.binary_accuracy,
    }
    try:
        model = tf.keras.models.load_model(
            BASE_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
        )
        print(f"Loaded base model from {BASE_MODEL_PATH}")
    except Exception as e:
        print(f"Could not load base model, building a new one: {e}")
        model = build_classification_model()
    return model


def make_children_and_replay_datasets():
    """
    Crée:
      - child_train_ds, child_val_ds sur enfants (0–17)
      - replay_ds sur un sous-échantillon du dataset global (replay_frac)
    """
    # enfants 0–17
    c_paths, c_ages, c_genders = load_children_subset()
    C_paths_tr, C_paths_val, C_age_tr, C_age_val, C_gen_tr, C_gen_val = train_test_split(
        c_paths,
        c_ages,
        c_genders,
        test_size=0.2,
        random_state=42,
    )

    child_train_ds = make_supervised_dataset(
        C_paths_tr,
        C_age_tr,
        C_gen_tr,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )
    child_val_ds = make_supervised_dataset(
        C_paths_val,
        C_age_val,
        C_gen_val,
        shuffle=False,
        batch_size=BATCH_SIZE,
    )

    # replay dataset (= fraction du train global)
    data = load_full_dataset()
    replay_frac = 0.10
    data_train_replay = data.sample(frac=replay_frac, random_state=123)
    replay_ds = make_supervised_dataset(
        data_train_replay["image_path"].values,
        data_train_replay["age"].values,
        data_train_replay["gender"].values,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )

    return child_train_ds, child_val_ds, replay_ds


def fine_tune_children():
    """
    Reproduit ta phase de fine-tuning enfants + replay,
    et sauvegarde CHILD_TUNED_MODEL_PATH.
    """
    configure_gpu_memory_growth()
    model = load_base_model()

    # Stratégie de (dé)gel des couches comme dans le notebook
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.trainable = False
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Dropout, tf.keras.layers.Flatten, tf.keras.layers.Conv2D)):
            layer.trainable = True
            if isinstance(layer, tf.keras.layers.Conv2D):
                break

    child_train_ds, child_val_ds, replay_ds = make_children_and_replay_datasets()

    age_delta = 2.0
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss={
            "age_output": tf.keras.losses.Huber(delta=age_delta),
            "gender_output": "binary_crossentropy",
        },
        loss_weights={
            "age_output": 2.5,
            "gender_output": 1.0,
        },
        metrics={
            "age_output": ["mae"],
            "gender_output": ["accuracy"],
        },
    )

    cb = [
        EarlyStopping(
            monitor="val_age_output_mae",
            mode="min",
            patience=5,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_age_output_mae",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("Fine-tuning on children only (stage 1)")
    hist_child = model.fit(
        child_train_ds,
        validation_data=child_val_ds,
        epochs=12,
        callbacks=cb,
        verbose=1,
    )

    # Deuxième phase: tout trainable, mix enfants + replay
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-6),
        loss={
            "age_output": tf.keras.losses.Huber(delta=age_delta),
            "gender_output": "binary_crossentropy",
        },
        loss_weights={
            "age_output": 2.0,
            "gender_output": 1.0,
        },
        metrics={
            "age_output": ["mae"],
            "gender_output": ["accuracy"],
        },
    )

    mixed = tf.data.Dataset.sample_from_datasets(
        [child_train_ds, replay_ds],
        weights=[0.8, 0.2],
    )

    print("Fine-tuning on mixed children + replay (stage 2)")
    hist_mixed = model.fit(
        mixed,
        validation_data=child_val_ds,
        epochs=8,
        callbacks=cb,
        verbose=1,
    )

    # Evaluation sur val enfants
    eval_dict = model.evaluate(child_val_ds, return_dict=True)
    print("Child validation metrics after fine-tuning:", eval_dict)

    # Sauvegarde du modèle final fine-tuned
    model.save(CHILD_TUNED_MODEL_PATH)
    print(f"Saved fine-tuned model to {CHILD_TUNED_MODEL_PATH}")

    # Sauvegarde des historiques
    pd.DataFrame(hist_child.history).to_csv("history_children_stage1.csv", index=False)
    pd.DataFrame(hist_mixed.history).to_csv("history_children_stage2_mixed.csv", index=False)

    return model, hist_child, hist_mixed, eval_dict


if __name__ == "__main__":
    fine_tune_children()
