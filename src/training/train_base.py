import tensorflow as tf
import pandas as pd

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config import (
    TRAIN_CACHE_FILE, TEST_CACHE_FILE,
    LEARNING_RATE, EPOCHS, BATCH_SIZE,
    BASE_MODEL_PATH
)
from src.data.dataset_utkface import load_full_dataset, train_test_split_utkface
from src.features.preprocessing import make_supervised_dataset
from src.models.age_gender_model import build_classification_model

def configure_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def get_datasets():
    data = load_full_dataset()
    train_df, test_df = train_test_split_utkface(data)

    train_paths = train_df["image_path"].values
    train_ages = train_df["age"].values
    train_genders = train_df["gender"].values

    test_paths = test_df["image_path"].values
    test_ages = test_df["age"].values
    test_genders = test_df["gender"].values

    ds_train = make_supervised_dataset(train_paths, train_ages, train_genders, shuffle=True, batch_size=BATCH_SIZE)
    ds_test = make_supervised_dataset(test_paths, test_ages, test_genders, shuffle=False, batch_size=BATCH_SIZE)

    # caching
    AUTOTUNE = tf.data.AUTOTUNE
    ds_train = ds_train.cache(TRAIN_CACHE_FILE).prefetch(AUTOTUNE)
    ds_test = ds_test.cache(TEST_CACHE_FILE).prefetch(AUTOTUNE)

    return ds_train, ds_test, train_df, test_df


def compile_model():
    model = build_classification_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss={
            'age_output': 'mae',
            'gender_output': 'binary_crossentropy'
        },
        metrics={
            'age_output': ['mae'],
            'gender_output': ['accuracy']
        }
    )
    return model


def train():
    configure_gpu_memory_growth()
    ds_train, ds_test, train_df, test_df = get_datasets()

    model = compile_model()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath=BASE_MODEL_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    callbacks_list = [early_stopping, model_checkpoint]

    train_steps_per_epoch = len(train_df) // BATCH_SIZE
    val_steps_per_epoch = len(test_df) // BATCH_SIZE

    history = model.fit(
        ds_train,
        steps_per_epoch=train_steps_per_epoch,
        epochs=EPOCHS,
        validation_data=ds_test,
        validation_steps=val_steps_per_epoch,
        callbacks=callbacks_list,
        verbose=1
    )

    print("Training finished.")

    print("\nFinal Evaluation on Test Data:")
    loss_results = model.evaluate(ds_test)
    metrics = model.metrics_names
    evaluation_dict = dict(zip(metrics, loss_results))
    print(evaluation_dict)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv("history_base_training.csv", index=False)

    return model, history, evaluation_dict


if __name__ == "__main__":
    train()
