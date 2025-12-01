import numpy as np
import tensorflow as tf
from src.config import gender_mapping, CHILD_TUNED_MODEL_PATH
from src.features.preprocessing import preprocess_facial_image

def load_final_model():
    custom_objects = {
        'mae': tf.keras.metrics.MeanAbsoluteError,
        'accuracy': tf.keras.metrics.BinaryAccuracy,
        'age_output_mae': tf.keras.metrics.MeanAbsoluteError,
        'gender_output_accuracy': tf.keras.metrics.BinaryAccuracy,
        'Huber': tf.keras.losses.Huber
    }
    model = tf.keras.models.load_model(
        CHILD_TUNED_MODEL_PATH,
        custom_objects=custom_objects,
        compile=True
    )
    return model


def predict_single_image(image_path, model):
    preprocessed_tensor = preprocess_facial_image(
        tf.constant(image_path, dtype=tf.string),
        apply_equalization=False,
        final_normalization='[-1, 1]'
    )
    image_for_prediction = tf.expand_dims(preprocessed_tensor, axis=0)
    predictions = model.predict(image_for_prediction, verbose=0)

    pred_age = predictions[0][0][0]
    pred_gender_prob = predictions[1][0][0]
    pred_gender = 1 if pred_gender_prob > 0.5 else 0

    return {
        'age': int(np.round(pred_age)),
        'gender': gender_mapping[pred_gender],
        'gender_probability': float(pred_gender_prob)
    }


def predict_child_tta(image_path, model, n=6):
    img = preprocess_facial_image(tf.constant(image_path), False, '[-1, 1]')
    imgs = [img]
    imgs.append(tf.image.flip_left_right(img))
    for _ in range(n - 2):
        x = tf.image.random_brightness(img, 0.05)
        x = tf.image.random_contrast(x, 0.95, 1.05)
        imgs.append(x)
    batch = tf.stack(imgs, axis=0)
    age_pred, gender_pred = model.predict(batch, verbose=0)
    return np.round(np.mean(age_pred)).item(), float(np.mean(gender_pred))
