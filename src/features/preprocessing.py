import numpy as np
import tensorflow as tf
import cv2

from src.config import IMAGE_SIZE, NUM_CHANNELS, BATCH_SIZE, BUFFER_SIZE
from src.config import gender_mapping

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)])
def tf_equalize_histogram(image):
    def equalize_and_normalize(img_tensor):
        img_array = img_tensor.numpy()
        img_uint8 = np.clip(img_array.astype(np.uint8), 0, 255)
        equalized_img = cv2.equalizeHist(img_uint8[:, :, 0])
        equalized_img = equalized_img.astype(np.float32) / 255.0
        return np.expand_dims(equalized_img, axis=-1)
    equalized_image = tf.py_function(
        func=equalize_and_normalize,
        inp=[image * 255.0],
        Tout=tf.float32
    )
    equalized_image.set_shape(image.shape)
    return equalized_image


def preprocess_facial_image(file_path, apply_equalization=True, final_normalization='[-1, 1]'):
    img_raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(img_raw, channels=NUM_CHANNELS)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    image = image / 255.0

    if apply_equalization:
        image = tf_equalize_histogram(image)

    if final_normalization == '[-1, 1]':
        image = (image * 2.0) - 1.0
    elif final_normalization == '[0, 1]':
        pass
    else:
        raise ValueError("Invalid final_normalization. Choose '[-1, 1]' or '[0, 1]'.")
    return image


def load_and_preprocess_data(file_path, age_label, gender_label):
    image = preprocess_facial_image(
        file_path,
        apply_equalization=False,
        final_normalization='[-1, 1]'
    )
    age_label = tf.cast(age_label, tf.float32)
    gender_label = tf.cast(gender_label, tf.float32)

    return image, (age_label, gender_label)


def make_supervised_dataset(paths, ages, genders, shuffle=True, batch_size=BATCH_SIZE):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((paths, ages, genders))
    if shuffle:
        ds = ds.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, a, g: load_and_preprocess_data(p, a, g),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds
