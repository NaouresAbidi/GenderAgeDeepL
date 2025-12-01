# src/models/age_gender_model.py

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from keras.models import Model
from src.config import IMAGE_SIZE, NUM_CHANNELS

def build_classification_model(input_shape=None):
    if input_shape is None:
        input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], NUM_CHANNELS)

    input_tensor = Input(shape=input_shape, name='input_image')

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(5, 5))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    shared_dense = Dense(512, activation='relu')(x)
    shared_dense = Dropout(0.5)(shared_dense)

    gender_output = Dense(
        1, activation='sigmoid', name='gender_output'
    )(shared_dense)

    age_output = Dense(
        1, activation='linear', name='age_output'
    )(shared_dense)

    model = Model(
        inputs=input_tensor,
        outputs=[age_output, gender_output],
        name='Age_Gender_Classifier'
    )
    return model
