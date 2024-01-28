import tensorflow as tf
from PIL import Image
import numpy as np
import glob
#from helpers import calculate_loss, truth_matrixes, xcenter_to_xmin_individual
from helpers import Yolo_Reshape, data, xywh2minmax, y_train, yolo_loss, calculate_loss, make_truth_matrixes
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import cv2
import pandas as pd


def YOLO_v1_architecture(input_layer):
    x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_layer)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2))(x)

    x = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    return x


def resNet_model(input_layer):
    x = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=input,
        input_shape=(448, 448, 3),
        pooling=max,
        classes=1000
    )

    return x


def custom_head(output):

    x = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(output)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1470, activation="sigmoid")(x)
    x = Yolo_Reshape(target_shape=(7,7,30))(x)

    return x


def transfer_learning(layers_to_learn, model):
    for layer in model.layers[:-1*layers_to_learn]:
        layer.trainable = False


if __name__ == "__main__":
    data = pd.read_csv("data/train_solution_bounding_boxes (1) copy.csv")

    data_boxes = data[['x_center', 'y_center', 'width', 'height']].to_numpy()

    y_train = make_truth_matrixes(data_boxes)

    input = tf.keras.Input(shape=(448, 448, 3))

    x = resNet_model(input)

    x = custom_head(x.output)

    model = tf.keras.Model(inputs = input, outputs = x)

    x_train = np.load('out_array.npy')

    model.compile(loss=yolo_loss, optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), run_eagerly=True)

    checkpoint = ModelCheckpoint("model_weights.h5", monitor='loss', verbose=1,
        save_best_only=True, mode='auto', save_freq=1)

    model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=1, callbacks=[checkpoint])


