import pathlib
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
import keras
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
import matplotlib.pyplot as plt

dataset_url = '../Data'
data_dir = pathlib.Path(dataset_url)

batch_size = 64
img_height = 128
img_width = 128

with tf.device('/GPU:0'):
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5, nesterov=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    model = Sequential([
        # #VGG19
        # layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_height, img_width, 3)),
        # layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D((2, 2), strides=(2, 2)),
        #
        # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D((2, 2), strides=(2, 2)),
        #
        # layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D((2, 2), strides=(2, 2)),
        #
        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D((2, 2), strides=(2, 2)),
        #
        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        # layers.MaxPooling2D((2, 2), strides=(2, 2)),
        #
        # layers.Flatten(),
        #
        # layers.Dense(4096, activation='relu'),
        # layers.Dense(4096, activation='relu'),
        # layers.Dense(num_classes, activation='softmax')


        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    epochs=50
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()