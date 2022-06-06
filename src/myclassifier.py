
import utils as utils
import numpy as np
import matplotlib.pyplot as plt

import os
import tensorflow as tf


from sklearn.model_selection import train_test_split


import global_properties as my_global
json = my_global.get_properties()

"""
===> softmax
A função de ativação softmax é usada em redes neurais de classificação. 
Ela força a saída de uma rede neural a representar a probabilidade dos 
dados serem de uma das classes definidas. Sem ela as saídas dos neurônios 
são simplesmente valores numéricos onde o maior indica a classe vencedora.

"""


def train_and_save_model(data):

    print(f'=============================')
    print(f'== train_and_save_model ==')
    print(f'=============================')

    model_file_save = json['model_file_save']

    if os.path.exists(model_file_save) is True:
        print(f'=============================')
        print(f'=== model already exists ====')
        print(f'=============================')
        return

    (feature, labels) = data

    x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1)

    input_layer = tf.keras.layers.Input([224, 224, 3])

    conv1 = tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   activation='relu')(input_layer)

    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(pool1)

    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(filters=96,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(pool2)

    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(filters=96,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='relu')(pool3)

    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2, 2))(conv4)

    flt1 = tf.keras.layers.Flatten()(pool4)

    dn1 = tf.keras.layers.Dense(512, activation='relu')(flt1)
    out = tf.keras.layers.Dense(5, activation='softmax')(dn1)

    model = tf.keras.Model(input_layer, out)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=100, epochs=10)

    model.save(model_file_save)