
import tensorflow as tf
import numpy as np

import cv2
import utils

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import global_properties as my_global
json = my_global.get_properties()


def test_image_set(image_path):
    print('=================================')
    print('TESTE IMAGE SET')
    print('=================================')

    categories = json['categories']
    model_file = json['model_file_save']
    model = tf.keras.models.load_model(model_file)

    data = utils.load_data_pickle()
    (feature, labels) = data

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32)

    feature = np.array([image], dtype=np.float32)
    feature = feature / 255.0
    print(f'SHAPE: {image.shape}')

    prediction = model.predict(feature)
    print('=================================')
    print(f'prediction content: {prediction}')
    print(f'categoria: {categories[np.argmax(prediction[0])]}')
    print('=================================')

    # todo imagem abrindo quebrada, resolver
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.xlabel('Predicted:' + categories[np.argmax(prediction[0])])

    plt.show()



def test_with_image_window(data):

    print(f'=============================')
    print(f'=== test_with_image_window ==')
    print(f'=============================')

    model_file = json['model_file_save']
    categories = json['categories']

    (feature, labels) = data

    x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1)

    model = tf.keras.models.load_model(model_file)

    model.evaluate(x_test, y_test, verbose=1)

    print(f"x_train {len(x_train)}")
    print(f"x_test {len(x_test)}")
    print(f"y_train {len(y_train)}")
    print(f"y_test {len(y_test)}")

    print(f"=========> x_test {x_test[0].shape}")

    prediction = model.predict(x_test)

    print(f"x_test: {x_test[0]}")
    print(f"prediction: {categories}")
    print("=======================================")
    print(f"prediction: {prediction[0]}")
    print("=======================================")
    print(f"prediction: {np.argmax(prediction[0])}")

    plt.figure(figsize=(9, 9))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_test[i])
        plt.xlabel('Actual:' + categories[y_test[i]] +
                   '\n predicted:' + categories[np.argmax(prediction[i])])
        plt.xticks([])

    plt.show()


# test_image_set()
# test_with_image_window()