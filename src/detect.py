
import tensorflow as tf
import numpy as np

import cv2

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

    unmutable_image = cv2.imread(image_path)
    unmutable_image = cv2.cvtColor(unmutable_image, cv2.COLOR_BGR2RGB)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32)

    feature = np.array([image], dtype=np.float32)
    feature = feature / 255.0
    print(f'SHAPE: {image.shape}')

    prediction = model.predict(feature)

    prediction_percent_values = prediction[0]
    position_value_perdiction = np.argmax(prediction_percent_values)
    percent_prob = prediction_percent_values[position_value_perdiction]

    print('=================================')
    print(f'percent prob: {percent_prob}')
    print('=================================')
    print(f'prediction content: {prediction}')
    print(f'categoria: {categories[position_value_perdiction]}')
    print('=================================')


    # todo modularizar isso
    json_response = {}

    for index, value in enumerate(categories):
        percent_unformat = prediction_percent_values[index]
        json_response[value] = '{:.1%}'.format(percent_unformat)

    print(json_response)

    plt.subplot(1, 1, 1)
    plt.imshow(unmutable_image)
    plt.xlabel(f'Percent: {json_response}'
               f'\nPredicted:{categories[position_value_perdiction]}')

    plt.xticks([])
    plt.show()



def test_with_image_window(data):

    print(f'=============================')
    print(f'=== test_with_image_window ==')
    print(f'=============================')

    model_file = json['model_file_save']
    categories = json['categories']

    (feature, labels) = data

    print(f"feature: {feature}")
    print(f"labels: {labels}")

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
    print(f"categories: {categories}")
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