import unittest

import cv2
import tensorflow as tf

import keras
from keras import layers
import matplotlib.pyplot as plt

from src.global_properties import get_properties


class Testing(unittest.TestCase):

    def test_verify_filters_image(self):

        # oq a convolução faz exatamente? a convolução extrai as informações da imagem
        # e as transforma em "mapa" o kernel size representa a complexidade da informações
        # quanto maior o kernel maior a complexidade, neste exemplo estamos printando os "mapas"
        # os seja as caracteristicas mais marcantes da imagem

        img_path = "C:\dev\workspaceMateus\gaitule\mask\destiny.jpg"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # utilizando escala cinza para simplificar
        img = cv2.resize(img, (224, 224))
        height, width = img.shape

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        model = keras.Sequential()
        model.add(layers.Conv2D(input_shape=(height, width, 1),     #uma dimensão, apenas uma cor
                                filters=64,
                                kernel_size=(3, 3)))

        model.summary()

        filters, _ = model.layers[0].get_weights()
        f_min, f_max = filters.min(), filters.max()

        # normalizando valores para conseguir enxergar o resultado dos filtros com mais clareza
        filters = (filters - f_min) / (f_max - f_min)

        plt.figure(figsize=(9, 9))

        i = 30

        for i in range(9):
            plt.subplot(3, 3, i + 1)

            f = filters[:, :, :, i]  # altura, largura, dimensão e numero do filtro
            f = cv2.resize(f, (250, 250), interpolation=cv2.INTER_NEAREST)  # mantenha os pixels mesmo com o resize

            plt.imshow(f)
            plt.xlabel(f"Position: {i}")
            plt.xticks([])

        plt.show()

    def test_max_pool(self):

        """
        input:
        [[
           [[1.] [2.] [3.]]
           [[4.] [5.] [6.]]
           [[7.] [8.] [9.]]
        ]]

        result:
        [[
           [[5.] [6.]]
           [[8.] [9.]]
        ]]

        verificação 2x2 em uma matriz 3x3
        pegando o maior valor daquele quadrado verificador


        strides=(3, 3),
        significa os passos a o marcador vai andar, por exemplo

        [[
                     0   1   2   3
                 0  [1., 2., 3., 16., 14.],
                 1  [4., 5., 6., 11., 13.],
                 2  [19., 8., 9., 12., 15.]
                 3
        ]]

        result:
        [[
           [[5.] [16.]]
        ]]

        se andei 3 pra direita peguei para calculo o quadrante 2x2 de [16, 14,] [11, 13]
        depois de andar as colunas, vou andar as linhas, por tanto não existem valores
        o quadrante 3 das linhas não existe, por isso o resultado é [[5.] [16.]]

        """

        x = tf.constant([[1., 2., 3., 16., 14.],
                         [4., 5., 6., 11., 13.],
                         [19., 8., 9., 12., 15.]])

        x = tf.reshape(x, [1, 3, 5, 1])

        print(x)
        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                   strides=(3, 3),
                                                   padding='valid')

        response = max_pool_2d(x)
        print("============")
        print(response)

    def test_percent_convert_value(self):
        your_value = 1 / 3.0
        print(your_value)

        print('{:.1%}'.format(your_value))

    def test_parse_percent(self):
        matrix = [[0.97848517, 0.02151481]]
        perdiction_percent_values = matrix[0]

        json = get_properties()
        categories_list = json['categories']

        json_response = {}

        for index, value in enumerate(categories_list):
            percent_unformat = perdiction_percent_values[index]

            json_response[value] = '{:.1%}'.format(percent_unformat)

        print(json_response)

    def test_verify_properties_length(self):
        json = get_properties()
        categories_list = json['categories']
        print(f"categories {categories_list}")
        self.assertEqual(len(categories_list), 2)

    def test_string(self):
        a = 'some'
        b = 'some'
        self.assertEqual(a, b)

    def test_boolean(self):
        a = True
        b = True
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
