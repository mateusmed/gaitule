import unittest
import numpy as np

from src.global_properties import get_properties


class Testing(unittest.TestCase):

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