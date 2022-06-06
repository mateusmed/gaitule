import global_properties as my_global

import os
import utils
import myclassifier
import detect
import images


def generate_data_image():
    print(f"====================")
    print(f"generate_data_image")
    print(f"====================")
    images.main()


def test_image():
    detect.test_image_set('C:\\dev\\workspaceMateus\\gaitule\\archive_modify\\virus\\USVVQVIQSK_mirror.jpg')


def clean_trained():
    json = my_global.get_properties()
    os.remove(json['model_file_save'])
    os.remove(json['pickler_file'])


def main1():
    # clean_trained()
    data = utils.load_data_pickle()
    myclassifier.train_and_save_model(data)
    detect.test_with_image_window(data)
    # utils.open_image()


generate_data_image()
# main1()



# def main2():
#     data = utils.load_data_pickle()
#     detect.test_with_image_window(data, utils.categories)


# print(f"{load_properties()}")
# print(f"{load_properties()}")