import global_properties as my_global

import sys
import os
import utils
import my_classifier
import detect
import images
from pathlib import Path


def generate_data_image():
    print(f"====================")
    print(f"generate_data_image")
    print(f"====================")
    images.main()


def verify_path_exist(path_file):
    my_file = Path(path_file)

    if my_file.is_file():
        return True
    else:
        return False


def you_right_this():
    choice = input(""" Are you sure you want to do this?
                          1: Yes
                          2: No
                          
                          Please enter your choice:""")

    if choice != '1' and choice != '2':
        print("Select a valid option 1 or 2: ")
        you_right_this()

    return choice


def clean_img_processed():
    images.clean_archive_modify()


def test_image():
    image_path = input("""Please, provide path of image: """)
    my_file = Path(image_path)

    if my_file.is_file():
        detect.test_image_set(image_path)
    else:
        print("path of image not valid")
        return


def clean_trained():
    print("The training process takes time...")
    option = you_right_this()

    if option == '1':
        print("Init removing files")
        json = my_global.get_properties()

        model_file_save = json['model_file_save']
        pickler_file = json['pickler_file']

        if verify_path_exist(model_file_save):
            print("Removing: {}.", model_file_save)
            os.remove(model_file_save)

        if verify_path_exist(pickler_file):
            print("Removing: {}.", pickler_file)
            os.remove(pickler_file)

        print("ok...")

    if option == '2':
        print("Ok, the training base was not removed")
        return


def trained():
    data = utils.load_data_pickle()
    my_classifier.train_and_save_model(data)


def test_img_random():
    data = utils.load_data_pickle()
    detect.test_with_image_window(data)


def menu():
    print("=============== GAITULE - img classifier ===============")

    choice = input("""
                      1: Train the model
                      2: Clear training data
                      3: Process images
                      4: Clear processed images
                      5: Test random images
                      6: Test a chosen image
                      7: Exit
                      
                      Please enter your choice:""")

    if choice == '1':
        trained()
        menu()

    elif choice == '2':
        clean_trained()
        menu()

    elif choice == '3':
        generate_data_image()
        menu()

    elif choice == '4':
        clean_img_processed()
        menu()

    elif choice == '5':
        test_img_random()
        menu()

    elif choice == '6':
        test_image()
        menu()

    elif choice == '7':
        sys.exit()

    else:
        print("Select a valid option")
        print("Try again")
        print("========================================================= ")
        menu()


menu()

