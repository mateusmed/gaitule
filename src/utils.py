import os
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import pickle


import global_properties as my_global
json = my_global.get_properties()


def open_image():
    categories = json['categories']
    dir_pictures = json['dir_pictures_modify']

    for category in categories:
        path = os.path.join(dir_pictures, category)
        label = categories.index(category)

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            image = cv2.imread(image_path)

            cv2.imshow('ok', image)
            # break

        # break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# gerando pickle com dados discretos, tanto do rgb quando da categoria
def create_data_pickle_by_images_and_labels():

    categories = json['categories']

    print(f'==============================================')
    print(f'=== create_data_pickle_by_images_and_labels ==')
    print(f'==============================================')

    data = []

    for category in categories:
        path = os.path.join(json['dir_pictures_modify'], category)
        label = categories.index(category)

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            image = cv2.imread(image_path)

            try:
                # set image on cv2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = np.array(image, dtype=np.float32)

                # set image and label on data global
                data.append([image, label])

            except Exception as e:
                pass

    print(f"number of pictures: {len(data)}")

    # Pickle in Python is primarily used in serializing and
    # deserializing a Python object structure.
    pik = open(json['pickler_file'], 'wb')
    pickle.dump(data, pik)
    pik.close()


def load_data_pickle():
    pickler_file = json['pickler_file']

    if os.path.exists(pickler_file) is False:
        create_data_pickle_by_images_and_labels()

    print(f'=======================')
    print(f'=== load_data_pickle ==')
    print(f'=======================')
    pick = open(pickler_file, 'rb')
    data = pickle.load(pick)
    pick.close()

    #embaralhe
    np.random.shuffle(data)

    feature = []
    labels = []

    for img, label in data:
        feature.append(img)
        labels.append(label)

    feature = np.array(feature, dtype=np.float32)
    labels = np.array(labels)

    # In your image classification, dividing by 255 is good because the whole range is in [0,1].
    # You can't have anything less than 0 and greater than 1.
    feature = feature/255.0

    return [feature, labels]





