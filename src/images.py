
import imutils

import os
import cv2
import random
import string
import glob
from PIL import Image

import global_properties as my_global
json = my_global.get_properties()


def generate_image_name():
    letters = string.ascii_uppercase
    new_random_letters = ''.join(random.choice(letters) for i in range(10))
    return new_random_letters + ".jpg"


def rename_files_with_numbers(path):
    for i, filename in enumerate(os.listdir(path)):
        os.rename(path + filename, path + str(i) + ".jpg")


def write_image(path_category, path_category_image, img_content):

    if not os.path.exists(path_category):
        os.makedirs(path_category)

    print(f"create file {path_category_image}")
    cv2.imwrite(path_category_image, img_content)


def open_image(img):
    cv2.imshow('None', img)
    cv2.waitKey(0)


def get_image(image_path):
    return cv2.imread(image_path)


def resize_image(img, number_pixels):
    return cv2.resize(img, (number_pixels, number_pixels), interpolation=cv2.INTER_CUBIC)


#not working
def resize_circle():
    im = Image.open('C:\\dev\\workspaceMateus\\pos\\13_tcc\\mask\\circleMask.png')
    size = 512, 512
    im.thumbnail(size, Image.ANTIALIAS)
    im.save('C:\\dev\\workspaceMateus\\pos\\13_tcc\\mask\\circleMask_2.png', "png")


# add mask in all images
def add_mask():

    dir_pictures_modify = json['dir_pictures_modify']
    categories = json['categories']

    for category in categories:
        path = os.path.join(dir_pictures_modify, category)

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)

            background = Image.open(image_path)
            foreground = Image.open('C:\\dev\\workspaceMateus\\pos\\13_tcc\\mask\\circleMask.png')

            background.paste(foreground, (0, 0), foreground)
            background.convert('RGB')
            background.save(image_path)

            print(f'added mask on {image_path}')


def test_add_circle_mask():
    try:
        background = Image.open('C:\\dev\\workspaceMateus\\pos\\13_tcc\\mask\\AA.jpg')
        foreground = Image.open('C:\\dev\\workspaceMateus\\pos\\13_tcc\\mask\\circleMask.png')

        background.paste(foreground, (0, 0), foreground)
        background.convert('RGB')
        background.save('C:\\dev\\workspaceMateus\\pos\\13_tcc\\mask\\ok.jpg')

    except Exception as e:
        print(str(e))


def get_mirror_image(img_content):
    return cv2.flip(img_content, 1)


def normalize_size_and_mirror():
    dir_pictures = json['dir_pictures']
    dir_pictures_modify = json['dir_pictures_modify']
    categories = json['categories']
    normalize_size_image = json['normalize_size_image']

    for category in categories:
        path = os.path.join(dir_pictures, category)

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            img_content = get_image(image_path)
            img_resize = resize_image(img_content, normalize_size_image)

            img_name = generate_image_name()

            new_image_path = os.path.join(dir_pictures_modify, category, img_name)
            new_path_category = os.path.join(dir_pictures_modify, category)
            write_image(new_path_category, new_image_path, img_resize)

            img_mirrored = get_mirror_image(img_resize)
            new_image_name = img_name.replace(".jpg", "_mirror.jpg")

            new_image_path = os.path.join(dir_pictures_modify, category, new_image_name)
            write_image(new_path_category, new_image_path, img_mirrored)


def work_in_data_image():
    normalize_size_and_mirror()
    generate_rotated_images()
    add_mask()


#não funcionando
def clean_archive_modify():
    content = f"{json['dir_pictures_modify']}\\*"
    files = glob.glob(content)

    for f in files:
        print(f"remove {f}")
        os.remove(f)


def generate_rotated_images():
    categories = json['categories']
    dir_pictures_modify = json['dir_pictures_modify']
    interval = json['degress_rotation_interval']

    if interval >= 360:
        raise Exception('interval cant be >= 360')

    degress = 360
    degress_split = interval

    for category in categories:
        path_category = os.path.join(dir_pictures_modify, category)

        for img_name in os.listdir(path_category):
            path_category_image = os.path.join(path_category, img_name)
            img_content = get_image(path_category_image)

            while degress_split < degress:
                img_content_rotated = imutils.rotate(img_content, degress_split)
                path_category_random_image = os.path.join(path_category, generate_image_name())

                write_image(path_category, path_category_random_image, img_content_rotated)
                degress_split = degress_split + degress_split

            degress_split = interval


def main():
    work_in_data_image()


main()
