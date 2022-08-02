import global_properties as my_global

import sys
import os
import shutil
import utils
import myclassifier
import detect
import images


def generate_data_image():
    print(f"====================")
    print(f"generate_data_image")
    print(f"====================")
    images.main()


def clean_img_processed():
    #todo  verificar se o usuario realmente quer deletar os dados do modelo treinado,
    #todo pois o treinamento exige tempo de processamento
    json = my_global.get_properties()

    folder = json['dir_pictures_modify']

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    print("all files removed")

def test_image():
    choice = input("""Digite o caminho da imagem: """)
    #todo verificar se a imagem existe

    detect.test_image_set(choice)

def clean_trained():
    #todo  verificar se o usuario realmente quer deletar os dados do modelo treinado,
    #todo pois o treinamento exige tempo de processamento
    json = my_global.get_properties()
    os.remove(json['model_file_save'])
    os.remove(json['pickler_file'])


def trained():
    data = utils.load_data_pickle()
    myclassifier.train_and_save_model(data)


def test_img_random():
    data = utils.load_data_pickle()
    detect.test_with_image_window(data)


def menu():
    print("=============== GAITULE - img classifier ===============")

    choice = input("""
                      1: Treinar o modelo
                      2: Limpar dados de treinamento
                      3: Processar imagens
                      4: Limpar imagens processadas
                      5: Testar imagens aleatorias
                      6: Testar imagem que você escolher
                      7: Sair
                      
                      Please enter your choice: """)

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
        sys.exit

    else:
        print("You must only select either A or B")
        print("Please try again")
        print("========================================================= ")
        menu()


def register():
    print("function register example")
    pass


def login():
    print("function login example")
    pass


# the program is initiated, so to speak, here
menu()


#generate_data_image()
# main1()
