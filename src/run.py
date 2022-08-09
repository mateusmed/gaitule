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
    choice = input(""" Tem certeza que deseja fazer isso?
                          1: Sim
                          2: Não 
                          
                          Please enter your choice:""")

    if choice != '1' and choice != '2':
        print("Digite uma opcao valida 1 ou 2: ")
        you_right_this()

    return choice


def clean_img_processed():
    images.clean_archive_modify()


def test_image():
    image_path = input("""Digite o caminho da imagem: """)
    my_file = Path(image_path)

    if my_file.is_file():
        detect.test_image_set(image_path)
    else:
        print("caminho da imagem nao valido")
        return


def clean_trained():
    print("O processo de treinamento emana tempo...")
    option = you_right_this()

    if option == '1':
        print("Iniciando remocao")
        json = my_global.get_properties()

        model_file_save = json['model_file_save']
        pickler_file = json['pickler_file']

        if verify_path_exist(model_file_save):
            print("Removendo: {}.", model_file_save)
            os.remove(model_file_save)

        if verify_path_exist(pickler_file):
            print("Removendo: {}.", pickler_file)
            os.remove(pickler_file)

        print("ok...")

    if option == '2':
        print("Ok, a base de treinamento nao foi removida")
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
        print("Selecione uma das opcoes disponiveis")
        print("Tente novamente")
        print("========================================================= ")
        menu()


menu()

