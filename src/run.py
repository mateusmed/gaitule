import global_properties as my_global

import sys
import os
import shutil
import utils
import my_classifier
import detect
import images


def generate_data_image():
    print(f"====================")
    print(f"generate_data_image")
    print(f"====================")
    images.main()


def clean_img_processed():
    images.clean_archive_modify()

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

