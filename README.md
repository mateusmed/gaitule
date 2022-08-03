
## Gaitule

Bem-vindos ao Gaitule, um sistema que busca simplificar a classificação e  
treinamento de modelos para reconhecimento computacional de imagens.

---
## Tecnologias

- python
- tensorflow
- numpy
- cv2
- matplotlib
- sklearn
- pickle

---

## Funcionamento

Supondo que você vá usar o Gaitule para fazer uma classificação
de cachorros e gatos.

```
    gaitule
        |_ archive
            |_ cachorro
            |_ gato
        |_ archive_modify
        |_ dataPickle
        |_ mask
        |_ model
        |_ properties
```

#### archive
 - Na pasta archive, você deverá colocar as imagens de treinamento 
em pastas respectivas com os nomes das classificações 
 
#### archive_modify
- Quando as imagens forem processadas, elas irão para o archive_modify

#### dataPickle
- em dataPickle ficará salvo o arquivo usado para serializar o objeto com o conteudo das das imagens processadas 

#### mask
- devido o processamento das imagens usar tecnicas como espelhamento e rotação, 
nesta pasta fica armazenado a imagem da mascara que será aplicada por cima de todas as imagens processadas  

#### model
- em model ficara armazenado o arquivo de treinamento gerado pelo tersorflow .h5

#### properties
- aqui fica armazenado o aquivo .json com as configurações necessárias para o sitema 
executar sem problemas, nele você precisa configurar alguns parametros:

```
{
  "model_file_save": "C:\\dev\\workspaceMateus\\gaitule\\model\\mymodel.h5",
  "pickler_file": "C:\\dev\\workspaceMateus\\gaitule\\dataPickle\\data.pickle",
  "dir_pictures_modify": "C:\\dev\\workspaceMateus\\gaitule\\archive_modify",
  "dir_pictures": "C:\\dev\\workspaceMateus\\gaitule\\archive",
  "circle_mask": "C:\\dev\\workspaceMateus\\gaitule\\mask\\circleMask.png",
  "categories": ["inflamatorias", "normais"],
  "degress_rotation_interval": 20,
  "normalize_size_image": 512
}
```

Adicione o diretorio do seu projeto respectivamente 
ou altere para o diretorio que preferir respeitando o funcionamento de cada pasta



---
## Executando


criando virtual env
```
python -m venv env
```

ligando virtual env:

```
cd /env/Scripts/activate
```


dentro do virtual env:

para instalar as dependencias execute o comando:

```
pip install -r requirements.txt
```

```
python src/run.py
```


---



