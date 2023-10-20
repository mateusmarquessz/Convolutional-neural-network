from PIL import Image
import os

pasta_origem = "/content/drive/MyDrive/Datasets/carro/porsche 911"

pasta_destino = "/content/drive/MyDrive/Datasets/carro/porsche 911 redimensionado"


novo_tamanho = (500, 500)

if not os.path.exists(pasta_destino):
    os.makedirs(pasta_destino)

for filename in os.listdir(pasta_origem):

    if filename.endswith((".jpg", ".jpeg", ".png", ".gif")):

        imagem = Image.open(os.path.join(pasta_origem, filename))

        imagem = imagem.convert("RGB")

        imagem_redimensionada = imagem.resize(novo_tamanho)


        imagem_redimensionada.save(os.path.join(pasta_destino, filename))

print("Todas as imagens foram redimensionadas com sucesso.")