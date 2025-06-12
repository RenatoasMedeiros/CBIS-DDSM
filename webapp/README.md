Testado com python 3.8 (anaconda).
# Required packages:
pip install pillow flask tensorflow

# Instruções:
Para testar esta aplicação foi usado o melhor modelo criado durante os treinos (modelo_T_com_data_augmentation_adam.keras)
Que dá resize das imagem para 150x150.

Se quiser testar com outro modelo (modelo_S) da lista, tem que alterar na linha 104 do app.py:
img = img.resize((32, 32))

modelos_T:
img = img.resize((150, 150))

# NOTA: 
Não esquecer de tambem mudar o nome do modelo que quer utilizar para 'bestmodel.keras' ou alterar o nome na linha 76!

# Trabalho realizado por: Carlos Franco (2212574) Renato Medeiros (2211029)