import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

arquivo = open('preditor.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('preditor.h5')

imagem_teste = image.load_img('./val/real_normal.jpeg',
                              target_size = (64,64))

testar = imagem_teste
testar = image.img_to_array(testar)
testar /= 255
testar = np.expand_dims(testar, axis = 0)

previsao = classificador.predict(testar)[0][0]

print('Probabilidade de ter Pneumonia:', np.round((previsao*100), 1),"%\n")
