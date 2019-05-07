import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

arquivo = open('pneumonia.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)


classificador.load_weights('pneumonia.h5')


imagem_teste = image.load_img('dataset/val/PNEUMONIA/person1954_bacteria_4886.jpeg',
                              target_size = (64,64))

testar = imagem_teste
testar = image.img_to_array(testar)
testar /= 255
testar = np.expand_dims(testar, axis = 0)

previsao = classificador.predict(testar)[0][0]




print('Probabilidade de ter Pneumonia:', np.round((previsao*100), 1),"%\n")

print("Neural Networks description: loss: 0.1432 - acc: 0.9443 - val_loss: 0.4090 - val_acc: 0.8550")

