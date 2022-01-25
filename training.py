#IMPORTANDO BIBLIOTECAS
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


#TREINANDO REDE NEURAL
classificador = Sequential()
classificador.add(Conv2D(64, (3,3), input_shape=(64, 64, 3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2, 2)))
classificador.add(Flatten())
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.1))
classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#ESTABELECENDO PARAMETROS DA REDE NEURAL
gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('train',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')

base_teste = gerador_teste.flow_from_directory('test',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')

#EXECUTANDO TREINAMENTO
classificador.fit_generator(base_treinamento, steps_per_epoch = 3800 / 32,
                            epochs = 10, validation_data = base_teste,
                            validation_steps = 390 / 32)

#EXPORTANDO PREDITOR
classificador_json = classificador.to_json()
with open('preditor.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('preditor.h5')
