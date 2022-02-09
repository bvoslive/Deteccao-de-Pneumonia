
# Detecção de Pneumonia

Este projeto faz parte do desafio do Kaggle cujo os dados estão disponíveis em https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia.

## Objetivo

Este projeto tem o objetivo de criar uma Rede Neural Convolucinal em conjunto com uma base de dados de amostras de imagens Raio-X contendo a região tórax para predição de pacientes com pneumonia.

--Status do Projeto: On-Hold

## Métodos Usados















Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

RESULTADO:
loss: 0.1805
accuracy: 0.9346
val_loss: 0.3744
val_accuracy: 0.8798

#_________________________________________________________________
 Layer (type)                Output Shape              Param #
#=================================================================
 conv2d (Conv2D)             (None, 62, 62, 64)        1792

 batch_normalization (BatchN  (None, 62, 62, 64)       256
 ormalization)

 max_pooling2d (MaxPooling2D  (None, 31, 31, 64)       0
 )

 flatten (Flatten)           (None, 61504)             0

 dense (Dense)               (None, 128)               7872640

 dropout (Dropout)           (None, 128)               0

 dense_1 (Dense)             (None, 128)               16512

 dropout_1 (Dropout)         (None, 128)               0

 dense_2 (Dense)             (None, 1)                 129

=================================================================
Total params: 7,891,329
Trainable params: 7,891,201
Non-trainable params: 128


RESULTADO VALIDAÇÃO:

Normal:
Probabilidade de ter Pneumonia: 0.1 %

Doente:
Probabilidade de ter Pneumonia: 89.4 %
