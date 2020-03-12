from createDataFrame import tShirtsDF
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import to_categorical
from keras.layers.advanced_activations import LeakyReLU
from scipy import ndimage
import tensorflow as tf
from PIL import Image
import pickle

# Importo el DataFrame anteriormente creado para posteriormente entrenarlo
tShirtsDF = pd.read_pickle("tShirtsDF.pkl")

# Selecciono la X,y para el modelo a entrenar
img_rows, img_cols, img_rgb = 256, 256, 3

X = np.stack(np.array(tShirtsDF.Camisetas),axis=2).swapaxes(2,0)
y = np.array(tShirtsDF.Labels)
print("Shapes X={} y={}".format(X.shape,y.shape))

classes = np.unique(y)
nClasses = len(classes)
print('Total numero de etiquetas: ', nClasses)
print('Tipos de etiquetas: ', classes)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
print('Training data shape : ', X_train.shape, y_train.shape)
print('Testing data shape : ', X_test.shape, y_test.shape)
 
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

y_train = keras.utils.to_categorical(y_train, nClasses)
y_test = keras.utils.to_categorical(y_test, nClasses)

# Guardo X_test y_test para su posterior análisis en el archivo de predicción
with open('X_test', 'wb') as Xtest_file:
  pickle.dump(X_test, Xtest_file)
with open('y_test', 'wb') as ytest_file:
  pickle.dump(y_test, ytest_file)

# Esta será la arquitectura de mi modelo a entrenar
input_shape = (img_rows, img_cols, img_rgb)
tShirts_model = Sequential()
tShirts_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
tShirts_model.add(Conv2D(64, (3, 3), activation='relu'))
tShirts_model.add(MaxPooling2D(pool_size=(2, 2)))
tShirts_model.add(Dropout(0.25))
tShirts_model.add(Flatten())
tShirts_model.add(Dense(128, activation='relu'))
tShirts_model.add(Dropout(0.5))
tShirts_model.add(Dense(nClasses, activation='softmax'))

tShirts_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Entreno la Red Neuronal
batch_size = 20
epochs = 10

tShirts_model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = tShirts_model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Guardo los 2 Modelos que mejor val_accuracy obtuvieron

#tShirts_model.save("tShirts_LaLiga.h5py")  --> test accuracy de 0.75
#tShirts_model.save("tShirts073_LaLiga.h5py")  --> test accuracy de 0.73


