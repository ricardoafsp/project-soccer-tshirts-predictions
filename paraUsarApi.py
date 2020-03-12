import numpy as np
import cv2
import tensorflow as tf

def reconoceLaCamiseta(path):
    def ryrImage(ruta):
        img2 = cv2.imread(ruta)
        img2 = cv2.resize(img2,(256,256),3)
        img2 = np.expand_dims(img2, axis=0)
        return img2

    resultado = ryrImage(path)

    modelo_a_trabajar = tf.keras.models.load_model(
        filepath="tShirts_LaLiga.h5py"
    )

    def predict(otraPrueba):
        predictions_single = modelo_a_trabajar.predict(otraPrueba)
        return predictions_single

    resultadoFinal = predict(resultado)

    for x in resultadoFinal:
        if x[0] == max(x):
            print('La camiseta es del -> Barcelona')
        elif x[1] == max(x):
            print('La camiseta es del -> Real Madrid')
        elif x[2] == max(x):
            print('La camiseta es del -> Valencia')
        elif x[3] == max(x):
            print('La camiseta es del -> Getafe')
        elif x[4] == max(x):
            print('La camiseta es del -> Betis')
        elif x[5] == max(x):
            print('La camiseta es del -> Atletico de Madrid')
        elif x[6] == max(x):
            print('La camiseta es del -> Athletic Bilbao')
    return None

#reconoceLaCamiseta('Prueba/20c6d52136.jpg')