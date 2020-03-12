import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image
import os
import pickle

# Busco los archivos en un directorio
images=[]
equipos = ['Barcelona', 'Real Madrid', 'Valencia', 'Getafe', 'Betis', 'Atletico de Madrid', 'Athletic Bilbao']
for x in equipos:
    basepath = f'Jugadores/{x}'
    for entry in os.listdir(basepath):
        if os.path.isfile(os.path.join(basepath, entry)):
            images.append(basepath+'/'+entry)

# Creo un diccionario para tener el array de las camisetas y los nombres de los equipos
tShirts = {'Camisetas':[],'Equipos':[]}

# Creo una función para rotar las imágenes en los grados deseados y así poder tener más datos
def rotateImg(x,img,num):
    tShirts['Camisetas'].append(ndimage.rotate(img, num, reshape=False))
    tShirts['Equipos'].append(x.split('/')[1])
    return tShirts

# Función que lee las imagenes y las reduce a 256
def readAndResize(images):
    for x in images:
        if '.DS_Store' not in x:
            img = cv2.imread(x)
            img = cv2.resize(img,(256,256),3)
            tShirts['Camisetas'].append(img)
            tShirts['Equipos'].append(x.split('/')[1])
            rotateImg(x,img,45)
            rotateImg(x,img,90)
            rotateImg(x,img,180)
            rotateImg(x,img,270)
    return tShirts

# Aplico la función y guardo los elementos en el diccionario
readAndResize(images)

# Creo un DataFrame con el diccionario obtenido
tShirtsDF = pd.DataFrame(tShirts)

# Creo etiquetas para categorizar los equipos para el modelo a entrenar
labels = {
    'Barcelona' : 0,
    'Real Madrid' : 1,
    'Valencia' : 2,
    'Getafe' : 3,
    'Betis' : 4,
    'Atletico de Madrid' : 5,
    'Athletic Bilbao' : 6
}

# Agrego la colomna labels al DataFrame
tShirtsDF['Labels'] = tShirtsDF['Equipos'].map(labels)
tShirtsDF

# Guardo el DataFrame para importarlo en el .py donde entrenaré el modelo
tShirtsDF.to_pickle("tShirtsDF.pkl")


