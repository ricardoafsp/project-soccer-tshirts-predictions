import pandas as pd
import tensorflow as tf
import pickle

# Importo el modelo que mejor me dió
modelo_a_trabajar = tf.keras.models.load_model(
    filepath="tShirts_LaLiga.h5py"
)

# Importo las variables test para comprobar el resultado predictivo
with open('X_test', 'rb') as Xtest_file:
    X_test = pickle.load(Xtest_file)
with open('y_test', 'rb') as ytest_file:
    y_test = pickle.load(ytest_file)

# Defino la variable de la predicción del array de las camisetas
prediccionCamisetas = modelo_a_trabajar.predict(X_test)

# Creo un DataFrame de las etiquetas de los equipos
equiposTest = pd.DataFrame(y_test)
equiposTest.columns = ['BCN', 'RMD', 'VAL', 'GET', 'BET', 'ATM', 'ATB']
equiposTest

# Obtengo el mayor valor por fila del DataFrame de las etiquetas (en este caso los = 1)
equiposTestMax = equiposTest.idxmax(axis=1)
equiposTestMax

# Creo un DataFrame de los array de las camisetas de los equipos
prediccionesResultado = pd.DataFrame(prediccionCamisetas)
prediccionesResultado.columns = ['BCN', 'RMD', 'VAL', 'GET', 'BET', 'ATM', 'ATB']
prediccionesResultado

# Obtengo el mayor valor por fila del DataFrame de las camisetas (el mayor porcentaje predicho)
prediccionesResultadoMax = prediccionesResultado.idxmax(axis=1)
prediccionesResultadoMax

# Uno ambos datasets, creo una nueva columna 'Predicción', y si las filas son iguales es que acertó 'True'
resultadoFinal = [equiposTestMax, prediccionesResultadoMax]
resultadoFinal = pd.concat(resultadoFinal, axis=1, join='inner')
resultadoFinal.columns = ['EQUIPO', 'PREDICCION']
resultadoFinal['ACIERTO'] = (resultadoFinal['EQUIPO'] == resultadoFinal['PREDICCION'])
resultadoFinal

# Se puede observar que de 350 imágenes acertó en 318
resultadoFinal.ACIERTO.value_counts()

# Esto nos da un % de acierto de 90,85
porcentajeAcierto = resultadoFinal.ACIERTO.value_counts(True)
print(porcentajeAcierto)

# En la primera predicción me dió un 90,85% de acierto (trabajado en el jupyter)
# Al crear nuevos archivos en Visual, volví a hacer un train test split y ahora me da 85,71% de acierto.
