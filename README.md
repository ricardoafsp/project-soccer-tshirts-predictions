# project-soccer-tshirts-predictions
El proyecto lo basé en lograr el mejor porcentaje posible al entrenar una red neuronal que lograra reconocer camisetas de equipos de fútbol de La Liga (7 equipos actualmente), en un futuro espero poder seguir trabajando en ello y ampliar los equipos y posiblemente hasta ligas de distintos países.

## Estructura del trabajo realizado:
* Dataset creado de imágenes obtenidas mediante Webscraping
* Modelo "secuencial" de entrenamiento utilizado mediante Keras y Tensorflow
* Demo de presentación por una Api, mediante el uso de Flask.

## Modelos utilizados y su respectivo resultado:
* Train test split 80 - 20.
* Optimizador Adagrad, batch 64, epoch 6. Resultado -> Accuracy: 0,52.
* Optimizador Adadelta, batch 64, epoch 6. Resultado -> Accuracy: 0,60.
* Optimizador Adadelta, batch 32, epoch 10. Resultado -> Accuracy: 0,73.
* Optimizador Adadelta, batch 20, epoch 10. Resultado -> Accuracy: 0,75.

## Conclusión:
Luego de obtener el mejor resultado posible tras varios entrenamientos al modelo, lo aplicamos a nuestro dataset obteniendo un porcentaje de 90% de acierto.
