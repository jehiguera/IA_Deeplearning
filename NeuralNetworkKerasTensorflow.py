#Deeplearning example. Database imagenes MMIST que tiene 70000 imagenes con una resolucion de 28x28
#utilizarremos 60000 imagens con una resolucion de 28x28
#60000 imagenens para entrenamiento y otras 10000 para prediccion
from matplotlib.pyplot import plot, show
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt


ropa = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = ropa.load_data()
print(train_images.shape)

#vamos a visualizar la primera imagen
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
print(train_labels[0])

#crear clase para tener el texto de la clasificacion
class_ropa =['Top', 'Pantalones', 'Jersey', 'Vestido', 'Abrigo', 'Sandalias', 'Camisa', 'bambas', 'Bolso', 'Botas']
print(class_ropa[train_labels[0]])

#vamos a crear una funcion que nos dibuje la imagen y nos diga el tipo de ropa que es

def identificar_ropa(x):
    plt.figure()
    plt.imshow(train_images[x])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print("la imagen muestra: " + class_ropa[train_labels[x]])

identificar_ropa(41373)

#vamos a escalar los valores antes de alimentarlos en el modelo de la red neuronal

train_images = train_images/255.0
test_images = test_images/255.0

#vamos a ver las imagenes
plt.figure(figsize=(10,10))

#vamos a hacer un bucle for para ver las primeras 25 imagenes

for i in range(25):
    plt.subplot(7,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_ropa[train_labels[i]])

plt.show()

#vamos a crear la red neuronal con una entrada de imagen de 28 x 28 pixeles en la capa de entrada y 128 neuronas con activacion relu y 10 neuronas con activacion softmax en la capa oculta

modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

#construir el modelo, requiere configurar las capas y despues compilar el modelo

modelo.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


#vamos a entrenar el modelo con fit

modelo.fit(train_images, train_labels, epochs=20)

#modelo.fit(train_images, train_labels, epochs=20)

#funcion para identificar las imagenes
def identificar_ropa2(x):
    plt.figure()
    plt.imshow(test_images[x])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print("la imagen muestra: " + class_ropa[test_labels[x]])

identificar_ropa2(450)

#vamos ha hacer la prediccion
prediccion = modelo.predict(test_images)
print(prediccion[450])

#ver el elemento del test
i=2050
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot.images(i, prediccion[i], test_labels, test_images)

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

#https://github.com/tensorflow/docs-l10n/blob/master/site/es-419/tutorials/keras/classification.ipynb


num_rows = 6
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_images(i, prediction[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


