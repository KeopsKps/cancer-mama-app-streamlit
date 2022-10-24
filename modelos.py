import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SHAPE = 256

class_names = ['benigno', 'maligno', 'normal']

def modelo_segmentacion(imagen):
    imagen = cargar_imagen(imagen)
    imagen = np.squeeze(imagen)
    print('Segmentacion shape:',imagen.shape)
    modelo = tf.keras.models.load_model('BreastCancerSegmentor.h5')
    #prediccion = modelo.predict(imagen)
    prediccion = modelo.predict(tf.expand_dims(imagen, 0))
    print('prediccion:', prediccion.shape)
    return prediccion[0]#[:, :, 0]

def modelo_clasificacion(ruta_imagen):
    #imagen_prueba = []
    #imagen = ruta_imagen.copy()
    imagen = cargar_imagen(ruta_imagen)
    #print('shape clasificación')
    #print(imagen.shape)
    #imagen_prueba.append(imagen)
    #print(imagen_prueba.shape)
    modelo = tf.keras.models.load_model('ModeloClasificacion.h5')
    #archivo_imagen = tf.keras.utils.get_file(ruta_imagen, origin=ruta_imagen)
    #img_height = 256
    #img_width = 256
    #img = tf.keras.utils.load_img(
    #    archivo_imagen, target_size=(img_height, img_width)
    #)
    #img_array = cargar_dicom_file(ruta_imagen)#tf.keras.utils.img_to_array(img)
    #img_array = tf.expand_dims(img_array, 0) # Create a batch
    batch_image = tf.expand_dims(imagen, 0)

    predictions = modelo.predict(batch_image)
    score = tf.nn.softmax(predictions)
    return (class_names[np.argmax(score)], 100 * np.max(score))

def cargar_imagen(imagen):
    img = cv2.resize(imagen, (IMG_SHAPE, IMG_SHAPE))
    #img = imagen.resize((IMG_SHAPE, IMG_SHAPE))
    #img_array = np.array(img)
    return img#cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

def cargar_imagen_2(imagen):
    img = plt.imread(imagen)
    img = cv2.resize(imagen, (IMG_SHAPE, IMG_SHAPE))
    return img