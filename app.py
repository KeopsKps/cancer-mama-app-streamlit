import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from preprocess import cargar_dicom_file
from modelos import modelo_segmentacion, modelo_clasificacion

INFORMACION_PROYECTO = 'Mostrar información del proyecto'
SUBIR_ARCHIVO = 'Subir archivo DICOM'

OPCIONES_MENU_LATERAL= [
    INFORMACION_PROYECTO,
    SUBIR_ARCHIVO
]

realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

def mostrar_resultados_prediccion(image_array):
    resultado_segmentacion = modelo_segmentacion(image_array)
    clase, porcentaje = modelo_clasificacion(image_array)
    columna_izquierda, columna_derecha = st.columns(2)
    columna_izquierda.image(image_array, caption='Imagen seleccionada')
    columna_derecha.image(resultado_segmentacion, caption='Imagen Segmentada')
    st.write(f"La imagen de entrada tiene un {porcentaje:.2f}% de probabilidad de ser {clase}")

def main():
    st.write("""
    # Modelo de machine learning para el diagnóstico del cáncer de mama
    """
    )
    st.sidebar.title("Explorar opciones")
    app_mode = st.sidebar.selectbox("Selecciona una de las siguientes opciones", OPCIONES_MENU_LATERAL)
    if app_mode == INFORMACION_PROYECTO:
        st.sidebar.success("Se esta mostrando la información del proyecto a la derecha!")
        st.write(
            """
                Nuestro proyeto de innovación tecnologica titulado: Implementación de modelo de machine learning
                para el diagnostico del cáncer de mama, teniendo como principal finalidad contribuir a un
                diagnostico oportuno de este padecimiento en mujeres que presenten algún tipo de sintomatologia
                asociada a la enfermedad, en sus primeras etapas.

                Autores:

                - Sara Iris Bobadilla Moreno
                - Christhofer Gutierrez Acevedo

                Tutor:

                - M. Sc. Narciso Javier Aguilera Centeno

                Asesora Externa:

                - Dra. Gloria Carol Ramírez Reyes

            """
        )
    elif app_mode == SUBIR_ARCHIVO:
        uploaded_file = st.file_uploader("Subir archivo DICOM", type=[".dcm", ".dicom"])

        if uploaded_file:
            #st.empty()
            #realizar_prediccion = st.button("Realizar predicción!")
            #if realizar_prediccion:

            
            array_image = cargar_dicom_file(uploaded_file)
            img = Image.fromarray(array_image)
            rect = st_cropper(img,realtime_update=realtime_update, box_color=box_color,
                            aspect_ratio=aspect_ratio, return_type='box')
            #st.image(array_image)
            #print('Get Box')
            #print(cropped_img.crop())
            #print(cropped_img)
            #st.write("Imagen recortada")
            #_ = cropped_img.thumbnail((256,256))
            #st.image(cropped_img)
            realizar_prediccion = st.button("Realizar predicción!")
            if realizar_prediccion:
                st.empty()
                #crop = img.crop((rect['left'], rect['top'], rect['width'] + rect['left'], rect['height'] + rect['top']))
                #crop = array_image[rect['top']:rect['left'], rect['height'] + rect['top']:rect['width'] + rect['left']]
                crop = array_image[
                    rect['top']:rect['height'] + rect['top'], 
                    rect['left']:rect['width'] + rect['left']]
                #mostrar_resultados_prediccion(np.array(cropped_img.convert('RGB')))
                #crop_array = Image.fromarray(crop)
                print(crop.shape)
                matplotlib.image.imsave('./temp.png', crop)
                img = plt.imread('./temp.png')[:, :, :3, ]
                print(img.shape)
                mostrar_resultados_prediccion(img)
            

main()
