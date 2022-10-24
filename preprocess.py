import tempfile
from io import BytesIO

import pydicom

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


def cargar_dicom_file(image_bytes):
    lossy_image = tfio.image.decode_dicom_image(image_bytes.getvalue(), scale='auto', on_error='lossy', dtype=tf.uint8)
    numpy_lossy_image = np.squeeze(lossy_image.numpy())
    return numpy_lossy_image
    #ds = pydicom.dcmread(BytesIO(image_bytes.getvalue()))
    #lossy_image = tfio.image.decode_dicom_image(ds, scale='auto', on_error='lossy', dtype=tf.uint8)
    #numpy_lossy_image = np.squeeze(lossy_image.numpy())
    #return numpy_lossy_image
    
    # with tempfile.NamedTemporaryFile(mode='r+b', suffix='.dcm') as temp:
    #     bytes_data = image_bytes.getvalue()
    #     temp.write(bytes_data)
    #     ds = pydicom.dcmread(temp.name)

    #     return len(ds.SequenceOfUltrasoundRegions)