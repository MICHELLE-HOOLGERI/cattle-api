import numpy as np
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)
