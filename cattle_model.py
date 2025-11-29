from tensorflow.keras.models import load_model
from image_utils import preprocess_image
import numpy as np

IMG_SIZE = 224
CATTLE_CLASSES = ["foot-and-mouth", "healthy", "lumpy"]

class CattleModel:
    def __init__(self):
        self.model = load_model("cattle_model.h5")

    def predict(self, img_path):
        img_array = preprocess_image(img_path, IMG_SIZE)

        preds = self.model.predict(img_array)
        pred = CATTLE_CLASSES[np.argmax(preds)]
        conf = float(np.max(preds))
        return pred, conf
