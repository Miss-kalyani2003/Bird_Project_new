import json
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def load_trained_model(model_path):
    return load_model(model_path)

def load_class_names(json_path):
    with open(json_path, "r") as f:
        class_names = json.load(f)
    return {int(k): v for k, v in class_names.items()}

def predict_image(model, class_names, img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index] * 100

    return class_names[class_index], confidence
