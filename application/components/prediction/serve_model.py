from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image
import PIL
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import decode_predictions

model = None


def load_model():
    model = tf.keras.models.load_model('saved_model\my_model')
    print("Model loaded")
    return model


def predict(image: Image.Image):
   
    global model
    if model is None:
        model = load_model()
    class_names= ['Non_recyclable', 'Recyclable']
    img_height = 180
    img_width = 180
    #model = tf.keras.models.load_model('saved_model\my_model')
    #data= Image
    #image = Image
    image0 = image.resize((img_height, img_width))

    img_array = tf.keras.utils.img_to_array(image0)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    classe_name = class_names[np.argmax(score)]
    score = round(100 * np.max(score),2)

    print(predictions)
    print(score)

    response= [classe_name,score]

    return response





def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
