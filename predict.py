from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
from io import BytesIO

model = load_model('pneumonia_resnet50.h5')

def read_image(file) -> Image.Image:
    pil_image = Image.open(BytesIO(file))
    return pil_image

def predictor(file: Image.Image):

    img = file.resize((224, 224))
    x = image.img_to_array(img)
    if x.shape[-1] == 1:
        x = np.concatenate([x] * 3, axis=-1)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    predictions = model.predict(x)
    if predictions[0] > 0.5:
        predicted_class= 'X-ray image is classified Pneumonia'
    else:
        predicted_class= 'X-ray image is classified Normal'

    response = {
        "class": predicted_class,
        "score": f"{float(predictions[0]):.4f}"
    }
    return response