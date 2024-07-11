from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import tensorflow as tf
from PIL import Image
import requests
import io
import numpy as np
import cv2
import base64

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.preprocessing import image as tf_image

app = FastAPI()

# Load pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=True)
class_labels = ["Fe", "Fi", "Le", "Ln", "Se", "Si", "Te", "Ti"]

class ImageData(BaseModel):
    id: str
    image_url: str

def download_image(image_url: str) -> Image:
    response = requests.get(image_url)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        return image
    else:
        raise HTTPException(status_code=400, detail="Unable to download image from URL")

def preprocess_image(image: Image) -> np.ndarray:
    # Convert PIL Image to numpy array
    img = tf_image.img_to_array(image)
    img = np.expand_dims(img, axis=0)
    img = vgg_preprocess_input(img)
    return img

@app.post("/predict")
async def predict(image_data: ImageData):
    try:
        # Download image from URL
        image = download_image(image_data.image_url)

        # Preprocess image for VGG16
        processed_image = preprocess_image(image)

        # Make prediction using VGG16 model
        prediction = vgg_model.predict(processed_image)
        predicted_label = class_labels[np.argmax(prediction)]

        # Convert image to base64 for response
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "id": image_data.id,
            "image": image_base64,
            "prediction": predicted_label
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
