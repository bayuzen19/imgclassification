from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import tensorflow as tf
from PIL import Image
import requests
import io
import numpy as np
import base64

app = FastAPI()

# Load the custom small CNN model
model = tf.keras.models.load_model('small_cnn_model.h5')
class_labels = ['Fe', 'Fi', 'Ie', 'Ii', 'In', 'Se', 'Si', 'Te', 'Ti']

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
    print(f"Initial image size: {image.size}")
    img = np.array(image.resize((224, 224)))
    print(f"Resized image shape: {img.shape}")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = img.astype('float32') / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

@app.post("/predict")
async def predict(image_data: ImageData):
    try:
        # Download image from URL
        image = download_image(image_data.image_url)

        # Preprocess image for the small CNN model
        processed_image = preprocess_image(image)

        # Make prediction using the small CNN model
        prediction = model.predict(processed_image)
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
