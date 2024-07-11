# FastAPI TensorFlow Image Prediction

This project implements a FastAPI service that loads a pre-trained TensorFlow model, receives input in the form of an ID and base64 encoded image, decodes the image, preprocesses it, and returns the prediction result along with the original image and ID.

## Features

- Load a pre-trained TensorFlow model
- Decode base64 encoded images
- Preprocess images similar to the training process
- Predict the class of the image
- Return the ID, original image in base64, and prediction result

## Requirements

- Python 3.7+
- TensorFlow
- FastAPI
- Uvicorn
- Pillow
- OpenCV
- NumPy

## Installation

Create and activate a virtual environment:

1. ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure you have your pre-trained TensorFlow model saved as `complex_cnn_vgg_like_model.h5`.
2. Start the FastAPI server using Uvicorn:

   ```bash
   uvicorn main:app --reload
   ```
3. The server will be running at `http://127.0.0.1:8000`.

## Usage

### Endpoint: `/predict`

**Method:** `POST`

**Description:** Accepts an image in base64 format and returns the ID, original image in base64, and prediction result.

**Request Body:**

```json
{
    "id": "unique_identifier",
    "image_base64": "base64_encoded_image_string"
}
```
