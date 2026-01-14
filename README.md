# Skin Disease Custom CNN

This project implements a custom CNN for skin disease classification using TensorFlow and FastAPI.

## Features

- Custom CNN model for classifying 10 types of skin diseases
- FastAPI web API for predictions
- Web UI for uploading images and viewing results
- Python client script for testing

## Classes

The model classifies the following skin diseases:
1. Acne
2. Actinic Keratosis
3. Basal Cell Carcinoma
4. Dermatofibroma
5. Melanoma
6. Nevus
7. Pigmented Benign Keratosis
8. Seborrheic Keratosis
9. Squamous Cell Carcinoma
10. Vascular Lesion

## Requirements

- Python 3.12
- TensorFlow
- FastAPI
- Uvicorn
- Pillow
- NumPy
- Requests

## Installation

1. Install Python 3.12 from https://www.python.org/downloads/
2. Install dependencies: `pip install -r requirements.txt`
3. Install Microsoft Visual C++ Redistributable if not present

## Running the API and UI

Double-click `run_api.bat` or run:

```
C:\Users\Home\AppData\Local\Programs\Python\Python312\python.exe -m uvicorn api.app:app --host 0.0.0.0 --port 8001
```

Open http://127.0.0.1:8001 in your browser for the web UI.

## Testing the API

Run the client script:

```
C:\Users\Home\AppData\Local\Programs\Python\Python312\python.exe client/predict_client.py
```

This sends `test.jpg` to the API and prints the predictions.

## Training

To train the model, place the dataset in `./IMG_CLASSES` with subfolders for each class, then run:

```
C:\Users\Home\AppData\Local\Programs\Python\Python312\python.exe SkinDisease.py
```

## Files

- `SkinDisease.py`: Training script
- `api/app.py`: FastAPI application
- `client/predict_client.py`: Test client
- `run_api.bat`: Batch file to start the API
- `labels.json`: Class labels
- `skin_disease_custom_cnn.h5`: Trained model
- `test.jpg`: Sample test image