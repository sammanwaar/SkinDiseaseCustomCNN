from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import io
import json
import os
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = "skin_disease_custom_cnn.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = 224

app = FastAPI(title="Skin Disease Classifier")

# Load model once at startup
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Load labels mapping if available (expects class_name -> index)
idx_to_label = None
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r") as f:
            class_indices = json.load(f)
        # invert mapping to index -> class_name (keys may be strings)
        idx_to_label = {int(v): k for k, v in class_indices.items()}
        print("Labels loaded successfully")
    except Exception as e:
        print(f"Error loading labels: {e}")

def read_imagefile(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/", response_class=HTMLResponse)
def ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Skin Disease Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            form { margin-bottom: 20px; }
            input[type="file"] { margin-bottom: 10px; }
            button { padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            #result { margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>Skin Disease Classifier</h1>
        <form id="uploadForm">
            <input type="file" id="fileInput" accept="image/*" required>
            <br>
            <button type="submit">Predict</button>
        </form>
        <div id="result"></div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async function(event) {
                event.preventDefault();
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (response.ok) {
                        let resultHtml = '<h2>Predictions:</h2><ul>';
                        data.predictions.forEach(pred => {
                            resultHtml += `<li>${pred.label}: ${pred.score.toFixed(4)}</li>`;
                        });
                        resultHtml += '</ul>';
                        document.getElementById('result').innerHTML = resultHtml;
                    } else {
                        document.getElementById('result').innerHTML = `<p>Error: ${data.detail}</p>`;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = '<p>Error occurred</p>';
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    contents = await file.read()
    try:
        img = read_imagefile(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    x = preprocess_image(img)
    preds = model.predict(x)[0]
    top_k = min(top_k, len(preds))
    top_indices = preds.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        label = idx_to_label.get(int(idx), str(int(idx))) if idx_to_label else str(int(idx))
        results.append({"label": label, "index": int(idx), "score": float(preds[int(idx)])})
    return JSONResponse({"predictions": results})
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)