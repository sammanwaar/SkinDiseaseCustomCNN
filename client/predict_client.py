import requests

API_URL = "http://127.0.0.1:8001/predict"
IMAGE_PATH = "test.jpg"

with open(IMAGE_PATH, "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    resp = requests.post(API_URL, files=files, params={"top_k": 3})
print(resp.status_code, resp.json())