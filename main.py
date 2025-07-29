import os
from fastapi import FastAPI, UploadFile, File
from services.data_processor import process_data
from services.model_trainer import train_model
from models.predictor import predict
import os
app = FastAPI()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"filename": file.filename, "file_path": file_path}

@app.get("/stats/{filename}")
async def get_stats(filename: str):
    file_path = os.path.join("uploads", filename)
    stats = process_data(file_path)
    return {"stats": stats}

@app.get("/train/{filename}")
async def train(filename: str):
    file_path = os.path.join("uploads", filename)
    accuracy = train_model(file_path)
    return {"accuracy": accuracy}

@app.post("/predict")
async def get_prediction(data: list):
    prediction = predict(data)
    return {"prediction": prediction}

