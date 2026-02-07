from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import logging
from typing import List, Dict, Optional
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Laptop Price Predictor API")

# Load model and data once at startup
pipe = None
df = None

try:
    # Check if files exist
    if not os.path.exists('pipe.pkl'):
        logger.error("pipe.pkl not found!")
    if not os.path.exists('df.pkl'):
        logger.error("df.pkl not found!")
    
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('df.pkl', 'rb'))
    logger.info(" Model loaded successfully")
except Exception as e:
    logger.error(f" Failed to load model: {e}")
    sys.exit(1)

# Define request/response models
class PredictionRequest(BaseModel):
    company: str
    type_laptop: str
    ram: int
    weight: float
    touchscreen: bool
    ips: bool
    screen_size: float
    resolution: str
    cpu: str
    hdd: int
    ssd: int
    gpu: str
    os_type: str

class PredictionResponse(BaseModel):
    predicted_price: float
    status: str = "success"

class OptionsResponse(BaseModel):
    companies: List[str]
    types: List[str]
    cpus: List[str]
    gpus: List[str]
    os: List[str]
    ram_options: List[int]
    storage_options: List[int]
    resolutions: List[str]

# API Endpoints
@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": "loaded"}

@app.get("/options", response_model=OptionsResponse)
def get_options():
    """Get all available options for dropdowns"""
    return OptionsResponse(
        companies=['ASUS', 'Dell', 'HP', 'Lenovo', 'Apple', 'MSI', 'Acer', 'Vaio', 'Razer', 'Medion'],
        types=['Ultrabook', 'Notebook', '2 in 1 Convertible', 'Gaming', 'Netbook'],
        cpus=['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'Other Intel Processor', 'AMD Processor'],
        gpus=['None', 'Nvidia', 'Intel', 'AMD'],
        os=['Windows', 'mac', 'Others / linux /no OS'],
        ram_options=[2, 4, 6, 8, 12, 16, 24, 32, 64],
        storage_options=[0, 8, 128, 256, 512, 1024, 2048],
        resolutions=['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
    )

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict laptop price"""
    try:
        touchscreen_val = 1 if request.touchscreen else 0
        ips_val = 1 if request.ips else 0
        
        # Calculate PPI
        X_res = int(request.resolution.split('x')[0])
        Y_res = int(request.resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / request.screen_size
        
        # Prepare query
        query = np.array([
            request.company, request.type_laptop, request.ram, request.weight,
            touchscreen_val, ips_val, ppi, request.cpu, request.hdd,
            request.ssd, request.gpu, request.os_type
        ], dtype=object)
        
        query = query.reshape(1, 12)
        
        # Make prediction
        log_price = pipe.predict(query)[0]
        predicted_price = np.exp(log_price)
        
        logger.info(f"Prediction: {predicted_price}")
        return PredictionResponse(predicted_price=round(predicted_price, 2))
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
