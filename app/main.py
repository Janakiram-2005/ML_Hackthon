from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import sys

# Add src to python path so we can import model
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from model import SimpleFeedForwardNN

app = FastAPI(title="ML Hackathon Prediction API")

# Define request schema
class PredictionRequest(BaseModel):
    features: list[float]

# Initialize model
model = SimpleFeedForwardNN(input_dim=10)
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'best_model.pth')

@app.on_event("startup")
def load_model():
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model checkpoint not found at {model_path}. Using uninitialized model.")

@app.post("/predict")
def predict(request: PredictionRequest):
    if len(request.features) != 10:
        raise HTTPException(status_code=400, detail="Expected exactly 10 features.")
    
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        x = torch.tensor([request.features], dtype=torch.float32)
        prediction = model(x)
        
    return {"prediction": prediction.item(), "class": 1 if prediction.item() >= 0.5 else 0}
