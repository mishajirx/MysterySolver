from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import MysteryNet
import os

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = MysteryNet()
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

data_X = None
data_Y = None
training_active = False

def load_data():
    global data_X, data_Y
    csv_path = os.path.join(os.path.dirname(__file__), "..", "visualizer", "public", "data", "train.csv")
    print(f"Loading data from {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        # Fallback for dev environment if path differs
        csv_path = "c:\\JHU\\Intro2NN\\MysterySolver\\visualizer\\public\\data\\train.csv"
        
    try:
        # Load data (no header)
        df = pd.read_csv(csv_path, header='infer')
        
        # ID (0), Features (1-205), Label (206)
        X = df.iloc[:, 1:206].values.astype(np.float32)
        y = df.iloc[:, 206].values.astype(np.longlong)
        
        data_X = torch.tensor(X)
        data_Y = torch.tensor(y)
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data: {e}")

@app.on_event("startup")
async def startup_event():
    pass # Load data on demand or startup? Let's load on startup to be ready.
    load_data()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Mystery Solver Backend"}

@app.get("/model/architecture")
def get_architecture():
    # Return structure for visualization: Nodes per layer
    # Input: 205
    # Layer 1: 512
    # Layer 2: 512
    # Output: 5
    return {
        "layers": [205, 512, 512, 5]
    }

@app.get("/model/weights")
def get_weights():
    # Return a subsample or summary of weights for visualization to avoid massive payloads
    # User asked for simplified visualization. 
    # Sending 512x205 matrix is ~100k params. 
    # Let's send a reduced representation or full if local network allows.
    # 30MB CSV loaded fine, so 1MB JSON weights is fine.
    
    # We will send a "snapshot" of weights.
    # To reduce size for visualizer, we might just send the first N neurons or average?
    # But user wants "circles with weights inside".
    
    # Let's return the full state dict but convert to list
    # Warning: This is heavy.
    # Optimization: Only send mean/std per neuron? 
    # Or send a small subset, e.g., 10x10 neurons?
    
    # Let's assume for purely visual purposes we can downsample 
    # if the frontend only draws a few nodes (e.g. 20 nodes per layer).
    
    # Return raw text? No, JSON.
    # We'll just return the first 20 neurons of each layer for the visualizer
    # to keep it snappy, unless asked otherwise.
    
    vis_limit = 32 # Only visualize 32 nodes per layer to make it readable
    
    with torch.no_grad():
        w1 = model.layer1.weight[:vis_limit, :vis_limit].tolist() # [Out, In]
        b1 = model.layer1.bias[:vis_limit].tolist()
        
        w2 = model.layer2.weight[:vis_limit, :vis_limit].tolist()
        b2 = model.layer2.bias[:vis_limit].tolist()
        
        wo = model.output_layer.weight[:, :vis_limit].tolist() # All 5 outputs, input subset
        bo = model.output_layer.bias.tolist()

    return {
        "layer1": {"weights": w1, "bias": b1},
        "layer2": {"weights": w2, "bias": b2},
        "output": {"weights": wo, "bias": bo}
    }

@app.post("/train/step")
def train_step(epochs: int = 1, lr: float = 0.001):
    global optimizer
    
    if data_X is None:
        return {"error": "Data not loaded"}
        
    # Update Learning Rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    model.train()
    
    # Simple full batch or mini-batch? 
    # Let's do a few batches per step call for responsiveness
    batch_size = 64
    indices = torch.randperm(data_X.size(0))
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Run a few iterations, not necessarily full epochs if dataset is huge (30MB is fine for full epoch though)
    # 30MB text is roughly 50k rows?
    # Let's run 1 epoch per call
    
    for _ in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        batches = 0
        
        for i in range(0, len(data_X), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = data_X[batch_idx]
            batch_y = data_Y[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            batches += 1
            
        total_loss = epoch_loss / batches
        
    return {
        "loss": total_loss,
        "accuracy": correct / total
    }

@app.post("/reset")
def reset_model():
    global model, optimizer
    model = MysteryNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    return {"message": "Model reset"}
