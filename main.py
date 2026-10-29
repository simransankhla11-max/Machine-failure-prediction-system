from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Create FastAPI App
app = FastAPI(title="Machine Failure Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema
class MachineData(BaseModel):
    type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int

# Home Route
@app.get("/")
def home():
    return {"message": "Machine Failure Prediction API is Running"}

# Prediction Route
@app.post("/predict")
def predict(data: MachineData):

    machine_type = label_encoder.transform([data.type])[0]

    df = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [data.air_temperature],
        "Process temperature [K]": [data.process_temperature],
        "Rotational speed [rpm]": [data.rotational_speed],
        "Torque [Nm]": [data.torque],
        "Tool wear [min]": [data.tool_wear]
    })

    df = scaler.transform(df)

    prediction = model.predict(df)[0]

    if prediction == 1:
        result = "Machine Failure"
    else:
        result = "No Machine Failure"

    return {
        "prediction": int(prediction),
        "result": result
    }
