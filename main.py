from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Create FastAPI app
app = FastAPI()

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Input schema
class MachineData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

# Home API
@app.get("/")
def home():
    return {"message": "Machine Failure Prediction API"}

# Prediction API
@app.post("/predict")
def predict(data: MachineData):

    df = pd.DataFrame(
        [[
            data.air_temperature,
            data.process_temperature,
            data.rotational_speed,
            data.torque,
            data.tool_wear
        ]],
        columns=[
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    )

    prediction = model.predict(df)

    if prediction[0] == 1:
        result = "Machine Failure"
    else:
        result = "No Machine Failure"

    return {
        "prediction": int(prediction[0]),
        "result": result
    }
