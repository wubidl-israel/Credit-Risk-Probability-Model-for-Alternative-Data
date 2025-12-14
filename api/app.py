from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
app= FastAPI()
scaler= StandardScaler()
MODEL_PATH= 'model.pkl'

try:
    model= joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}.")

class predictionrequest(BaseModel):
    features: list[float] #Input for prediction

class predictionresponse(BaseModel):
    prediction: float # Predicted values

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Prediction API"}

@app.post("/predict", response_model=predictionresponse)
def predict(request: predictionrequest):
    try:
        # Convert features to a NumPy array and reshape for the model
        features = np.array(request.features).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(features)
        
        # Return the prediction
        return {"prediction": float(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")
