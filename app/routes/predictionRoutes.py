from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Create a router
router = APIRouter()

# Load your trained model (ensure the model file is in the correct path)
model = joblib.load('model.joblib')

# Pydantic model for request validation
class CarInput(BaseModel):
    year: int
    make: str
    model: str
    miles: int
    trim: str


# Price prediction endpoint
@router.post("/pricepredict", tags=["Price prediction"])
def predict_price(input_data: CarInput):
    # Convert input data to a DataFrame
    input_dict = input_data.dict()
    df = pd.DataFrame([input_dict])
    
    # Predict the price
    try:
        prediction = model.predict(df)
        return {"predicted_price": f"${prediction[0]:,.2f}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To use this router in your main FastAPI app, include it like this:
# app.include_router(router)

