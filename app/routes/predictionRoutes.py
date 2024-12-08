from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi import Depends
from ..controllers.userSignupControllers import get_current_user 
from ..config.usersdatabase import signupcollectioninfo 

# Create a router
router = APIRouter()

# Load your trained model (ensure the model file is in the correct path)
model = joblib.load('best_model.joblib')

# Pydantic model for request validation
class CarInput(BaseModel):
    year: int
    make: str
    model: str
    miles: int
    trim: str



@router.post("/pricepredict", tags=["Price prediction"])
async def predict_price(input_data: CarInput, current_user: str = Depends(get_current_user)):
    # Check if the user is authenticated
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Get user from the database
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Convert input data to a DataFrame for prediction
    input_dict = input_data.dict()
    df = pd.DataFrame([input_dict])

    # Predict the price
    try:
        prediction = model.predict(df)
        predicted_price = prediction[0]

        # Prepare the price for saving in the user's statistics
        price_entry = {
            "calculated_price": predicted_price,
            "timestamp": pd.Timestamp.now().isoformat()
        }

        # Save the calculated price in the user's statistics
        if "statistics" not in user:
            user["statistics"] = {}
        
        # Ensure there is an array to store calculated prices
        if "calculatedPrices" not in user["statistics"]:
            user["statistics"]["calculatedPrices"] = []
        
        user["statistics"]["calculatedPrices"].append(price_entry)

        # Update the user document in the database
        await signupcollectioninfo.update_one(
            {"email": current_user},
            {"$set": {"statistics": user["statistics"]}}
        )

        # Return the predicted price
        return {"predicted_price": f"${predicted_price:,.2f}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




# # Price prediction endpoint
# @router.post("/pricepredict", tags=["Price prediction"])
# def predict_price(input_data: CarInput):
#     # Convert input data to a DataFrame
#     input_dict = input_data.dict()
#     df = pd.DataFrame([input_dict])
#     # Predict the price
#     try:
#         prediction = model.predict(df)
#         return {"predicted_price": f"${prediction[0]:,.2f}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

