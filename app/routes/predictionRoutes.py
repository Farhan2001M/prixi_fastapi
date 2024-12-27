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




class TaxData(BaseModel):
    taxAmount: float
    state: str
    timestamp: str

@router.post("/savetax", tags=["Tax Calculation"])
async def save_calculated_tax(tax_data: TaxData, current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized") 
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Prepare the tax entry to save in the user's statistics
    tax_entry = {
        "calculated_tax": tax_data.taxAmount,
        "state": tax_data.state,
        "timestamp": tax_data.timestamp }
    # Ensure there is a 'calculatedTaxes' array in statistics
    if "statistics" not in user:
        user["statistics"] = {}
    if "calculatedTaxes" not in user["statistics"]:
        user["statistics"]["calculatedTaxes"] = []
    user["statistics"]["calculatedTaxes"].append(tax_entry)
    # Update the user's document
    await signupcollectioninfo.update_one(
        {"email": current_user},
        {"$set": {"statistics": user["statistics"]}}
    )
    return {"message": "Tax data saved successfully"}




































# from pydantic import BaseModel
# import random
# from fastapi import APIRouter , HTTPException


# # import joblib
# from fastapi import Depends
# from ..controllers.userSignupControllers import get_current_user 
# from ..config.usersdatabase import signupcollectioninfo 
# import pandas as pd



# # Create a router
# router = APIRouter()

# # # Load your trained model (ensure the model file is in the correct path)
# # model = joblib.load('best_model.joblib')

# # Pydantic model for request validation
# class CarInput(BaseModel):
#     year: int
#     make: str
#     model: str
#     miles: int
#     trim: str



# # @router.get('/')
# # async def home():
# #     # send_simple_message()
# #     return {'msg': 'Welcome in my Prediction route'}


# @router.post("/pricepredict", tags=["Price prediction"])
# async def predict_price(input_data: CarInput, current_user: str = Depends(get_current_user)):
#     # Check if the user is authenticated
#     if not current_user:
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     # Get user from the database
#     user = await signupcollectioninfo.find_one({"email": current_user})
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")
    
#     # Convert input data to a DataFrame for prediction
#     input_dict = input_data.dict()
#     df = pd.DataFrame([input_dict])

#     # Predict the price
#     try:
#         prediction = random.randint(1000, 50000)
#         predicted_price = prediction[0]

#         # Prepare the price for saving in the user's statistics
#         price_entry = {
#             "calculated_price": predicted_price,
#             "timestamp": pd.Timestamp.now().isoformat()
#         }

#         # Save the calculated price in the user's statistics
#         if "statistics" not in user:
#             user["statistics"] = {}
        
#         # Ensure there is an array to store calculated prices
#         if "calculatedPrices" not in user["statistics"]:
#             user["statistics"]["calculatedPrices"] = []
        
#         user["statistics"]["calculatedPrices"].append(price_entry)

#         # Update the user document in the database
#         await signupcollectioninfo.update_one(
#             {"email": current_user},
#             {"$set": {"statistics": user["statistics"]}}
#         )

#         # Return the predicted price
#         return {"predicted_price": f"${predicted_price:,.2f}"}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")




