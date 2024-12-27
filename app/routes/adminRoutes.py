from fastapi import APIRouter, HTTPException
from ..models.adminmodel import LoginResponse , ForgotPasswordRequest , ValidateOTPRequest , PasswordChangeRequest , DeleteModelRequest , Brand , VehicleModel , BrandModel 
from ..controllers.adminControllers import get_current_user
from ..controllers.adminControllers import verify_admin
from ..config.admindatabase import adminlogininfo , VehicleData , Vehiclecollection
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, HTTPException, status
import random
import requests
router = APIRouter()
from fastapi import HTTPException, status
import bcrypt
import random 
import smtplib
from email.message import EmailMessage
from typing import Dict, Tuple
from datetime import datetime, timedelta
from fastapi import Depends 
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from pymongo import ReturnDocument
import shutil
import os
import base64
from fastapi import APIRouter
from typing import List, Dict, Any


router = APIRouter()


# @router.get('/')
# async def home():
#     # send_simple_message()
#     return {'msg': 'Welcome in my Admin Routes '} 




@router.post("/adminlogin", response_model=LoginResponse, tags=["Admin"])
async def login_admin(email: str, password: str):
    result = await verify_admin(email, password)    
    # Handle the different failure cases by raising HTTP exceptions
    if result == "email_not_registered":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not registered.")
    if result == "invalid_password":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password.")
    return result    


@router.post("/admin-forgot-password", tags=["Admin"])
async def forgot_password(request: ForgotPasswordRequest):
    email = request.email
    user = await adminlogininfo.find_one({"email": email})
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not registered.")
    # Generate a 6-digit OTP
    otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    print(f"Generated OTP: {otp}")

    # Email configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    from_email = 'prixihelpcentre@gmail.com'
    from_password = "jgtn fvsj ymuc wzje"   # Use environment variable for the password
    # Send OTP via email
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(from_email, from_password)
            msg = EmailMessage()
            msg['Subject'] = "OTP Verification"
            msg['From'] = from_email
            msg['To'] = email
            msg.set_content(f"Your OTP for password reset for the Ultimate Experience of Prixi system is: {otp}")
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to send OTP email.")
    # Store OTP and its expiry time in the user's document in MongoDB
    otp_expiry = datetime.utcnow() + timedelta(minutes=2)  # OTP valid for 2 minutes
    await adminlogininfo.update_one(
        {"email": email},
        {"$set": {"otp": {"code": otp, "expiry": otp_expiry}}}
    )
    return {"message": "OTP sent successfully."}


@router.post("/admin-validate-otp", tags=["Admin"])
async def validate_otp(request: ValidateOTPRequest):
    email = request.email
    entered_otp = request.otp

    # Fetch the user document
    user = await adminlogininfo.find_one({"email": email})
    
    if user is None or "otp" not in user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OTP not generated or expired.")
    
    # Extract OTP and expiry time from the user document
    stored_otp = user["otp"]["code"]
    otp_expiry = user["otp"]["expiry"]

    if datetime.utcnow() > otp_expiry:
        # OTP expired
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OTP expired.")
    
    if stored_otp != entered_otp:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OTP is not correct.")
    
    print("OTP MATCHES")
    
    # Optionally, clear the OTP after successful validation
    await adminlogininfo.update_one(
        {"email": email},
        {"$unset": {"otp": ""}}  # Remove the otp field from the document
    )
    return {"message": "OTP validated successfully."}


@router.post("/admin-change-password", tags=["Admin"])
async def change_password(request: PasswordChangeRequest):
    user = await adminlogininfo.find_one({"email": request.email})
    if not user:
        raise HTTPException(404, "User not found")
    # Hash the new password
    hashed_password = bcrypt.hashpw(request.new_password.encode('utf-8'), bcrypt.gensalt())
    # Update the user's password in the database
    try:
        await adminlogininfo.update_one(
            {"email": request.email}, 
            {"$set": {"password": hashed_password}}
        )
        return {"message": "Password successfully updated"}
    except Exception as e:
        raise HTTPException(500, "An error occurred while updating the password")


@router.get("/protected-route", tags=["Admin"])
async def protected_route(current_user: str = Depends(get_current_user)):
    # Your protected code here
    return {"message": "This is a protected route", "current_user": current_user}


@router.get("/get-brands&models")
async def get_brands_models():
    try:
        # Fetch all documents from VehicleData collection
        car_brands = await VehicleData.find({}, {"name": 1, "models": 1, "_id": 0}).to_list(length=None)        
        if not car_brands:
            raise HTTPException(status_code=404, detail="No car brands found")
        if car_brands is None:
            car_brands = []  # Return an empty array if no brands are found
        return car_brands
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching brands and models: {e}")
# Route to get model information for a specific brand and model using query parameters
@router.get("/get-brand-model")
async def get_brand_model(brand_name: str, model_name: str):
    try:
        # Find the brand by name
        brand = await VehicleData.find_one({"name": brand_name})
        if not brand:
            raise HTTPException(status_code=404, detail=f"Brand '{brand_name}' not found")
        # Find the model inside the brand's 'models' array
        model = next((model for model in brand.get("models", []) if model["model"] == model_name), None)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found for brand '{brand_name}'")
        # Return the model data
        return {"model": model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model: {e}")



@router.post("/addvehiclebrand")
async def add_vehicle_brand(brand: Brand):
    existing_brand = await Vehiclecollection.find_one({"brandName": brand.brandName})
    if existing_brand:
        raise HTTPException(status_code=400, detail="Brand already exists.")
    new_brand = {"brandName": brand.brandName,  "models": [] }
    await Vehiclecollection.insert_one(new_brand)
    return {"message": "Brand added successfully."}


@router.post("/addVehiclemodel")
async def add_vehicle_model(brand_name: str, model: VehicleModel, images: List[UploadFile] = File(...)):
    # Find the brand in the database
    brand = await Vehiclecollection.find_one({"brandName": brand_name})
    if not brand:
        raise HTTPException(status_code=404, detail="Brand not found.")
    # Create the model dictionary
    model_dict = model.dict()
    # Encode images to base64
    if images:
        encoded_images = []
        for image in images:
            contents = await image.read()
            encoded_image = base64.b64encode(contents).decode('utf-8')
            encoded_images.append({
                "filename": image.filename,
                "content": encoded_image
            })
        model_dict["images"] = encoded_images  # Store images in the model dict
    # Append the new model to the brand's models array
    if "models" not in brand:
        brand["models"] = []    
    brand["models"].append(model_dict)
    # Update the brand document in the database
    await Vehiclecollection.update_one({"brandName": brand_name}, {"$set": brand})
    return {"message": "Model added successfully."}


# Helper function to serialize the MongoDB documents 
def serialize_vehicle(vehicle: Dict[str, Any]) -> Dict[str, Any]:
    vehicle["_id"] = str(vehicle["_id"])  # Convert ObjectId to string
    return vehicle

@router.get("/vehicles", response_model=List[Dict[str, Any]])
async def get_vehicles():
    vehicles_cursor = Vehiclecollection.find()  # Get a cursor for all documents
    vehicles = await vehicles_cursor.to_list(length=None)  # Fetch all documents into a list
    return [serialize_vehicle(vehicle) for vehicle in vehicles]


@router.get("/getBrandData/{brand_name}", response_model=Optional[BrandModel])
async def get_vehicle_brand(brand_name: str):
    brand_data = await Vehiclecollection.find_one({"brandName": brand_name})
    if brand_data is None:
        raise HTTPException(status_code=404, detail="Brand not found")
    # Convert ObjectId to string for JSON serialization
    brand_data["_id"] = str(brand_data["_id"])
    # Ensure models is a valid list
    if "models" not in brand_data:
        brand_data["models"] = []
    return brand_data



@router.post("/vehicles/{brand_name}/add-model")
async def add_new_model(
    brand_name: str,
    modelName: str = Form(...),
    vehicleType: str = Form(...),
    engineType: str = Form(...),
    description: str = Form(...),
    torque: int = Form(...),
    year: int = Form(...),
    launchPrice: int = Form(...),
    horsepower: int = Form(...),
    seatingCapacity: int = Form(...),
    variants: List[str] = Form(...),
    colors: List[str] = Form(...),
    images: List[UploadFile] = File(...),
):
    # Step 1: Check if brand exists
    brand_document = await Vehiclecollection.find_one({"brandName": brand_name})
    if not brand_document:
        raise HTTPException(status_code=404, detail="Brand not found.")

    # Step 2: Check if model already exists in the models array
    for model in brand_document.get("models", []):
        if model["modelName"].lower() == modelName.lower():
            raise HTTPException(status_code=400, detail=f"Model '{modelName}' already exists.")

    # Step 3: Convert images to base64 and ensure they are under 2MB and are PNG/JPEG
    base64_images = []
    for image in images:
        if image.content_type not in ["image/png", "image/jpeg"]:
            raise HTTPException(status_code=400, detail="Images must be PNG or JPEG format.")

        contents = await image.read()  # Read image contents
        if len(contents) > 2 * 1024 * 1024:  # Check if image is larger than 2MB
            raise HTTPException(status_code=400, detail="Each image must be under 2MB.")

        # Convert image to base64 string
        base64_image = base64.b64encode(contents).decode("utf-8")
        base64_images.append(base64_image)

    # Step 4: Create the new model object
    new_model = {
        "modelName": modelName,
        "vehicleType": vehicleType,
        "engineType": engineType,
        "description": description,
        "torque": torque,
        "year": year,
        "launchPrice": launchPrice,
        "horsepower": horsepower,
        "seatingCapacity": seatingCapacity,
        "variants": variants,  # Variants are now a list
        "colors": colors,      # Colors are now a list
        "images": base64_images,  # Store the base64 images
        "comments": [],  # Initialize comments as an empty array
    }

    # Step 5: Add the new model to the models array of the brand document
    result = await Vehiclecollection.update_one(
        {"brandName": brand_name},
        {"$push": {"models": new_model}},
    )
    if result.modified_count == 1:
        return {"message": "Model added successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to add model. Please try again.")



# Route to fetch the details of a specific model by brand and model name
@router.get("/vehicles/{brand_name}/{model_name}")
async def get_model_data(brand_name: str, model_name: str):
    # Step 1: Find the brand by brand_name
    brand_document = await Vehiclecollection.find_one({"brandName": brand_name})
    if not brand_document:
        raise HTTPException(status_code=404, detail="Brand not found.")
    # Step 2: Find the model within the brand document
    model_data = next((model for model in brand_document.get("models", []) if model["modelName"].lower() == model_name.lower()), None)
    if not model_data:
        print("modelname not found")
        raise HTTPException(status_code=404, detail="Model not found.")
    # Step 3: Return the model data, no modification needed on base64 images
    return model_data



# Update a specific model for a brand
@router.put("/vehicles/{brand_name}/update-model/{model_name}")
async def update_model(
    brand_name: str,
    model_name: str,
    new_modelName: str = Form(...),
    vehicleType: str = Form(...),
    engineType: str = Form(...),
    description: str = Form(...),
    torque: int = Form(...),
    launchPrice: int = Form(...),
    horsepower: int = Form(...),
    seatingCapacity: int = Form(...),
    variants: List[str] = Form(...),
    colors: List[str] = Form(...),
    images: List[UploadFile] = File(None),  # Optional file input
):
    # Step 1: Find the brand by brand_name
    brand_document = await Vehiclecollection.find_one({"brandName": brand_name})
    if not brand_document:
        raise HTTPException(status_code=404, detail="Brand not found.")
    # Step 2: Ensure the model exists and fetch the model by model_name
    model_index = next(
        (index for index, model in enumerate(brand_document.get("models", [])) if model["modelName"].lower() == model_name.lower()),
        None
    )
    if model_index is None:
        raise HTTPException(status_code=404, detail="Model not found.")
    # Step 3: Check for uniqueness of new_modelName in the models array
    for model in brand_document.get("models", []):
        if model["modelName"].lower() == new_modelName.lower() and model["modelName"].lower() != model_name.lower():
            raise HTTPException(status_code=400, detail=f"Model '{new_modelName}' already exists.")
    # Step 4: Convert new images to base64
    base64_images = []
    if images:
        for image in images:
            contents = await image.read()
            base64_image = base64.b64encode(contents).decode('utf-8')
            base64_images.append(base64_image)
    # Step 5: Prepare the updated model data
    updated_model = {
        "modelName": new_modelName,
        "vehicleType": vehicleType,
        "engineType": engineType,
        "description": description,
        "torque": torque,
        "launchPrice": launchPrice,
        "horsepower": horsepower,
        "seatingCapacity": seatingCapacity,
        "variants": variants,
        "colors": colors,
        "images": base64_images if base64_images else brand_document["models"][model_index]["images"]  # If no new images, keep old images
    }
    # Step 6: Update the model data in the array
    brand_document["models"][model_index] = updated_model
    # Step 7: Save the updated brand document back to the database
    result = await Vehiclecollection.update_one(
        {"brandName": brand_name},
        {"$set": {"models": brand_document["models"]}}
    )
    if result.modified_count == 1:
        return {"message": "Model updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update model")



@router.delete("/delete-brand-model")
async def delete_brand_model(data: DeleteModelRequest):
    try:
        # Find the brand by name
        brand = await Vehiclecollection.find_one({"brandName": data.brandName})
        if not brand:
            raise HTTPException(status_code=404, detail=f"Brand '{data.brandName}' not found")
        # Find the model inside the brand's 'models' array
        models = brand.get("models", [])
        model_to_delete = next((model for model in models if model["modelName"] == data.modelName), None)
        if not model_to_delete:
            raise HTTPException(status_code=404, detail=f"Model '{data.modelName}' not found for brand '{data.brandName}'")
        # Remove the model from the 'models' array
        new_models = [model for model in models if model["modelName"] != data.modelName]
        # Update the brand document by setting the new 'models' array
        update_result = await Vehiclecollection.update_one(
            {"brandName": data.brandName},  # Correct filter here
            {"$set": {"models": new_models}}
        )
        if update_result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to delete the model")
        return {"message": f"Model '{data.modelName}' successfully deleted from brand '{data.brandName}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {e}")






























