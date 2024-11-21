from fastapi import APIRouter, HTTPException
from ..models.usersmodel import User , LoginResponse , UserDetailsResponse , UpdateUserRequest , ForgotPasswordRequest , ValidateOTPRequest , PasswordChangeRequest
from ..config.usersdatabase import signupcollectioninfo 
from ..controllers.userSignupControllers import create_user , verify_user , get_current_user , get_user_by_email , generate_initials_image

from fastapi import Depends, HTTPException, status
import base64
from fastapi import UploadFile, File 
import logging
from fastapi import Request
from datetime import datetime, timedelta

import random 
import smtplib
from email.message import EmailMessage
from typing import Dict, Tuple
import bcrypt

from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import io

router = APIRouter()




@router.post("/signup", tags=["User"])
async def signup_user(user: User):
    res = await create_user(user)
    if res:
        return {"message": "User successfully signed up", "user": res}
    if res == None:
        return {"message": "This email is already registered with us."}
    raise HTTPException(400, "Something went wrong")


@router.post("/login", response_model=LoginResponse, tags=["User"])
async def login_user(email: str, password: str):
    result = await verify_user(email, password)    
    # Handle the different failure cases by raising HTTP exceptions
    if result == "email_not_registered":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not registered.")
    if result == "invalid_password":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password.")
    return result    


@router.get("/profileinfo" , tags=["General"])
async def protected_endpoint(current_user: str = Depends(get_current_user)):
    return {"message": "This is a protected endpoint", "user": current_user}


@router.delete("/deleteuser" , tags=["Profile Update"])
async def delete_user(current_user: str = Depends(get_current_user)):
    try:
        # Ensure that delete_one is awaited
        result = await signupcollectioninfo.delete_one({"email": current_user})
        if result.deleted_count == 1:
            return {"message": "User deleted successfully"}
        else:
            # Return a 404 error if the user is not found
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException as e:
        # Directly raise known HTTP exceptions
        raise e
    except Exception as e:
        # Catch other exceptions and provide a generic error message
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.post("/upload-image" , tags=["Profile Update"])
async def upload_image(image: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    # Read the image file
    contents = await image.read()
    # Encode the image in base64
    encoded_image = base64.b64encode(contents).decode('utf-8')
    # Update the user document with the base64 image
    result = await signupcollectioninfo.update_one(
        {"email": current_user},
        {"$set": {"image": encoded_image}} )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Image uploaded successfully"}


# @router.post("/remove-image" , tags=["Profile Update"])
# async def remove_image(current_user: str = Depends(get_current_user)):
#     # Remove the image field from the user's document
#     result = await signupcollectioninfo.update_one(
#         {"email": current_user},
#         {"$unset": {"image": ""}} )
#     if result.modified_count == 0:
#         raise HTTPException(status_code=404, detail="User not found or image not removed")
#     return {"message": "Image removed successfully"}


@router.post("/remove-image", tags=["Profile Update"])
async def remove_image(current_user: str = Depends(get_current_user)):
    # Remove the custom image field from the user's document
    result = await signupcollectioninfo.update_one(
        {"email": current_user},
        {"$unset": {"image": ""}}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="User not found or image not removed")

    # Fetch the stored initials image (`GenImage`) for this user
    user = await signupcollectioninfo.find_one({"email": current_user}, {"GenImage": 1})
    if user is None or "GenImage" not in user:
        raise HTTPException(status_code=404, detail="Initials image not found for this user")

    # Return the success message and the stored initials image
    return {"message": "Image removed successfully", "GenImage": user["GenImage"]}


@router.get("/user-image", tags=["Profile Update"])
async def get_user_image(current_user: str = Depends(get_current_user)):
    user = await signupcollectioninfo.find_one({"email": current_user}, {"firstName": 1, "lastName": 1, "image": 1})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    # Retrieve the custom image if it exists; otherwise, provide GenImage
    custom_image = user.get("image", None)
    gen_image = user.get("GenImage") or generate_initials_image(user["firstName"], user["lastName"])
    return {"image": custom_image, "GenImage": gen_image, "imageType": "png"}



# @router.get("/user-image", tags=["Profile Update"])
# async def get_user_image(current_user: str = Depends(get_current_user)):
#     user = await signupcollectioninfo.find_one({"email": current_user}, {"firstName": 1, "lastName": 1, "image": 1})
#     if user is None:
#         raise HTTPException(status_code=404, detail="User not found")
#     # Check if a custom image is set; if not, generate initials-based image
#     image = user.get("image", None)
#     if image is None:
#         # Generate initials image if no custom image is set
#         image = generate_initials_image(user["firstName"], user["lastName"])
#     return {"image": image, "imageType": "png"}  # Specify that this is a PNG image



@router.get("/getfulluserinfo", response_model=UserDetailsResponse , tags=["Profile Update"])
async def get_full_user_info(current_user: str = Depends(get_current_user)):
    user = await get_user_by_email(current_user)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user



@router.post("/updateuser" , tags=["Profile Update"])
async def update_user_details( update_data: UpdateUserRequest, current_user: str = Depends(get_current_user) ):
    update_fields = {key: value for key, value in update_data.dict().items() if value is not None}
    if not update_fields:
        raise HTTPException(status_code=400, detail="No update fields provided.")
    result = await signupcollectioninfo.update_one(
        {"email": current_user},  # Assuming email is used to identify the user
        {"$set": update_fields} )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"message": "User details updated successfully"}

otp_storage: Dict[str, Tuple[str, datetime]] = {}

@router.post("/forgot-password", tags=["User"])
async def forgot_password(request: ForgotPasswordRequest):
    email = request.email
    user = await signupcollectioninfo.find_one({"email": email})
    
    if user is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not registered.")
    
    # Generate a 6-digit OTP
    otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    print(f"Generated OTP: {otp}")
    
    # Email configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    from_email = 'prixihelpcentre@gmail.com'
    from_password = "jgtn fvsj ymuc wzje"  # Use environment variable for the password

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
    
    await signupcollectioninfo.update_one(
        {"email": email},
        {"$set": {"otp": {"code": otp, "expiry": otp_expiry}}}
    )
    return {"message": "OTP sent successfully."}


@router.post("/validate-otp", tags=["User"])
async def validate_otp(request: ValidateOTPRequest):
    email = request.email
    entered_otp = request.otp

    # Fetch the user document
    user = await signupcollectioninfo.find_one({"email": email})
    
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
    await signupcollectioninfo.update_one(
        {"email": email},
        {"$unset": {"otp": ""}}  # Remove the otp field from the document
    )
    
    return {"message": "OTP validated successfully."}



@router.post("/change-password", tags=["User"])
async def change_password(request: PasswordChangeRequest):
    user = await signupcollectioninfo.find_one({"email": request.email})
    if not user:
        raise HTTPException(404, "User not found")
    # Hash the new password
    hashed_password = bcrypt.hashpw(request.new_password.encode('utf-8'), bcrypt.gensalt())
    # Update the user's password in the database
    try:
        await signupcollectioninfo.update_one(
            {"email": request.email}, 
            {"$set": {"password": hashed_password}}
        )
        return {"message": "Password successfully updated"}
    except Exception as e:
        raise HTTPException(500, "An error occurred while updating the password")














