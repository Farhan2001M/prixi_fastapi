
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
import logging
from bson import ObjectId
from typing import Optional
from ..config.usersdatabase import signupcollectioninfo
from datetime import datetime, timedelta
import bcrypt






SECRET_KEY = "Extremely9Sensitive9Super5Secret6Key3"  
ALGORITHM = "HS256" 
ACCESS_TOKEN_EXPIRE_MINUTES = 100


# Set up logging
logging.basicConfig(level=logging.INFO)  # You can change to DEBUG, ERROR, etc., depending on your needs
logger = logging.getLogger(__name__)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"}, )
    try:
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])        
        email: str = payload.get("sub")
        # If email is not found, raise an exception
        if email is None:
            logger.error("Failed to extract email from the token")
            raise credentials_exception
        # Log success if email is found
        logger.info(f"Successfully authenticated user: {email}")
        return email
    except JWTError as e:
        # Log the exception if there is an error decoding the token
        logger.error(f"JWT Error: {str(e)} - Could not validate credentials")
        raise credentials_exception


async def get_user_by_email(email: str) -> Optional[dict]:
    user = await signupcollectioninfo.find_one({"email": email})
    if user:
        # Remove the password field from the user data
        user_data = {key: value for key, value in user.items() if key != 'password'}
        return user_data    
    return None




# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# def verify_access_token(token: str, credentials_exception: HTTPException):
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         email: str = payload.get("sub")
#         if email is None:
#             logger.error("Failed to extract email from the token")
#             raise credentials_exception
#         logger.info(f"Successfully verified access token for user: {email}")
#         return email
#     except JWTError as e:
#         logger.error(f"JWT Error: {str(e)} - Could not validate credentials")
#         raise credentials_exception

# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     logger.info("Attempting to get current user from token")
#     return verify_access_token(token, credentials_exception)








def serialize_dict(document):
    # Convert MongoDB document to a serializable dictionary, excluding the password.
    if document is None:
        return None
    serialized = {key: str(value) if isinstance(value, ObjectId) else value for key, value in document.items()}
    if 'password' in serialized:
        del serialized['password']  # Remove password from the response
    return serialized



def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_user(email: str, password: str):
    user = await signupcollectioninfo.find_one({"email": email})
    if user is None:
        return "email_not_registered"
    if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return "invalid_password"
    # Create a token and return it
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": email}, expires_delta=access_token_expires)
    return {"token": access_token, "user": serialize_dict(user)}



























from PIL import Image, ImageDraw, ImageFont
import io
import base64
from ..models.usersmodel import User 
from pymongo.errors import DuplicateKeyError



async def create_user(user_data: User):
    try:
        # Check if email already exists
        existing_user = await signupcollectioninfo.find_one({"email": user_data.email})
        if existing_user:
            return None  # Email already exists
        # Hash the password before storing it
        hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
        user_dict = user_data.dict()
        user_dict['password'] = hashed_password
        # Generate initials image and store as base64
        user_dict['GenImage'] = generate_initials_image(user_data.firstName, user_data.lastName)
        # Insert into MongoDB
        result = await signupcollectioninfo.insert_one(user_dict)
        created_user = await signupcollectioninfo.find_one({"_id": result.inserted_id})
        return serialize_dict(created_user)
    except DuplicateKeyError:
        return None  # Handle duplicate email insertion differently if needed
    except Exception as e:
        print("An error occurred during user creation:", e)
        raise HTTPException(status_code=500, detail="Internal Server Error")



def generate_initials_image(first_name: str, last_name: str) -> str:
    initials = (first_name[0] + last_name[0]).upper()
    # Create an image with a black background
    img = Image.new('RGB', (100, 100), color='black')
    draw = ImageDraw.Draw(img)
    # Load font, defaulting if custom font is unavailable
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
    # Center the text
    text_width, text_height = draw.textbbox((0, 0), initials, font=font)[2:]
    position = ((img.width - text_width) // 2, (img.height - text_height) // 2)
    # Draw initials in white
    draw.text(position, initials, fill="white", font=font)
    # Save image to a bytes buffer
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    # Convert the image to base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str









