import bcrypt
from bson import ObjectId
from ..config.admindatabase import adminlogininfo
# from ..models.adminmodel import User 
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt



from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer





SECRET_KEY = "Extremely9Sensitive9Super5Secret6Key3"  # Replace with your actual secret key # Use environment variable
ALGORITHM = "HS256" # Use environment variable
ACCESS_TOKEN_EXPIRE_MINUTES = 100


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


async def verify_admin(email: str, password: str):
    user = await adminlogininfo.find_one({"email": email})
    if user is None:
        return "email_not_registered"
    if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return "invalid_password"
    # Create a token and return it
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": email}, expires_delta=access_token_expires)
    return {"token": access_token, "user": serialize_dict(user)}




oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_access_token(token: str, credentials_exception):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return email
    except JWTError:
        raise credentials_exception

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return verify_access_token(token, credentials_exception)