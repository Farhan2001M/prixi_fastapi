from pydantic import BaseModel
from typing import List , Optional


class User(BaseModel):
    firstName: str
    lastName: str
    email: str
    phoneNumber: str
    password: str
    image: Optional[str] = None  # Default to None if not provided  # Base64 encoded image string
    favorites: Optional[List[dict]] = []  # To store user's favorite vehicles

class LoginResponse(BaseModel):
    token: str
    user: dict


class UserDetailsResponse(BaseModel):
    firstName: str
    lastName: str
    email: str
    phoneNumber: str
    image: Optional[str] = None


class UpdateUserRequest(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    phoneNumber: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    email: str


class ValidateOTPRequest(BaseModel):
    email: str
    otp: str


class PasswordChangeRequest(BaseModel):
    email: str
    new_password: str