from pydantic import BaseModel
from typing import Optional


class User(BaseModel):
    firstName: str
    lastName: str
    email: str
    phoneNumber: str
    password: str
    image: Optional[str] = None  # Default to None if not provided  # Base64 encoded image string


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