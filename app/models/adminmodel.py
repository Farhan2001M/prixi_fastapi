from pydantic import BaseModel
from typing import List
from typing import List, Dict, Any
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class LoginResponse(BaseModel):
    token: str
    user: dict

class ForgotPasswordRequest(BaseModel):
    email: str

class ValidateOTPRequest(BaseModel):
    email: str
    otp: str

class PasswordChangeRequest(BaseModel):
    email: str
    new_password: str













################# 2nd module ######################





# Add UPDATE DELETE Vehicles

class VehicleModel:
    def __init__(self, vehicleType, modelName, seatingCapacity, engineType, colorsAvailable, torque, variants, images):
        self.vehicleType = vehicleType
        self.modelName = modelName
        self.seatingCapacity = seatingCapacity
        self.engineType = engineType
        self.colorsAvailable = colorsAvailable
        self.torque = torque
        self.variants = variants
        self.images = images





# Request body model
class DeleteModelRequest(BaseModel):
    brandName: str
    modelName: str





# Request model to get specific model info
class ModelRequest(BaseModel):
    brand_name: str
    modelname: str





# Request model for adding a new model
class AddModelRequest(BaseModel):
    brand_name: str
    mymodel_name: str
    description: str
    launch_price: str
    vehicle_type: str
    seating_capacity: str
    engine_type: str
    colors_available: list
    horsepower: str
    torque: str
    transmission: str
    release_date: str
    starting_price: str
    variants: list
    img: list



class ModelDetails(BaseModel):
    modelName: str
    vehicleType: str
    engineType: str
    description: str
    torque: int
    launchPrice: int
    horsepower: int
    seatingCapacity: int
    variants: List[str]
    colors: List[str]
    images: List[str]
    comments: List[Dict[str, Any]]


class BrandResponse(BaseModel):
    brandName: str
    Models: list






class ModelBase(BaseModel):
    modelName: Optional[str] = None
    vehicleType: Optional[str] = None
    engineType: Optional[str] = None
    description: Optional[str] = None
    torque: Optional[int] = None
    year: Optional[int] = None
    launchPrice: Optional[int] = None
    horsepower: Optional[int] = None
    seatingCapacity: Optional[int] = None
    variants: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    images: Optional[List[str]] = None
    comments: Optional[List[Dict[str, Any]]] = None



class BrandModel(BaseModel):
    brandName: str
    models: List[ModelBase] = Field(default_factory=list)  # Allow empty list by default











class Brand(BaseModel):
    brandName: str



class VehicleModel(BaseModel):
    modelName: str
    engineType: str
    vehicleType: str
    transmission: str
    description: str
    launchPrice: float
    seatingCapacity: int
    horsepower: float
    torque: float
    colorsAvailable: List[str] = []
    variants: List[str] = []











# Pydantic model for new model data (without images)
class NewModel(BaseModel):
    modelName: str
    vehicleType: str
    engineType: str
    description: str
    torque: int
    launchPrice: int
    horsepower: int
    seatingCapacity: int
    variants: List[str]
    colors: List[str]
