from pydantic import BaseModel
from typing import List, Optional


# Define the data model for the request
class ModelData(BaseModel):
    id: int
    model: str
    description: str
    launchPrice: str
    vehicleType: str
    seatingCapacity: str
    engineType: str
    colorsAvailable: List[str]
    horsepower: str
    torque: str
    transmission: str
    releaseDate: str
    startingPrice: str
    variants: Optional[List[str]] = None # Make it optional
    img: Optional[List[str]] = None  # Make img optional

class BrandData(BaseModel):
    name: str
    models: List[ModelData]

class VehicleRequest(BaseModel):
    brand: BrandData

