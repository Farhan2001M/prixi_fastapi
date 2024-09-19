from fastapi import APIRouter, HTTPException
from ..models.VehiclesModel import VehicleRequest
from ..config.VehicleDatabase import collection 
from ..controllers.VehicleControllers import get_new_id



router = APIRouter()




# POST route to save vehicle data
@router.post("/save-vehicles")
async def save_vehicle_data(request: VehicleRequest):
    # Check if the brand already exists
    brand_data = await collection.find_one({"name": request.brand.name})
    if brand_data:
        raise HTTPException(status_code=400, detail="Brand already exists in the database")
    # Get the next ID for the brand
    new_id = await get_new_id(collection)
    # Prepare the document to be inserted
    new_brand_document = {
        "M_id": new_id,
        "name": request.brand.name,
        "models": [model.dict() for model in request.brand.models]
    }
    # Insert the new document
    await collection.insert_one(new_brand_document)
    return {"message": "Brand and models saved successfully", "brand_id": new_id}



@router.get("/get-car-brands")
async def get_car_brands():
    # Fetch only the `id` and `name` fields from the VehicleData collection
    car_brands = await collection.find({}, {"_id": 0, "M_id": 1, "name": 1}).to_list(length=None)
    if not car_brands:
        raise HTTPException(status_code=404, detail="No car brands found")
    return car_brands



@router.get("/get-car-brand/{car_name}/{car_id}")
async def get_car_brand(car_name: str, car_id: int):
    # Find the car brand document that matches the name and ID
    car_brand = await collection.find_one({"name": car_name, "M_id": car_id}, {"_id": 0})
    if not car_brand:
        raise HTTPException(status_code=404, detail="Car brand not found")
    return car_brand


