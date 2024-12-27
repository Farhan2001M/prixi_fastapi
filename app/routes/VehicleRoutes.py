from fastapi import APIRouter, HTTPException
from ..models.VehiclesModel import VehicleRequest
from ..config.VehicleDatabase import collection 
from ..controllers.VehicleControllers import get_new_id
from ..controllers.userSignupControllers import  get_current_user  
from ..config.usersdatabase import signupcollectioninfo 

from fastapi import Depends
router = APIRouter()
from pydantic import BaseModel
from typing import List, Optional
from bson import ObjectId
from datetime import datetime
import uuid
import logging



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
    # Fetch only the `brandName` field from the Vehicles collection
    car_brands = await collection.find({}, {"_id": 0, "brandName": 1}).to_list(length=None)
    if not car_brands:
        raise HTTPException(status_code=404, detail="No car brands found")
    return car_brands

@router.get("/get-car-brand/{brand_name}")
async def get_car_brand(brand_name: str):
    # Find the car brand document that matches the `brandName`
    car_brand = await collection.find_one({"brandName": brand_name}, {"_id": 0})
    if not car_brand:
        raise HTTPException(status_code=404, detail="Car brand not found")
    return car_brand


@router.get("/get-brand-model/{brand_name}/{model_name}")
async def get_model_data(brand_name: str, model_name: str):
    try:
        # Find the brand document that contains the specified model
        brand_data = await collection.find_one(
            {"brandName": brand_name, "models.modelName": model_name},
            {"models.$": 1}  # Return only the matched model from the models array
        )
        if not brand_data or not brand_data.get("models"):
            raise HTTPException(status_code=404, detail="Brand or Model not found")
        model_data = brand_data['models'][0]  # The model data we are interested in
        return model_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    



@router.get("/get-brands-with-images")
async def get_brands_with_images():
    try:
        # Aggregation to fetch brandName and up to 5 images (first image of each model)
        brands = await collection.aggregate([
            {
                "$project": {
                    "_id": 0,
                    "brandName": 1,
                    # Get the first image from each model, limit to 5 models
                    "modelImages": {
                        "$slice": [
                            {
                                "$map": {
                                    "input": {"$slice": ["$models", 5]},  # First 5 models
                                    "as": "model",
                                    "in": {"$arrayElemAt": ["$$model.images", 0]}  # First image of each model
                                }
                            },
                            5  # Limit to 5 images
                        ]
                    }
                }
            }
        ]).to_list(length=None)
        if not brands:
            raise HTTPException(status_code=404, detail="No brands found")
        return brands
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.get("/get-brand-images/{brand_name}")
async def get_brand_images(brand_name: str):
    try:
        # Fetch only the first image from each model's images array for the given brand
        brand_data = await collection.find_one(
            {"brandName": brand_name},
            {
                "_id": 0,
                "brandName": 1,
                "models.images": {"$slice": 1}  # First image of each model
            }
        )
        if not brand_data:
            raise HTTPException(status_code=404, detail="Brand not found")
        # Extract the first image of each model
        images = [model["images"][0] for model in brand_data.get("models", []) if model.get("images")]
        return {"brandName": brand_data["brandName"], "images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Comment model
class Comment(BaseModel):
    commentId: str
    userEmail: str
    commentText: str
    parentId: str = None
    Likes: List[str] = []
    timestamp: str
    replies: List['Comment'] = []

@router.get("/get-comments/{brand_name}/{model_name}", response_model=List[dict])
async def get_comments(brand_name: str, model_name: str):
    # Find the brand and specific model by brand name and model name
    result = await collection.find_one(
        { "brandName": brand_name, "models.modelName": model_name },
        {"models.$": 1}  # Project only the specific model's comments
    )
    # Check if the brand and model exist
    if result and 'models' in result and len(result['models']) > 0:
        model_data = result['models'][0]
        if 'comments' in model_data:
            return model_data['comments']  # Return the comments array
        else:
            return []  # If no comments exist
    else:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    

class CommentRequest(BaseModel):
    commentText: str

async def add_comment_to_model(brand_name: str, model_name: str, comment: dict):
    # Add the comment to the specific model's comments array
    result = await collection.update_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"$push": {"models.$.comments": comment}}
    )
    return result

@router.post("/post-comment/{brand_name}/{model_name}", tags=["Comments"])
async def post_comment(
    brand_name: str,
    model_name: str,
    comment_data: CommentRequest,
    current_user: str = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Create a new comment
    comment = {
        "commentId": str(uuid.uuid4()),  # Generate unique comment ID
        "userEmail": current_user,  # User's email
        "commentText": comment_data.commentText,  # Comment content
        "Likes": [],  # Initialize with no likes
        "timestamp": datetime.utcnow().isoformat(),  # Timestamp in ISO format
        "replies": []  # Initialize with no replies
    }
    # Add comment to the model
    update_result = await add_comment_to_model(brand_name, model_name, comment)
    if update_result.modified_count == 1:
        return {"message": "Comment added successfully"}
    else:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    


# Helper function to find and delete a comment by its ID
async def remove_comment_from_model(brand_name: str, model_name: str, comment_id: str, user_email: str):
    # Find the document with the matching brand and model
    result = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"models.$": 1} )
    if not result:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    # Get the specific model's comments
    model = result["models"][0]
    comments = model.get("comments", [])
    # Check if the comment exists and if the current user is the owner
    for comment in comments:
        if comment["commentId"] == comment_id:
            if comment["userEmail"] != user_email:
                raise HTTPException(status_code=403, detail="You can only delete your own comments.")
            comments.remove(comment)
            break
    else:
        raise HTTPException(status_code=404, detail="Comment not found")
    # Update the comments array after deletion
    update_result = await collection.update_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"$set": {"models.$.comments": comments}} )
    if update_result.modified_count == 1:
        return {"message": "Comment deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete the comment")
    
@router.delete("/delete-comment/{brand_name}/{model_name}/{comment_id}", tags=["Comments"])
async def delete_comment(brand_name: str, model_name: str, comment_id: str, current_user: str = Depends(get_current_user)):
    # Ensure that the current user is the owner of the comment before deleting it
    return await remove_comment_from_model(brand_name, model_name, comment_id, current_user)


@router.get("/check-comment-owner/{brand_name}/{model_name}/{comment_id}", tags=["Comments"])
async def check_comment_owner(brand_name: str, model_name: str, comment_id: str, current_user: str = Depends(get_current_user)):
    # Find the document with the matching brand and model
    result = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"models.$": 1} )
    if not result:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    model = result["models"][0]
    comments = model.get("comments", [])
    for comment in comments:
        if comment["commentId"] == comment_id:
            return {"isOwner": comment["userEmail"] == current_user}
    raise HTTPException(status_code=404, detail="Comment not found")


@router.put("/edit-comment/{brand_name}/{model_name}/{comment_id}", tags=["Comments"])
async def edit_comment(
    brand_name: str,
    model_name: str,
    comment_id: str,
    comment_data: CommentRequest,
    current_user: str = Depends(get_current_user) ):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Find the document with the matching brand and model
    result = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"models.$": 1} )
    if not result:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    model = result["models"][0]
    comments = model.get("comments", [])
    # Check if the comment exists and if the current user is the owner
    for comment in comments:
        if comment["commentId"] == comment_id:
            if comment["userEmail"] != current_user:
                raise HTTPException(status_code=403, detail="You can only edit your own comments.")
            # Update the comment text
            comment["commentText"] = comment_data.commentText
            break
    else:
        raise HTTPException(status_code=404, detail="Comment not found")
    # Update the comments array after editing
    update_result = await collection.update_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"$set": {"models.$.comments": comments}}  # Make sure this correctly updates the array
    )
    if update_result.modified_count == 1:
        return {"message": "Comment updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update the comment")



class ReplyRequest(BaseModel):
    commentText: str

async def add_reply_to_comment(brand_name: str, model_name: str, comment_id: str, reply: dict):
    # Add the reply to the specific comment's replies array
    result = await collection.update_one(
        {"brandName": brand_name, "models.modelName": model_name, "models.comments.commentId": comment_id},
        {"$push": {"models.$.comments.$[comment].replies": reply}},
        array_filters=[{"comment.commentId": comment_id}]  # Use array filters to target the specific comment
    )
    return result

@router.post("/post-reply/{brand_name}/{model_name}/{comment_id}", tags=["Comments"])
async def post_reply(
    brand_name: str,
    model_name: str,
    comment_id: str,
    reply_data: ReplyRequest,
    current_user: str = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Create a new reply
    reply = {
        "commentId": str(uuid.uuid4()),  # Generate unique reply ID
        "userEmail": current_user,  # User's email
        "commentText": reply_data.commentText,  # Reply content
        "Likes": [],  # Initialize with no likes
        "timestamp": datetime.utcnow().isoformat(),  # Timestamp in ISO format
    }
    # Add reply to the comment
    update_result = await add_reply_to_comment(brand_name, model_name, comment_id, reply)
    if update_result.modified_count == 1:
        return {"message": "Reply added successfully"}
    else:
        raise HTTPException(status_code=404, detail="Brand or model not found")




@router.delete("/delete-reply/{brand_name}/{model_name}/{comment_id}/{reply_id}", tags=["Comments"])
async def delete_reply(brand_name: str, model_name: str, comment_id: str, reply_id: str, current_user: str = Depends(get_current_user)):
    # Ensure that the current user is the owner of the reply before deleting it
    result = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name, "models.comments.commentId": comment_id},
        {"models.$": 1}
    )
    if not result:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    model = result["models"][0]
    comments = model.get("comments", [])
    # Find the specific comment and check for the reply
    for comment in comments:
        if comment["commentId"] == comment_id:
            for reply in comment["replies"]:
                if reply["commentId"] == reply_id:
                    if reply["userEmail"] != current_user:
                        raise HTTPException(status_code=403, detail="You can only delete your own replies.")
                    comment["replies"].remove(reply)
                    break
            else:
                raise HTTPException(status_code=404, detail="Reply not found")
            break
    else:
        raise HTTPException(status_code=404, detail="Comment not found")
    # Update the comments array after deletion
    update_result = await collection.update_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"$set": {"models.$.comments": comments}}
    )
    if update_result.modified_count == 1:
        return {"message": "Reply deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete the reply")


@router.get("/check-reply-owner/{brand_name}/{model_name}/{comment_id}/{reply_id}", tags=["Comments"])
async def check_reply_owner(
    brand_name: str,
    model_name: str,
    comment_id: str,
    reply_id: str,
    current_user: str = Depends(get_current_user)
):
    print(f"Current user: {current_user}")  # Log the current user to check
    print(f"brand_name : {brand_name}")  # Log the current user to check
    print(f"model_name : {model_name}")  # Log the current user to check
    print(f"comment_id : {comment_id}")  # Log the current user to check
    print(f"reply_id : {reply_id}")  # Log the current user to check

    # Find the document with the matching brand, model, and comment
    result = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"models.$": 1}  # Project only the specific model data
    )
    if not result:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    model = result["models"][0]
    comments = model.get("comments", [])
    # Locate the comment with the matching comment_id
    for comment in comments:
        if comment["commentId"] == comment_id:
            # Locate the reply with the matching reply_id
            for reply in comment.get("replies", []):
                if reply["commentId"] == reply_id:
                    # Return whether the current user is the owner of the reply
                    return {"isOwner": reply["userEmail"] == current_user}
            raise HTTPException(status_code=404, detail="Reply not found")
    raise HTTPException(status_code=404, detail="Comment not found")




@router.put("/edit-reply/{brand_name}/{model_name}/{comment_id}/{reply_id}", tags=["Comments"])
async def edit_reply(
    brand_name: str,
    model_name: str,
    comment_id: str,
    reply_id: str,
    reply_data: ReplyRequest,
    current_user: str = Depends(get_current_user)
):
    # Find the document containing the brand, model, and comment
    result = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name, "models.comments.commentId": comment_id},
        {"models.$": 1}
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Brand, model, or comment not found")
    
    model = result["models"][0]
    comments = model.get("comments", [])
    
    # Locate the comment and the reply
    for comment in comments:
        if comment["commentId"] == comment_id:
            for reply in comment.get("replies", []):
                if reply["commentId"] == reply_id:
                    # Check if the user is the owner of the reply
                    if reply["userEmail"] != current_user:
                        raise HTTPException(status_code=403, detail="You can only edit your own replies.")
                    
                    # Update the reply text
                    reply["commentText"] = reply_data.commentText
                    break
            else:
                raise HTTPException(status_code=404, detail="Reply not found")
            break
    else:
        raise HTTPException(status_code=404, detail="Comment not found")

    # Perform the update in the database using array filters for nested updates
    update_result = await collection.update_one(
        {"brandName": brand_name, "models.modelName": model_name, "models.comments.commentId": comment_id},
        {"$set": {"models.$.comments.$[comment].replies.$[reply].commentText": reply_data.commentText}},
        array_filters=[
            {"comment.commentId": comment_id},  # Targets the correct comment
            {"reply.commentId": reply_id}       # Targets the correct reply
        ]
    )
    
    if update_result.modified_count == 1:
        return {"message": "Reply updated successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update the reply")


@router.get("/user/email")
async def get_user_email(current_user: str = Depends(get_current_user)):
    return {"email": current_user}


@router.post("/like-comment/{brand_name}/{model_name}/{comment_id}", tags=["Comments"])
async def like_comment(
    brand_name: str,
    model_name: str,
    comment_id: str,
    current_user: str = Depends(get_current_user)
):
    # Find the document with the matching brand and model
    result = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"models.$": 1}  # Only project the models array
    )
    if not result:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    model = result["models"][0]
    comments = model.get("comments", [])
    for comment in comments:
        if comment["commentId"] == comment_id:
            # Check if the current user has already liked the comment
            if current_user in comment.get("Likes", []):  # Note the uppercase "L" in "Likes"
                # Unlike the comment (remove the user's email from the Likes array)
                update_result = await collection.update_one(
                    {"brandName": brand_name, "models.modelName": model_name, "models.comments.commentId": comment_id},
                    {"$pull": {"models.$[model].comments.$[comment].Likes": current_user}},
                    array_filters=[{"model.modelName": model_name}, {"comment.commentId": comment_id}]
                )
                if update_result.modified_count == 1:
                    return {"message": "Unliked the comment"}
            else:
                # Like the comment (add the user's email to the Likes array)
                update_result = await collection.update_one(
                    {"brandName": brand_name, "models.modelName": model_name, "models.comments.commentId": comment_id},
                    {"$push": {"models.$[model].comments.$[comment].Likes": current_user}},
                    array_filters=[{"model.modelName": model_name}, {"comment.commentId": comment_id}]
                )
                if update_result.modified_count == 1:
                    return {"message": "Liked the comment"}
    
    raise HTTPException(status_code=404, detail="Comment not found")


# Model for the favorite vehicle
class FavoriteVehicle(BaseModel):
    brandName: str
    modelName: str

@router.get("/favorites", tags=["Favorites"])
async def get_favorites(current_user: str = Depends(get_current_user)):
    user = await signupcollectioninfo.find_one({"email": current_user}, {"favorites": 1})
    if not user:
        return {"favorites": []}
    return {"favorites": user.get("favorites", [])}


@router.post("/favorites/add", tags=["Favorites"])
async def add_to_favorites(favorite: FavoriteVehicle, current_user: str = Depends(get_current_user)):
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    existing_favorites = user.get("favorites", [])
    for fav in existing_favorites:
        if fav["modelName"] == favorite.modelName and fav["brandName"] == favorite.brandName:
            raise HTTPException(status_code=400, detail="Vehicle already in favorites")
    await signupcollectioninfo.update_one(
        {"email": current_user},
        {"$push": {"favorites": favorite.dict()}} )
    return {"message": "Vehicle added to favorites"}


# Endpoint to return full details of favorited vehicles (for the favorites page)
@router.get("/favorites/details", tags=["Favorites"])
async def get_detailed_favorites(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Fetch the user's favorites from their document
    user = await signupcollectioninfo.find_one({"email": current_user}, {"favorites": 1})
    if not user or "favorites" not in user:
        return {"favorites": []}
    # Prepare a list to store full vehicle details for each favorite
    favorite_details = []
    # Iterate over user's favorites and fetch details from Vehicles collection
    for favorite in user["favorites"]:
        vehicle = await collection.find_one(
            {"brandName": favorite["brandName"], "models.modelName": favorite["modelName"]},
            {"models.$": 1, "brandName": 1}  # Fetch only the matched model and brand name
        )
        if vehicle and "models" in vehicle:
            favorite_details.append({
                "brandName": vehicle["brandName"],
                "model": vehicle["models"][0]  # Since we fetched with the $ operator, this will be the exact model
            })
    return {"favorites": favorite_details}



@router.post("/favorites/remove", tags=["Favorites"])
async def remove_from_favorites(favorite: FavoriteVehicle, current_user: str = Depends(get_current_user)):
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await signupcollectioninfo.update_one(
        {"email": current_user},
        {"$pull": {"favorites": {"modelName": favorite.modelName, "brandName": favorite.brandName}}} )
    return {"message": "Vehicle removed from favorites"}


class TrackBrandVisitRequest(BaseModel):
    brandName: str

@router.post("/track-brand-visit", tags=["Statistics"])
async def track_brand_visit(
    brandName: str,  # This will automatically extract the brandName from the query parameters
    current_user: str = Depends(get_current_user) ):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Initialize the statistics attribute if it doesn't exist
    if "statistics" not in user:
        user["statistics"] = {"brandVisited": {}}
    stats = user["statistics"]
    # Track the brand visit (increment the count)
    stats["brandVisited"][brandName] = stats["brandVisited"].get(brandName, 0) + 1
    # Save the updated statistics back to the database
    await signupcollectioninfo.update_one(
        {"email": current_user},
        {"$set": {"statistics": stats}} )
    return {"message": "Brand visit tracked successfully"}


class TrackModelVisit(BaseModel):
    brandName: str
    modelName: str
    vehicleType: str  # One of: Sedan, SUV, Coupe, Hatchback, Pickup-Truck
    engineType: str   # One of: Petrol, Diesel, Hybrid, Electric
    price: Optional[float] = 0.0  # Price of the model
    horsepower: Optional[float] = 0.0  # Horsepower of the model
    torque: Optional[float] = 0.0  # Torque of the model
    year: int  # Year of the model

@router.post("/track-model-visit", tags=["Statistics"])
async def track_visit(data: TrackModelVisit, current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")    
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")    
    # Ensure the "statistics" field exists
    if "statistics" not in user:
        user["statistics"] = {
            "totalVisitedCards": 0,
            "cumulativePrice": 0.0,
            "averagePrice": 0.0,
            "cumulativeHorsepower": 0.0,
            "averageHorsepower": 0.0,
            "cumulativeTorque": 0.0,
            "averageTorque": 0.0,
            "uniqueVisitedModels": [],
            "VehicleTypesVisited": {},
            "EngineTypesVisited": {},
            "yearRangesVisited": {}
        }    
    stats = user["statistics"]    
    # Initialize missing fields explicitly in case they are missing
    stats["totalVisitedCards"] = stats.get("totalVisitedCards", 0)
    stats["cumulativePrice"] = stats.get("cumulativePrice", 0.0)
    stats["averagePrice"] = stats.get("averagePrice", 0.0)
    stats["cumulativeHorsepower"] = stats.get("cumulativeHorsepower", 0.0)
    stats["averageHorsepower"] = stats.get("averageHorsepower", 0.0)
    stats["cumulativeTorque"] = stats.get("cumulativeTorque", 0.0)
    stats["averageTorque"] = stats.get("averageTorque", 0.0)
    stats["uniqueVisitedModels"] = stats.get("uniqueVisitedModels", [])
    stats["VehicleTypesVisited"] = stats.get("VehicleTypesVisited", {})
    stats["EngineTypesVisited"] = stats.get("EngineTypesVisited", {})
    stats["yearRangesVisited"] = stats.get("yearRangesVisited", {})
    # 1. Increment total visited cards count
    stats["totalVisitedCards"] += 1    
    # 2. Track unique visits
    model_identifier = f"{data.brandName}_{data.modelName}"
    if model_identifier not in stats["uniqueVisitedModels"]:
        stats["uniqueVisitedModels"].append(model_identifier)        
        # Update cumulative values and calculate averages
        stats["cumulativePrice"] += data.price
        stats["averagePrice"] = stats["cumulativePrice"] / len(stats["uniqueVisitedModels"])        
        if data.horsepower:
            stats["cumulativeHorsepower"] += data.horsepower
            stats["averageHorsepower"] = stats["cumulativeHorsepower"] / len(stats["uniqueVisitedModels"])        
        if data.torque:
            stats["cumulativeTorque"] += data.torque
            stats["averageTorque"] = stats["cumulativeTorque"] / len(stats["uniqueVisitedModels"])    
    # 3. Track Vehicle Types Visited
    stats["VehicleTypesVisited"][data.vehicleType] = stats["VehicleTypesVisited"].get(data.vehicleType, 0) + 1    
    # 4. Track Engine Types Visited
    stats["EngineTypesVisited"][data.engineType] = stats["EngineTypesVisited"].get(data.engineType, 0) + 1    
    # 5. Track Year Range Visited
    year_range_start = (data.year // 5) * 5
    year_range_end = year_range_start + 5
    year_range_key = f"{year_range_start}-{year_range_end}"
    stats["yearRangesVisited"][year_range_key] = stats["yearRangesVisited"].get(year_range_key, 0) + 1    
    # Save the updated statistics back to the database
    await signupcollectioninfo.update_one(
        {"email": current_user},
        {"$set": {"statistics": stats}}  )    
    return {"message": "Visit tracked successfully"}





@router.get("/get-user-visited-brands", tags=["Statistics"])
async def get_user_visited_brands(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user or "statistics" not in user or "brandVisited" not in user["statistics"]:
        return {"visitedBrands": []}
    
    visited_brands = list(user["statistics"]["brandVisited"].keys())
    return {"visitedBrands": visited_brands}


@router.get("/get-anonymous-comments/{brand_name}/{model_name}", tags=["Comments"])
async def get_anonymous_comments(brand_name: str, model_name: str):
    # Fetch the vehicle model document
    document = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"models.$": 1}  # Fetch only the specific model
    )
    if not document:
        raise HTTPException(status_code=404, detail="Brand or model not found")
    # Extract anonymous opinions
    model = document.get("models", [{}])[0]
    anonymous_opinions = model.get("AnonymousAsk", [])
    return {"anonymousComments": anonymous_opinions}


async def add_anonymous_comment_to_model(brand_name: str, model_name: str, anony_comment: dict):
    # Add the anonymous comment to the AnonymousAsk array
    result = await collection.update_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"$push": {"models.$.AnonymousAsk": anony_comment}}
    )
    return result

@router.post("/post-anonymous-comment/{brand_name}/{model_name}", tags=["Comments"])
async def post_anonymous_comment(
    brand_name: str,
    model_name: str,
    anony_data: dict
):
    # Create a new anonymous comment
    anony_comment = {
        "anonyTextId": str(uuid.uuid4()),  # Unique ID
        "anonyText": anony_data["anonyText"],  # Anonymous comment text
        "timestamp": anony_data["timestamp"],  # Timestamp
    }
    # Add anonymous comment to the model
    update_result = await add_anonymous_comment_to_model(brand_name, model_name, anony_comment)
    if update_result.modified_count == 1:
        return {"message": "Anonymous comment added successfully"}
    else:
        raise HTTPException(status_code=404, detail="Brand or model not found")




@router.get("/user-overview", tags=["Statistics"])
async def get_user_overview(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Fetch the user document from MongoDB
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch statistics and favorites data
    statistics = user.get("statistics", {})
    favorites = user.get("favorites", [])
    
    # 1. Total Unique Vehicle Models Visited
    total_unique_models = len(statistics.get("uniqueVisitedModels", []))
    
    # 2. Total number of favorite vehicles
    total_favorites = len(favorites)
    
    # 3. Average Price of Viewed Vehicles
    average_price = statistics.get("averagePrice", 0.0)
    
    # 4. Average Torque of Vehicles Viewed
    average_torque = statistics.get("averageTorque", 0.0)
    
    # 5. Average Horsepower of Vehicles Viewed
    average_horsepower = statistics.get("averageHorsepower", 0.0)
    
    # 6. Top 3 Most Visited Brands
    top_brands = sorted(statistics.get("brandVisited", {}).items(), key=lambda x: x[1], reverse=True)[:3]
    
    # 7. Most Favorite Engine Type of User
    most_favorite_engine = max(statistics.get("EngineTypesVisited", {}).items(), key=lambda x: x[1], default=None)
    most_favorite_engine = most_favorite_engine[0] if most_favorite_engine else None
    
    # 8. Most Favorite Vehicle Type of User
    most_favorite_vehicle_type = max(statistics.get("VehicleTypesVisited", {}).items(), key=lambda x: x[1], default=None)
    most_favorite_vehicle_type = most_favorite_vehicle_type[0] if most_favorite_vehicle_type else None
    
    # 9. Most Viewed Vehicle Year Range of User
    most_viewed_year_range = max(statistics.get("yearRangesVisited", {}).items(), key=lambda x: x[1], default=None)
    most_viewed_year_range = most_viewed_year_range[0] if most_viewed_year_range else None

    # Prepare the response data
    return {
        "totalUniqueModels": total_unique_models,
        "totalFavorites": total_favorites,
        "averagePrice": average_price,
        "averageTorque": average_torque,
        "averageHorsepower": average_horsepower,
        "topBrands": top_brands,
        "mostFavoriteEngineType": most_favorite_engine,
        "mostFavoriteVehicleType": most_favorite_vehicle_type,
        "mostViewedYearRange": most_viewed_year_range,
        "description": "Overview of your statistics and favorite vehicles"
    }





@router.get("/user-data-summary", tags=["User Data"])
async def get_user_data_summary(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Initialize counters
    total_comments = 0
    total_replies = 0
    total_replies_received = 0
    total_models = 0

    # Aggregation Pipeline
    pipeline = [
        {"$unwind": "$models"},  # Unwind the models array to work with individual models
        {
            "$project": {
                "modelName": "$models.modelName",  # Include the model name
                "comments": "$models.comments",  # Only include the comments array of each model
            }
        },
        {"$match": {}},  # No specific filter for brands/models; we want all vehicles
    ]
    
    async for doc in collection.aggregate(pipeline):
        total_models += 1  # Each iteration represents a model in the collection
        
        # Access comments safely using .get() to avoid KeyError if comments field doesn't exist
        comments = doc.get("comments", [])

        # Loop through each model's comments if they exist
        for comment in comments:
            # Count the comments made by the current user
            if comment.get("userEmail") == current_user:
                total_comments += 1
            
            # Loop through replies to count the ones posted by the user
            for reply in comment.get("replies", []):
                if reply.get("userEmail") == current_user:
                    total_replies += 1  # Count replies posted by the user
                
                # Count replies the user received
                if reply.get("userEmail") != current_user:
                    total_replies_received += 1  # Count replies received by the user

    # Log the results to the console (VS Code terminal)
    logging.info(f"Total Comments: {total_comments} , Total Replies: {total_replies} , Total Replies Received: {total_replies_received} , Total Models: {total_models}")

    return {
        "totalComments": total_comments,
        "totalReplies": total_replies,
        "totalRepliesReceived": total_replies_received,
        "totalModels": total_models
    }












@router.get("/brand-visits-chart", tags=["Charts"])
async def get_brand_visits_chart(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Fetch the user document
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Extract brand visits
    brand_visits = user.get("statistics", {}).get("brandVisited", {})
    # Prepare the response
    return {
        "brands": list(brand_visits.keys()),  # X-axis labels
        "visits": list(brand_visits.values())  # Y-axis values
    }


@router.get("/vehicle-types-donut", tags=["Charts"])
async def get_vehicle_types_donut(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Fetch the user document
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Extract vehicle type data
    vehicle_types = user.get("statistics", {}).get("VehicleTypesVisited", {})
    total_visits = sum(vehicle_types.values())
    if total_visits == 0:
        raise HTTPException(status_code=400, detail="No data available for Vehicle Types")
    # Calculate percentages
    percentages = {k: (v / total_visits) * 100 for k, v in vehicle_types.items()}
    # Prepare the response
    return {
        "vehicleTypes": list(percentages.keys()),
        "percentages": list(percentages.values())
    }


@router.get("/engine-types-donut", tags=["Charts"])
async def get_engine_types_donut(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Fetch the user document
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Extract engine type data
    engine_types = user.get("statistics", {}).get("EngineTypesVisited", {})
    total_visits = sum(engine_types.values())
    if total_visits == 0:
        raise HTTPException(status_code=400, detail="No data available for Engine Types")
    # Calculate percentages
    percentages = {k: (v / total_visits) * 100 for k, v in engine_types.items()}
    # Prepare the response
    return {
        "engineTypes": list(percentages.keys()),
        "percentages": list(percentages.values())
    }



@router.get("/year-ranges-chart", tags=["Charts"])
async def get_year_ranges_chart(current_user: str = Depends(get_current_user)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Fetch the user document
    user = await signupcollectioninfo.find_one({"email": current_user})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Extract year range data
    year_ranges = user.get("statistics", {}).get("yearRangesVisited", {})
    # Prepare the response
    return {
        "yearRanges": list(year_ranges.keys()),  # Y-axis labels
        "visits": list(year_ranges.values())  # X-axis values
    }












# def determine_year_range(year: int) -> str:
#     """Determine the 5-year range for a given year."""
#     start_year = (year // 5) * 5
#     end_year = start_year + 4
#     return f"{start_year}-{end_year}"
