from fastapi import APIRouter, HTTPException
from ..models.VehiclesModel import VehicleRequest
from ..config.VehicleDatabase import collection 
from ..controllers.VehicleControllers import get_new_id
from ..controllers.userSignupControllers import create_user , verify_user , get_current_user , get_user_by_email 

from fastapi import Depends
router = APIRouter()
from pydantic import BaseModel
from typing import List, Optional
from bson import ObjectId
from datetime import datetime
import uuid


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
        {
            "brandName": brand_name,
            "models.modelName": model_name
        },
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
    current_user: str = Depends(get_current_user)  # Assuming current_user gets user's email from token
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
        {"models.$": 1}
    )
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
        {"$set": {"models.$.comments": comments}}
    )
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
        {"models.$": 1}
    )
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
    current_user: str = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Find the document with the matching brand and model
    result = await collection.find_one(
        {"brandName": brand_name, "models.modelName": model_name},
        {"models.$": 1}
    )
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









