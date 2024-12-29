from fastapi import APIRouter, HTTPException
import logging
from ..config.usersdatabase import signupcollectioninfo
from ..config.VehicleDatabase import collection 
from typing import List, Dict, Any
from fastapi import Depends
from ..config.admindatabase import Vehiclecollection
from ..controllers.userSignupControllers import get_current_user  




from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter




import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler




router = APIRouter()
# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# @router.get('/')
# async def home():
#     # send_simple_message()
#     return {'msg': 'Welcome in my Recommended routes'}


@router.get("/trending-vehicles", tags=["Recommendation"])
async def get_trending_vehicles():
    # Fetch all users' uniqueVisitedModels
    cursor = signupcollectioninfo.find({}, {"statistics.uniqueVisitedModels": 1})

    all_visited_models = []

    # Step 1: Collect all uniqueVisitedModels from users' statistics
    async for user in cursor:
        unique_visited_models = user.get("statistics", {}).get("uniqueVisitedModels", [])
        all_visited_models.extend(unique_visited_models)

    if not all_visited_models:
        return {"message": "No models found"}

    # Step 2: Count the frequency of each model and find the top 5 most visited models
    model_counts = {}
    for model in all_visited_models:
        model_counts[model] = model_counts.get(model, 0) + 1

     # Log each model count on a new line
    logging.info("Model counts are:")
    for model, count in model_counts.items():
        logging.info(f"{model}: {count}")
    
    # Sort models by frequency and get the top 5
    top_5_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:4]

    # Step 3: Fetch details of these top 5 models from the Vehicles collection
    trending_vehicles = []
    for model, _ in top_5_models:
        # Split model name to extract the brand name and model name
        brand_name, model_name = model.split("_", 1)  # Assuming model name is in the format 'Brand_Model'

        # Search for the model in the Vehicles collection under the corresponding brand
        vehicle = await collection.find_one(
            {"brandName": brand_name, "models.modelName": model_name},
            {"models.$": 1, "brandName": 1}  # Fetch only the model and brand name
        )

        if vehicle:
            # Find the model details within the brand
            model_details = next((m for m in vehicle["models"] if m["modelName"] == model_name), None)
            launch_price = model_details.get("launchPrice")  # Get the launchPrice
            
            if model_details:
                # Get the first image from the images array
                first_image = model_details.get("images", [])[0] if model_details.get("images") else None

                trending_vehicles.append({
                    "brandName": vehicle["brandName"],
                    "modelName": model_name,
                    "image": first_image,  # Add the first image of the model
                    "launchPrice": launch_price  # Add the launch price
                })

    return {"trendingVehicles": trending_vehicles}









# Helper function to calculate score based on user's statistics
def calculate_behavior_score(vehicle_info: Dict, user_stats: Dict) -> float:
    score = 0

    # Safe access to 'brandName', in case it's missing in the vehicle_info
    brand_name = vehicle_info.get('brandName', '')
    if brand_name:
        brand_visited = user_stats.get('brandVisited', {})
        brand_score = brand_visited.get(brand_name, 0)
        score += brand_score * 0.2  # Weighting this factor

    # Score based on vehicle type frequency
    vehicle_type = vehicle_info.get('vehicleType', '')
    if vehicle_type:
        vehicle_types_visited = user_stats.get('VehicleTypesVisited', {})
        vehicle_type_score = vehicle_types_visited.get(vehicle_type, 0)
        score += vehicle_type_score * 0.2  # Weighting this factor

    # Score based on engine type frequency
    engine_type = vehicle_info.get('engineType', '')
    if engine_type:
        engine_types_visited = user_stats.get('EngineTypesVisited', {})
        engine_type_score = engine_types_visited.get(engine_type, 0)
        score += engine_type_score * 0.2  # Weighting this factor

    # Score based on price range
    price_range = vehicle_info.get('launchPrice', 0)
    average_price = user_stats.get('averagePrice', 0)
    if average_price > 0 and price_range > 0:
        price_score = 1 - abs(price_range - average_price) / average_price
        score += price_score * 0.3

    # Score based on horsepower
    horsepower_range = vehicle_info.get('horsepower', 0)
    average_horsepower = user_stats.get('averageHorsepower', 0)
    if average_horsepower > 0 and horsepower_range > 0:
        horsepower_score = 1 - abs(horsepower_range - average_horsepower) / average_horsepower
        score += horsepower_score * 0.05

    # Score based on torque
    torque_range = vehicle_info.get('torque', 0)
    average_torque = user_stats.get('averageTorque', 0)
    if average_torque > 0 and torque_range > 0:
        torque_score = 1 - abs(torque_range - average_torque) / average_torque
        score += torque_score * 0.05

    return score



# Route to get vehicle recommendations based on user behavior (click history)
@router.get("/recommendationsbybehavior", response_model=List[Dict[str, Any]], tags=["Recommendation"])
async def get_recommended_vehicles_based_on_behavior(current_user: str = Depends(get_current_user)):
    try:
        # Fetch user and statistics
        user = await signupcollectioninfo.find_one({"email": current_user})
        if not user or "statistics" not in user:
            raise HTTPException(status_code=404, detail="User not found or no statistics found")

        user_stats = user["statistics"]
        logger.info(f"User found: {user['email']} with statistics data.")

        # Fetch all vehicles and calculate behavior-based scores
        vehicles_cursor = Vehiclecollection.find()
        all_vehicles = await vehicles_cursor.to_list(length=None)

        vehicle_scores = []

        for vehicle in all_vehicles:
            for model in vehicle["models"]:
                # Check if the required fields exist
                if "launchPrice" in model and "vehicleType" in model and "engineType" in model:
                    behavior_score = calculate_behavior_score(model, user_stats)
                    vehicle_scores.append({
                        "brandName": vehicle.get("brandName", "Unknown Brand"),
                        "modelName": model.get("modelName", "Unknown Model"),
                        "launchPrice": model.get("launchPrice", 0),
                        "horsepower": model.get("horsepower", 0),
                        "torque": model.get("torque", 0),
                        "images": model.get("images", []),
                        "vehicleType": model.get("vehicleType", "Unknown Type"),
                        "engineType": model.get("engineType", "Unknown Engine"),
                        "finalScore": behavior_score,
                    })

                    # Print final score of each vehicle
                    print(f"Model: {model.get('modelName', 'Unknown Model')} | Final Score: {behavior_score}")

        # Sort vehicles by final score (highest first)
        vehicle_scores.sort(key=lambda x: x["finalScore"], reverse=True)

        # Select top 3 recommendations
        recommended_vehicles = vehicle_scores[:3]

        for recommendation in recommended_vehicles:
            logger.info(f"Recommended Vehicle: {recommendation['brandName']} {recommendation['modelName']}, "
                        f"Score: {recommendation['finalScore']}")

        return recommended_vehicles

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)










































# Define allowed categories
ALLOWED_VEHICLE_TYPES = ["SUV", "Sedan", "Compact", "Coupe", "Hatchback", "Pickup-Truck"]
ALLOWED_ENGINE_TYPES = ["Petrol", "Diesel", "Electric", "Hybrid"]
ALLOWED_COLORS = ["Black", "White", "Red", "Blue", "Gold", "Green", "Gray", "Brown", "Silver"]

# Helper function to calculate cosine similarity using sklearn (with normalization)
def cosine_similarity_func(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Returns cosine similarity between two normalized vectors"""
    vec1_normalized = normalize([vec1]) if np.linalg.norm(vec1) != 1 else vec1
    vec2_normalized = normalize([vec2]) if np.linalg.norm(vec2) != 1 else vec2
    return cosine_similarity(vec1_normalized, vec2_normalized)[0][0]

# Helper function to create user profile vector by averaging favorite vehicle vectors
def create_user_profile(user_favorites: List[Dict]) -> np.ndarray:
    profile_vector = None
    for favorite in user_favorites:
        try:
            vehicle_vector = np.array(favorite['vector'])
            if profile_vector is None:
                profile_vector = vehicle_vector
            else:
                profile_vector += vehicle_vector
        except Exception as e:
            raise ValueError(f"Error processing favorite vehicle vector: {favorite['brandName']} {favorite['modelName']} - {str(e)}")
    
    if profile_vector is not None and len(user_favorites) > 0:
        profile_vector /= len(user_favorites)
    
    return profile_vector

# Safe average calculation for handling missing values or extreme values
def safe_avg(attribute_name, user_favorites: List[Dict], clip_min=0, clip_max=100000):
    values = [min(max(fav.get(attribute_name, 0), clip_min), clip_max) for fav in user_favorites]
    return sum(values) / len(values) if values else 0

# Function to calculate attribute match score between user favorites and a vehicle
def calculate_attribute_match(user_favorites: List[Dict], vehicle_info: Dict) -> float:
    attribute_score = 0
    
    def attribute_range(attribute_name, user_favorites: List[Dict]):
        values = [fav.get(attribute_name, 0) for fav in user_favorites]
        return max(values) - min(values) if len(values) > 1 else 1  # Avoid division by zero
    
    # Average of the user's favorite attributes
    avg_launch_price = safe_avg('launchPrice', user_favorites)
    avg_horsepower = safe_avg('horsepower', user_favorites)
    avg_torque = safe_avg('torque', user_favorites)
    avg_seating_capacity = safe_avg('seatingCapacity', user_favorites)
    avg_year = safe_avg('year', user_favorites)
    avg_vehicle_type = safe_avg('vehicleType', user_favorites)
    avg_engine_type = safe_avg('engineType', user_favorites)
    avg_color = safe_avg('color', user_favorites)
    
    # Calculate the ranges for these attributes to compare how spread out the user's preferences are
    launch_price_range = attribute_range('launchPrice', user_favorites)
    horsepower_range = attribute_range('horsepower', user_favorites)
    torque_range = attribute_range('torque', user_favorites)
    seating_capacity_range = attribute_range('seatingCapacity', user_favorites)
    
    # Weigh each attribute based on importance
    weight_factors = {
        'launchPrice': 0.25,
        'vehicleType': 0.15,
        'engineType': 0.15,
        'year': 0.15,
        'colors': 0.10,
        'seatingCapacity': 0.10,
        'horsepower': 0.05,
        'torque': 0.05
    }

    if launch_price_range > 0:
        attribute_score += weight_factors['launchPrice'] * (1 - abs(vehicle_info['launchPrice'] - avg_launch_price) / launch_price_range)
    if horsepower_range > 0:
        attribute_score += weight_factors['horsepower'] * (1 - abs(vehicle_info['horsepower'] - avg_horsepower) / horsepower_range)
    if torque_range > 0:
        attribute_score += weight_factors['torque'] * (1 - abs(vehicle_info['torque'] - avg_torque) / torque_range)
    if seating_capacity_range > 0:
        attribute_score += weight_factors['seatingCapacity'] * (1 - abs(vehicle_info['seatingCapacity'] - avg_seating_capacity) / seating_capacity_range)
    
    # Handle vehicle type, engine type, and color with direct matching
    if vehicle_info['vehicleType'] == avg_vehicle_type:
        attribute_score += weight_factors['vehicleType']
    if vehicle_info['engineType'] == avg_engine_type:
        attribute_score += weight_factors['engineType']
    if vehicle_info['colors'] == avg_color:
        attribute_score += weight_factors['colors']
    
    # Handle the 'year' comparison
    if avg_year > 0:
        attribute_score += weight_factors['year'] * (1 - abs(vehicle_info['year'] - avg_year) / avg_year)
    
    return attribute_score


# Function to calculate final score combining cosine similarity and attribute match score
def calculate_final_score(cosine_sim: float, attribute_match_score: float, num_favorites: int) -> float:
    cosine_weight = min(0.7 + (0.3 * (num_favorites / 10)), 0.9)  # Weight adjustment based on the number of favorites
    attribute_weight = 1 - cosine_weight
    return (cosine_weight * cosine_sim) + (attribute_weight * attribute_match_score)

# Helper function to get top N most common types from favorites
def get_top_common_types(user_favorites: List[Dict], top_n: int = 3) -> List[str]:
    vehicle_types = [fav.get('vehicleType', '') for fav in user_favorites]
    engine_types = [fav.get('engineType', '') for fav in user_favorites]
    colors = [fav.get('colors', '') for fav in user_favorites]
    
    # Count most common vehicle types, engine types, and colors
    common_vehicle_types = [item[0] for item in Counter(vehicle_types).most_common(top_n)]
    common_engine_types = [item[0] for item in Counter(engine_types).most_common(top_n)]
    common_colors = [item[0] for item in Counter(colors).most_common(top_n)]
    
    return common_vehicle_types, common_engine_types, common_colors


@router.get("/recommendationsbyfavorites", response_model=List[Dict[str, Any]] , tags=["Recommendation"])
async def get_recommended_vehicles(current_user: str = Depends(get_current_user)):
    try:
        # Fetch user and favorites
        user = await signupcollectioninfo.find_one({"email": current_user})
        if not user or "favorites" not in user:
            raise HTTPException(status_code=404, detail="User not found or no favorites found")
        
        user_favorites = user["favorites"]
        logger.info(f"User found: {user['email']} with {len(user_favorites)} favorites.")
        
        # Get the top N most common vehicle and engine types and colors from favorites
        top_vehicle_types, top_engine_types, top_colors = get_top_common_types(user_favorites)

        # Fetch vehicle vectors for each favorite
        favorite_vehicle_vectors = []
        for favorite in user_favorites:
            vehicle = await Vehiclecollection.find_one({
                "brandName": favorite["brandName"], 
                "models.modelName": favorite["modelName"]
            })
            if vehicle:
                model = next((m for m in vehicle["models"] if m["modelName"] == favorite["modelName"]), None)
                if model:
                    favorite_vehicle_vectors.append({
                        "brandName": favorite["brandName"],
                        "modelName": favorite["modelName"],
                        "vector": model["vector"],
                        "launchPrice": model.get("launchPrice", 0),
                        "horsepower": model.get("horsepower", 0),
                        "torque": model.get("torque", 0),
                        "seatingCapacity": model.get("seatingCapacity", 0),
                        "year": model.get("year", 0),
                        "vehicleType": model.get("vehicleType", ""),
                        "engineType": model.get("engineType", ""),
                        "colors": model.get("colors", "")
                    })
                    logger.info(f"Processed vehicle {favorite['brandName']} {favorite['modelName']}.")
                else:
                    logger.warning(f"Model not found for {favorite['brandName']} {favorite['modelName']}.")
        
        # Create user profile vector (average of favorite vectors)
        user_profile_vector = create_user_profile(favorite_vehicle_vectors)
        
        # Fetch all vehicles and calculate cosine similarities
        vehicles_cursor = Vehiclecollection.find()
        all_vehicles = await vehicles_cursor.to_list(length=None)
        
        vehicle_scores = []
        
        for vehicle in all_vehicles:
            for model in vehicle["models"]:
                if "vector" in model:
                    vehicle_vector = np.array(model["vector"])
                    cosine_sim = cosine_similarity_func(user_profile_vector, vehicle_vector)
                    
                    # Calculate the attribute match score
                    attribute_match_score = calculate_attribute_match(user_favorites, model)
                    
                    # Combine cosine similarity and attribute match score
                    final_score = calculate_final_score(cosine_sim, attribute_match_score, len(user_favorites))
                    
                    # Check if vehicle matches top vehicle types or engine types or color
                    if model['vehicleType'] not in top_vehicle_types or model['engineType'] not in top_engine_types or model['colors'] not in top_colors:
                        final_score -= 0.1  # Slight penalty for not matching top types
                    
                    vehicle_scores.append({
                        "brandName": vehicle["brandName"],
                        "modelName": model["modelName"],
                        "launchPrice": model.get("launchPrice", 0), 
                        "images": model.get("images", []), 
                        "finalScore": final_score,
                        "cosineSimilarity": cosine_sim,
                        "attributeMatchScore": attribute_match_score
                    })
        
        # Sort vehicles by final score (highest first)
        vehicle_scores.sort(key=lambda x: x["finalScore"], reverse=True)
        
        # Select top 3 recommendations
        recommended_vehicles = vehicle_scores[:3]
        
        for recommendation in recommended_vehicles:
            logger.info(f"Recommended Vehicle: {recommendation['brandName']} {recommendation['modelName']}, "
                        f"Cosine Similarity: {recommendation['cosineSimilarity']}, "
                        f"Attribute Match Score: {recommendation['attributeMatchScore']}")
        
        return recommended_vehicles
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logger.error(error_message)
        raise HTTPException(status_code=500, detail=error_message)
































# Set Gensim's logger to only show WARNING level logs and higher
logging.getLogger('gensim').setLevel(logging.WARNING)


# Initialize Word2Vec and Scaler
word2vec_model = Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
scaler = MinMaxScaler()


# Helper function to preprocess and vectorize text data
def vectorize_text(text: str) -> np.ndarray:
    if not text:  # Check if the text is empty or None
        logger.debug(f"Empty text encountered, returning zero vector for text: {text}")
        return np.zeros(word2vec_model.vector_size)
    
    words = text.split()  # Basic tokenization (use more advanced tokenization if needed)
    
    # Check for words in the Word2Vec model
    valid_vectors = []
    for word in words:
        if word in word2vec_model.wv:
            valid_vectors.append(word2vec_model.wv[word])
    
    # If no valid words were found, we should return a random vector or a non-zero vector
    if not valid_vectors:  # If no valid words found, return a random non-zero vector
        logger.debug(f"No valid words found in Word2Vec model for text: {text}. Returning random vector.")
        return np.random.normal(size=word2vec_model.vector_size)  # Random vector as fallback
    
    # Compute the mean of the valid word vectors
    vector = np.mean(valid_vectors, axis=0)
    
    if np.any(np.isnan(vector)):  # Check if the vector contains NaN values
        logger.debug(f"Vectorization resulted in NaN for text: {text}")
        return np.zeros(word2vec_model.vector_size)
    return vector

def normalize_numerical_data(data: List[float]) -> np.ndarray:
    # Handle empty numerical data by filling with default values (e.g., 0)
    if not data or all(v is None for v in data):
        logger.debug(f"Empty or None numerical data encountered, returning zero vector: {data}")
        return np.zeros(len(data))  # Return a zero vector of appropriate length
    data = [d if d is not None else 0 for d in data]
    normalized = scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()
    return normalized

@router.get("/vehicles/vectorize" , tags=["Recommendation"])
async def vectorize_vehicles():
    try:
        # Fetch all vehicles from the database
        vehicles_cursor = Vehiclecollection.find()  # Get cursor for all vehicles
        vehicles = await vehicles_cursor.to_list(length=None)  # Fetch all documents
        
        # Initialize a list to store updated vehicles
        updated_vehicles = []

        # Iterate through each vehicle document
        for vehicle in vehicles:
            for model in vehicle.get("models", []):
                # Vectorize textual data, ensuring defaults for missing or empty fields
                textual_features = [
                    model.get("modelName", ""),  # Default empty string if modelName is missing
                    model.get("vehicleType", ""),  # Default empty string if vehicleType is missing
                    model.get("engineType", ""),  # Default empty string if engineType is missing
                    model.get("description", ""),  # Default empty string if description is missing
                    " ".join(model.get("variants", [])),  # Default empty string if variants is missing or empty
                    " ".join(model.get("colors", []))  # Default empty string if colors is missing or empty
                ]

                # Vectorize all textual fields using Word2Vec
                textual_vectors = [vectorize_text(text) for text in textual_features]

                # Check if any textual vector is zero-dimensional
                if any(vec.shape == () for vec in textual_vectors):
                    logger.warning(f"Zero-dimensional vector found in textual data for model: {model['modelName']}")
                
                # Concatenate all textual feature vectors
                combined_textual_vector = np.concatenate(textual_vectors)

                # Handle missing numerical data by filling with defaults (0)
                numerical_features = [
                    model.get("torque", 0),  # Default 0 if torque is missing
                    model.get("year", 0),  # Default 0 if year is missing
                    model.get("launchPrice", 0),  # Default 0 if launchPrice is missing
                    model.get("horsepower", 0),  # Default 0 if horsepower is missing
                    model.get("seatingCapacity", 0)  # Default 0 if seatingCapacity is missing
                ]
                # Normalize numerical data
                normalized_numerical_vector = normalize_numerical_data(numerical_features)

                # Combine textual and numerical vectors
                full_vector = np.concatenate([combined_textual_vector, normalized_numerical_vector])

                # Check if the full vector is a valid shape
                if full_vector.shape == ():
                    logger.warning(f"Zero-dimensional vector generated for model: {model['modelName']}")

                # Store the vectorized data in the model
                model["vector"] = full_vector.tolist()

            # Update the vehicle document in the database with the new vectorized data
            await Vehiclecollection.update_one(
                {"_id": vehicle["_id"]},
                {"$set": {"models": vehicle["models"]}}  # Update models with the vectorized data
            )

            updated_vehicles.append(vehicle["brandName"])

        return {"status": "success", "message": "Vehicle vectorization completed", "updated_vehicles": updated_vehicles}
    
    except Exception as e:
        logger.error(f"Error in vectorizing vehicles: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during vehicle vectorization")


