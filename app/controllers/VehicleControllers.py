




# Function to generate a new ID for a brand
async def get_new_id(collection):
    count = await collection.count_documents({})
    return count + 1