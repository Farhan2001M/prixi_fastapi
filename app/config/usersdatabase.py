# MongoDB Driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb+srv://chfarhanilyas550:farhan123@farhan0.k7f9z.mongodb.net/Prixi?retryWrites=true&w=majority')
# client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017')

db = client.PrixiDB

signupcollectioninfo = db.Userinfo










# import motor.motor_asyncio

# # MongoDB Atlas connection string
# atlas_uri = "mongodb+srv://chfarhanilyas550:farhan123@prixi.4pixy.mongodb.net/"

# # Create an AsyncIOMotorClient instance
# client = motor.motor_asyncio.AsyncIOMotorClient(atlas_uri)

# # Access the 'PrixiDB' database
# usersDB = client.prixidb

# # Access the 'Userinfo' collection
# signupcollectioninfo = usersDB.userinfo

# # This function confirms the connection and lists collections
# async def confirm_connection():
#     try:
#         # List collections in the database to confirm the connection
#         collections = await usersDB.list_collection_names()
        
#         # Print collections to confirm connection
#         print(f"Collections in 'prixidb': {collections}")
        
#         # Try fetching a document to check if we can interact with the collection
#         sample_user = await signupcollectioninfo.find_one()
#         if sample_user:
#             print("Connection successful! Sample user fetched.")
#         else:
#             print("Connection successful, but no users found.")
    
#     except Exception as e:
#         print(f"Error during connection or fetching data: {e}")
