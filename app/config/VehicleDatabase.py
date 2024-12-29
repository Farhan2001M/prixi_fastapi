# MongoDB Driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb+srv://chfarhanilyas550:farhan123@farhan0.k7f9z.mongodb.net/Prixi?retryWrites=true&w=majority')
# client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017')

db = client.PrixiDB
collection = db.VehiclesData





# import motor.motor_asyncio

# # MongoDB Atlas connection string
# atlas_uri = "mongodb+srv://chfarhanilyas550:farhan123@prixi.4pixy.mongodb.net/"

# # Create an AsyncIOMotorClient instance to connect to Atlas
# client = motor.motor_asyncio.AsyncIOMotorClient(atlas_uri)

# # Access the 'PrixiDB' database
# db = client.prixidb

# # Access the 'Vehicles' collection
# collection = db.vehiclesinfo  # You can change this to the appropriate collection name
