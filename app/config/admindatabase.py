#MongoDB Driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017')
usersDB = client.PrixiDB


#Vehicles collection
Vehiclecollection = usersDB.Vehicles


# For 2nd module 
VehicleData = usersDB.VehicleData

adminlogininfo = usersDB.AdminInfo







# import motor.motor_asyncio

# # MongoDB Atlas connection string
# atlas_uri = "mongodb+srv://chfarhanilyas550:farhan123@prixi.4pixy.mongodb.net/"

# # Create an AsyncIOMotorClient instance to connect to Atlas
# client = motor.motor_asyncio.AsyncIOMotorClient(atlas_uri)

# # Access the 'PrixiDB' database
# usersDB = client.prixidb

# # Access the necessary collections
# Vehiclecollection = usersDB.vehiclesinfo  # Vehicles collection
# VehicleData = usersDB.vehicledatainfo   # For 2nd module
# adminlogininfo = usersDB.admininfo      # Admin login info collection
