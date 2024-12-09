import motor.motor_asyncio

# MongoDB Atlas connection string
atlas_uri = "mongodb+srv://chfarhanilyas550:farhan123@prixi.4pixy.mongodb.net/"

# Create an AsyncIOMotorClient instance to connect to Atlas
client = motor.motor_asyncio.AsyncIOMotorClient(atlas_uri)

# Access the 'PrixiDB' database
usersDB = client.prixidb

# Access the necessary collections
Vehiclecollection = usersDB.vehiclesinfo  # Vehicles collection
VehicleData = usersDB.vehicledatainfo   # For 2nd module
adminlogininfo = usersDB.admininfo      # Admin login info collection
