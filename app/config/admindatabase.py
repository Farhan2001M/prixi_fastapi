import motor.motor_asyncio

# Use your connection string from MongoDB Atlas
client = motor.motor_asyncio.AsyncIOMotorClient('mongodb+srv://chfarhanilyas550:farhan123@farhan0.k7f9z.mongodb.net/Prixi?retryWrites=true&w=majority')
# client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017')

# Access the 'Prixi' database
usersDB = client.PrixiDB

# Access the 'AdminInfo' collection
adminlogininfo = usersDB.AdminInfo

#Vehicles collection
Vehiclecollection = usersDB.VehiclesData

# For 2nd module 
VehicleData = usersDB.VehicleData
