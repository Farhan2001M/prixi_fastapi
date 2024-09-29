#MongoDB Driver
import motor.motor_asyncio

client = motor.motor_asyncio.AsyncIOMotorClient('mongodb://localhost:27017')
usersDB = client.PrixiDB

#Vehicles collection
Vehiclecollection = usersDB.Vehicles



# For 2nd module 

VehicleData = usersDB.VehicleData



adminlogininfo = usersDB.AdminInfo
