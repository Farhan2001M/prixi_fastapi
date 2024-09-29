from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware


from .routes.adminRoutes import router as admin_router
from .routes.userSignupRoutes import router as signup_router
from .routes.VehicleRoutes import router as vehicle_router


#App object
app = FastAPI()

origins = ["http://localhost:3000",
           "https://localhost:3000",
           "http://localhost:3001",
           "https://localhost:3001",
          ]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


app.include_router(admin_router)
app.include_router(signup_router)
app.include_router(vehicle_router)


@app.get("/")
async def read_root():
    return {"message": " Hello World..!"}



