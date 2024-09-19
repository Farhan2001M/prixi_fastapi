from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware

from .routes.VehicleRoutes import router

#App object
app = FastAPI()

origins = ["http://localhost:3000",
           "https://localhost:3000"
          ]


app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


app.include_router(router)



@app.get("/")
async def read_root():
    return {"message": " Hello World..!"}







# from .routes.todoroutes import router
# from .routes.userSignupRoutes import router


# from .routes.todoroutes import router as todo_router
# from .routes.userSignupRoutes import router as user_router





# Include routers
# app.include_router(todo_router, prefix="/tasks", tags=["tasks"])
# app.include_router(user_router, prefix="/users", tags=["users"])

# app.include_router(todo_router, prefix="/tasks", )
# app.include_router(user_router, prefix="/users", )
