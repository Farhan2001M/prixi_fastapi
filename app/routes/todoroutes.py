from fastapi import APIRouter
router = APIRouter()

from fastapi import HTTPException
from ..models.todomodel import Todo
from ..controllers.todocontrollers import get_all_todos, fetch_one_todo, create_todo, update_todo, remove_todo



@router.get("/")
def read_root():
    return {"Ping" : "Pong"}


#READ_All_TODOS
@router.get("/api/todos" , tags=["Read"])
async def read_todo():
    res = await get_all_todos()
    return res


#READ_TODO_by_ID
@router.get("/api/todos{title}" , tags=["Read"] , response_model=Todo)
async def read_todo_by_id(title):
    res = await fetch_one_todo(title)
    if res:
        return res
    raise HTTPException(404, f"There is no todo item with this title {title}")


#Post_TODO
@router.post("/api/todos",  tags=["Read"] , response_model=Todo)
async def post_todo(todo:Todo):
    res = await create_todo(todo.dict())
    if res:
        return res
    raise HTTPException(400 , "Something went wrong / Bad Request")


#Update_TODO
@router.put("/api/todos{title}/" , tags=["Update"] , response_model=Todo)
async def put_todo(title:str , description:str):
    res = await update_todo(title , description)
    if res: 
        return res
    raise HTTPException(404, f"There is no todo item with this title {title}")


#Delete_TODO
@router.delete("/api/todos/{title}" , tags=["Update"])
async def delete_todo(title):
    res = await remove_todo(title)
    if res:
        return "Succesfully deleted todo item .! "
    raise HTTPException(404, f"There is no todo item with this title {title}")


# # User sign-up route
# @router.post("/api/signup", tags=["User"])
# async def signup_user(user: User):
#     res = await create_user(user)
#     if res:
#         return {"message": "User successfully signed up", "user": res}
#     raise HTTPException(400, "User with this email already exists or something went wrong")
