
# import uvicorn
# HOST = '127.0.0.1'


# if __name__== '__main__':
#     uvicorn.run('app.api:app', host = HOST, port = 8000, reload = True)



import uvicorn

PORT = 8000
HOST = '0.0.0.0'

if __name__ == '__main__':
    uvicorn.run('app.api:app', host = HOST, port = PORT, reload = True)