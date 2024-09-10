import uvicorn
HOST = '0.0.0.0'

if __name__== '__main__':
    uvicorn.run('app.api:app', host = HOST, port = 8000,reload = True)
