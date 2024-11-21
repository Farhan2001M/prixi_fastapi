
import uvicorn

PORT = 8000
HOST = '127.0.0.1'

if __name__ == '__main__':
    uvicorn.run('app.api:app', host = HOST, port = PORT, reload = True)