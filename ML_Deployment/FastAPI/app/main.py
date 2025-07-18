from fastapi import FastAPI
from pydantic import BaseModel
from model.model import __version__ as model_version
from model.model import predict_pipeline
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI()

# Serve static files from the "static" folder
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# FastAPI uses Pydantic models to define and validate request and response
class TextIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    language: str

@app.get("/")
def home():
    # return {"health_check": "OK"}#, "model_version":model_version}
    # Serve the index.html file
    with open("app/static/index.html", "r") as file:
                return HTMLResponse(content=file.read(), status_code=200)

@app.post("/predict", response_model=PredictOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"language": language}