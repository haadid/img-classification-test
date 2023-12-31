from PIL import Image
from predict import read_image
from predict import predictor
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}
    

@app.post("/uploadfile/")
async def create_upload_file(file: bytes = File(...)):

    image = read_image(file)
    prediction = predictor(image)

    return prediction
