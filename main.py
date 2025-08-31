from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from pipeline import pipeline
app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-pdf/")
async def upload_pdf(
    token: str = Form(...),  # Token string passed in form-data
    file: UploadFile = File(...)
):
    # Optional: validate token
    if token != "my-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")

    # Save PDF file
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse(
        content={
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "token": token
        }
    )
