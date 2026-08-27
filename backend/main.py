from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

# Import your new functions
from ai_pipeline import process_graph, process_pdf, process_audio

app = FastAPI(title="Local Graph AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary folder to save uploaded files
os.makedirs("temp_uploads", exist_ok=True)

@app.get("/api/network")
def get_network():
    return process_graph()

@app.post("/api/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a police FIR PDF to be processed by the AI."""
    file_path = f"temp_uploads/{file.filename}"
    
    # Save the uploaded file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Send it to the AI pipeline
    extracted_text = process_pdf(file_path)
    
    return {"message": "PDF Processed", "extracted_text": extracted_text}

@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload a wiretap audio file to be transcribed and processed."""
    file_path = f"temp_uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    extracted_text = process_audio(file_path)
    
    return {"message": "Audio Transcribed", "extracted_text": extracted_text}