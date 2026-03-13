"""
api/main.py
FastAPI application — single-image and batch endpoints.
"""
from fastapi.staticfiles import StaticFiles
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import ExtractionResponse, BatchRequest, HealthResponse
from pipeline import process_image, process_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Meter OCR API",
    description="Hackathon Sub-Challenge A: Backend OCR for smart meter photos",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIZE_MB = 20


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/extract", response_model=ExtractionResponse,
          summary="Extract meter readings from a single photo")
async def extract_single(image: UploadFile = File(...)):
    """
    Upload a meter photo (JPG/PNG/BMP/TIFF) and get the 5 field values.
    
    Returns:
    - Per-field value, confidence, and Pass/Fail
    - Overall Pass/Fail with reasons
    - Image quality flags
    """
    # Validate file type
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type '{ext}' not supported. Use: {allowed}")

    # Read and size-check
    content = await image.read()
    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {MAX_SIZE_MB} MB)")

    try:
        result = process_image(content)
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(500, f"OCR processing error: {str(e)}")


@app.post("/batch", summary="Trigger batch processing of a directory")
async def batch_process(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Trigger async batch processing of all images in input_dir.
    Results are saved to output_dir as JSON + CSV.
    """
    def run_batch():
        summary = process_batch(
            request.input_dir,
            request.output_dir,
            output_format=request.output_format,
            workers=request.workers
        )
        logger.info(f"Batch complete: {summary}")

    background_tasks.add_task(run_batch)
    return {"message": "Batch processing started", "input_dir": request.input_dir,
            "output_dir": request.output_dir}


@app.get("/results", summary="Download batch results CSV")
async def get_results(output_dir: str = "data/processed"):
    csv_path = os.path.join(output_dir, "results.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(404, "No results CSV found. Run /batch first.")
    return FileResponse(csv_path, media_type="text/csv", filename="meter_results.csv")


# Mount static files LAST so API routes take priority
qc_interface_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qc_interface")
app.mount("/", StaticFiles(directory=qc_interface_path, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)