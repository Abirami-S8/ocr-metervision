# MeterVision QC — Smart Meter OCR Pipeline
 
> **INSTINCT 4.0 Hackathon | Sub-Challenge A — Backend OCR**  
> Reliable, privacy-preserving smart meter reading extraction. Fully containerized. VPC-ready.
 
---
 
## Team Information
 
**Team Name:** Coffee and Code
 
**Team Members:**
| Name | Role | Institution |
|------|------|-------------|
| Monashini S | Team Leader | Sri Sairam Engineering College |
| Abirami S | Member | Sri Sairam Engineering College |
| Sankari RV | Member | Sri Sairam Engineering College |
 
**Department:** Computer Science and Engineering (AIML) — 3rd Year
 
---
 
## Problem Statement
 
**Accurate Meter Reading via OCR: Develop a solution for reliable meter reading extraction from field photographs.**
 
During smart meter replacement, field teams capture still photos of meter displays and nameplates. QC staff currently read these manually, which is slow and error-prone due to tilt, glare, dust, low light, and unclear decimal points. The goal is a highly reliable, privacy-preserving OCR system for smart-meter photos at the Quality Control (QC) Stage.
 
---
 
## Project Overview
 
**MeterVision QC** is an AI-powered Optical Character Recognition (OCR) system designed for quality control of smart meter replacement field photographs. The system addresses the critical problem of slow, error-prone manual meter reading at the QC stage by delivering automated, high-accuracy extraction of five mandatory fields: Meter Serial Number, kWh, kVAh, Maximum Demand (kW), and Demand (kVA).
 
The pipeline operates in three stages. First, an OpenCV-based preprocessing module performs adaptive image enhancement (CLAHE + Non-Local Means denoising + unsharp masking), skew/tilt correction using Hough line transforms, and perspective dewarping via homography to handle real-world photo conditions including glare, tilt, dust, and low light — covering all four dataset categories: Day Light, Night, Tilted, and Blurred. Second, a dual-model OCR engine uses **EasyOCR** (CRAFT text detector + CRNN recognizer) as the primary extraction model and **TrOCR** (`microsoft/trocr-base-printed`) — a transformer vision encoder–decoder specialized for printed text on LCD/segment displays — as the secondary model. **Qwen2-0.5B-Instruct** serves as the LLM-based field parser for decimal correction, OCR error repair (O↔0, l↔1, S↔5, B↔8), and structured JSON output, a validation layer applies per-field regex rules including range checks, decimal placement checks, and character-level sanity, producing per-field Pass/Fail verdicts with reason codes and calibrated confidence scores.
 
The system is fully containerized via Docker and docker-compose, deployable in any VPC. It exposes a FastAPI REST interface for both single-image and batch (up to 100 images) processing, returning structured JSON/CSV with field values, per-field probabilities, image quality flags (blur, glare, contrast, tilt), and Pass/Fail codes. A Streamlit-based industrial QC dashboard provides real-time upload, per-field result visualization, batch overview with confidence charts, and one-click JSON/CSV export. All processing stays on-premises — no public cloud OCR is used.
 
---
 
## Used Technologies
 
| Category | Technology |
|----------|------------|
| OCR Engine | EasyOCR (CRAFT + CRNN) |
| Transformer OCR | TrOCR (`microsoft/trocr-base-printed`) |
| LLM Field Parser | Qwen2-0.5B-Instruct |
| Image Processing | OpenCV, scikit-image, Pillow |
| Backend API | FastAPI, Uvicorn |
| QC Dashboard | HTML,CSS,JS |
| Deployment | Docker, Docker Compose |
| Language | Python 3.10.11 |
| Configuration | YAML |
| Testing | Pytest |
 
---
 
## Architecture
 
```
┌──────────────────────────────────────────────────────────────┐
│                        VPC / Docker                          │
│                                                              │
│  ┌──────────────┐    ┌────────────────────────────────────┐  │
│  │  QC Dashboard│───▶│  FastAPI Backend (:8000)          │   │
│  │  HTML        │    │                                    │  │
│  │  :8000       │    │  /extract        (POST)            │  │
│  └──────────────┘    │  /batch          (POST)            │  │
│                      │  /results        (GET)             │  │
│                      │  /health         (GET)             │  │
│                      └──────────┬─────────────────────────┘  │
│                                 │                            │
│                      ┌──────────▼──────────────┐             │
│                      │     OCR Pipeline         │            │
│                      │                          │            │
│                      │  1. Quality Check        │            │
│                      │     Blur detection       │            │
│                      │     Brightness check     │            │
│                      │     Tilt estimation      │            │
│                      │                          │            │
│                      │  2. Preprocessing        │            │
│                      │     CLAHE enhancement    │            │
│                      │     NLM denoising        │            │
│                      │     Perspective dewarp   │            │
│                      │     LCD region crop      │            │
│                      │                          │            │
│                      │  3. OCR Ensemble         │            │
│                      │     EasyOCR (primary)    │            │
│                      │     TrOCR (secondary)    │            │
│                      │     Confidence voting    │            │
│                      │                          │            │
│                      │  4. LLM Field Parser     │            │
│                      │     Qwen2-0.5B-Instruct  │            │
│                      │     Decimal correction   │            │
│                      │     OCR error repair     │            │
│                      │                          │            │
│                      │                          │            │
│                      │  5. Validation           │            │
│                      │     Per-field rules      │            │
│                      │     Confidence scoring   │            │
│                      │     Pass/Fail verdict    │            │
│                      └──────────────────────────┘            │
│                                                              │
│  Model Cache (HuggingFace): /root/.cache/huggingface         │
└──────────────────────────────────────────────────────────────┘
```
 
---
 
## Models Used
 
| Model | Role | Source |
|-------|------|--------|
| `EasyOCR` (CRAFT + CRNN) | Primary OCR — text detection + recognition | PyPI |
| `microsoft/trocr-base-printed` | Secondary OCR — LCD/printed text extraction | HuggingFace |
| `Qwen/Qwen2-0.5B-Instruct` | LLM field parsing, decimal correction, OCR error repair | HuggingFace |

 
 
---
 
## Five Extracted Fields
 
1. **serial_number** — Meter serial / identification number
2. **kwh** — Active energy in kWh (with decimal, e.g. `002090.3`)
3. **kvah** — Apparent energy in kVAh (with decimal)
4. **md_kw** — Maximum Demand in kW (with decimal)
5. **demand_kva** — Demand in kVA (with decimal)
 
---
 
## Implemented Features
 
1. **Dual-Engine OCR Ensemble (EasyOCR + TrOCR)**
   - Runs both engines in parallel and merges results via confidence voting
   - Handles LCD segment displays, printed nameplates, and degraded images
 
2. **Adaptive Image Preprocessing**
   - CLAHE contrast enhancement, NLM denoising, unsharp masking
   - Perspective dewarping via homography for tilted photos
   - Automatic LCD display region detection and cropping
 
3. **LLM-based Field Correction (Qwen2-0.5B-Instruct)**
   - Fixes common OCR character confusions: O↔0, l↔1, S↔5, B↔8, Z↔2
   - Corrects decimal placement for kWh/kVAh readings
   - Returns structured JSON with per-field confidence scores
 
4. **Image Quality Assessment**
   - Detects blur (Laplacian variance), darkness, overexposure, tilt, low resolution
   - Returns quality flags and composite quality score (0.0–1.0)
 
5. **Per-Field Validation with Pass/Fail**
   - Regex pattern validation for all 5 fields
   - Confidence thresholding: PASS / WARN / FAIL / MISSING
   - Reason codes for every failed field
 
6. **FastAPI REST Interface**
   - Single image endpoint (`/extract`) and batch endpoint (`/batch`)
   - Swagger UI at `/docs` for easy testing
   - JSON and CSV output formats
 
7. **QC Dashboard**
   - Real-time single image upload and result visualization
   - Batch results viewer with pass rate charts
   - Per-field confidence bar charts
   - One-click CSV download
 
8. **Batch Processing CLI**
   - Process entire dataset folders recursively
   - Parallel workers for throughput
   - Outputs `results.json` and `results.csv`
 
9. **Benchmark Evaluation Script**
   - Exact-value accuracy and character-level accuracy per field
   - Confidence calibration metrics
 
10. **Docker Container Deployment**
    - Fully containerized via Docker and docker-compose
    - Runs both API and Dashboard as separate services
    - VPC-ready, no public cloud OCR dependencies
 
---
 
## Dataset Structure
 
```
data/raw_images/
├── 1)Day Light/       ← clear daylight meter photos
├── 2)Night/           ← low light / dark conditions
├── 3)Tilted/          ← angled / skewed captures
└── 4)Blurred/         ← motion blur / out-of-focus
```
 
---
 
## Running the Project
 
### Prerequisites
- Python 3.10 with virtual environment (venv)
- 8GB RAM minimum (for model inference)
- Internet access for first-time model download
 
### Without Docker
 
```bash
# 1. Navigate to project root
cd meter_ocr
 
# 2. Activate virtual environment
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux
 
# 3. Install dependencies
pip install -r requirements.txt
 
# 4. Test single image
python scripts/demo.py --image data/raw_images/your_meter.jpg
 
# 5. Start API — Terminal 1
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
 
```
 
### With Docker
 
```bash
cd docker
docker-compose up --build
docker-compose up
```
 
| Service | URL |
|---------|-----|
| QC Dashboard | http://localhost:8000 |
| API Swagger UI | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
 
---
 
## API
 
| Endpoint | Method(s) | Description |
|----------|-----------|-------------|
| `/extract` | **POST** | Upload a single meter image and extract all 5 fields |
| `/batch` | **POST** | Trigger batch processing of up to 100 images from a directory |
| `/results` | **GET** | Download batch results as CSV |
| `/health` | **GET** | Health check — returns API status and version |
 
### POST /extract — Sample Response
```json
{
  "image_path": "meter_001.jpg",
  "overall_pass": true,
  "overall_confidence": 0.912,
  "quality_score": 0.85,
  "quality_flags": [],
  "processing_notes": [
    "Perspective correction applied",
    "Display region detected and cropped",
    "Image enhancement applied"
  ],
  "fields": {
    "serial_number": { "value": "TN123456",  "confidence": 0.97, "pass_fail": "PASS", "reason": "" },
    "kwh":           { "value": "002090.3",  "confidence": 0.96, "pass_fail": "PASS", "reason": "" },
    "kvah":          { "value": "002500.1",  "confidence": 0.94, "pass_fail": "PASS", "reason": "" },
    "md_kw":         { "value": "15.50",     "confidence": 0.89, "pass_fail": "PASS", "reason": "" },
    "demand_kva":    { "value": "18.20",     "confidence": 0.87, "pass_fail": "PASS", "reason": "" }
  }
}
```
 
### POST /batch — Sample Request
```json
{
  "input_dir": "data/raw_images",
  "output_dir": "data/processed",
  "output_format": "both",
  "workers": 4
}
```
 
---
 
## Output Codes
 
| Code | Meaning |
|------|---------|
| `PASS` | Field validated successfully with confidence ≥ 0.6 |
| `WARN` | Field pattern matched but confidence between 0.4–0.6 |
| `FAIL` | Field failed pattern validation or confidence < 0.4 |
| `MISSING` | Field not detected in image |
 
---
 
## Image Quality Flags
 
| Flag | Trigger |
|------|---------|
| `BLUR` | Laplacian variance < 100 |
| `DARK` | Mean brightness < 40 |
| `OVEREXPOSED` | Mean brightness > 220 |
| `TILT` | Estimated skew angle > 30° |
| `LOW_RES` | Image width < 400px or height < 300px |
 
---
 
## Project Structure
 
```
meter_ocr/
├── pipeline.py                   # Core OCR pipeline orchestrator
├── api/
│   ├── main.py                   # FastAPI application + all routes
│   └── schemas.py                # Pydantic request/response schemas
├── models/
│   ├── easyocr_engine.py         # EasyOCR wrapper (CRAFT + CRNN)
│   ├── trocr_engine.py           # TrOCR transformer wrapper
│   ├── llm_corrector.py          # Qwen2 / Mistral-1.3B LLM correction
│   └── ensemble.py               # Multi-engine voting + merging
├── preprocessing/
│   ├── quality_check.py          # Blur, brightness, tilt detection
│   ├── enhance.py                # CLAHE, NLM denoise, unsharp mask
│   └── dewarp.py                 # Perspective correction, deskew
├── postprocessing/
│   └── field_parser.py           # Field validation + Pass/Fail verdict
├── qc_interface/
│   └── index.html                # Streamlit QC dashboard
├── scripts/
│   ├── demo.py                   # Quick single-image demo CLI
│   ├── batch_process.py          # Batch processing CLI
│   └── evaluate.py               # Benchmark accuracy evaluation
├── config/
│   └── config.yaml               # Central configuration
├── docker/
│   ├── Dockerfile                # Container definition
│   └── docker-compose.yml        # Multi-service orchestration
├── tests/
│   └── test_pipeline.py          # Unit tests (pytest)
├── data/
│   ├── raw_images/               # Input meter photos
│   │   ├── 1)Day Light/
│   │   ├── 2)Night/
│   │   ├── 3)Tilted/
│   │   └── 4)Blurred/
│   ├── processed/                # OCR output JSON/CSV
│   └── benchmark/                # Labeled benchmark set
├── requirements.txt
└── README.md
```
 
---
 
## Batch CLI
 
```bash
# Process all images in dataset
python scripts/batch_process.py --input data/raw_images --output data/processed
 
# With format and workers
python scripts/batch_process.py \
  --input data/raw_images \
  --output data/processed \
  --format both \
  --workers 4
```
 
Output files:
- `data/processed/results.json` — full structured results per image
- `data/processed/results.csv` — flat table for Excel / reporting / audit
 
---
 
## Evaluation
 
```bash
# Run accuracy benchmark against labeled ground truth CSV
python scripts/evaluate.py \
  --benchmark data/benchmark/ground_truth.csv \
  --images data/benchmark/ \
  --output evaluation_report.json
```
 
Ground truth CSV format:
```
filename,serial_number,kwh,kvah,md_kw,demand_kva
meter_001.jpg,TN123456,002090.3,002500.1,15.50,18.20
```
 
Metrics computed:
- Exact-value accuracy per field (including decimal placement)
- Character-level accuracy per field
- Confidence calibration (correlation of confidence with correctness)
 
---
 
## Success Metrics Target
 
| Metric | Target | Approach |
|--------|--------|----------|
| Exact-value accuracy | ≥ 95% | EasyOCR + TrOCR ensemble + LLM correction |
| Confidence calibration | High correlation | Per-field confidence × quality score penalty |
| Throughput | ≥ 5 img/min (CPU) | Async FastAPI, parallel batch workers |
| Latency (single image) | < 15s/image | Preprocessing cached, models pre-loaded |
 
---
 
**MeterVision QC provides electricity utility field teams with a complete end-to-end OCR and QC pipeline designed to eliminate manual meter reading errors, increase throughput, and ensure audit-ready output — all running privately within a VPC using open-source components.**
 
