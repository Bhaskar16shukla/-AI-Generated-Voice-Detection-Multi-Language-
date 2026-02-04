# AI-Generated Voice Detection (Multi-Language)

API-based system that determines whether a voice sample is **AI-generated** or **human-generated**. Supports Tamil, English, Hindi, Malayalam, and Telugu.

## Features

- Accepts Base64-encoded MP3 (and WebM/WAV for recording)
- Returns structured JSON: classification, confidence score, explanation
- Multi-language support: Tamil, English, Hindi, Malayalam, Telugu
- Web UI for upload/record and testing

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** For WebM recording support, install [FFmpeg](https://ffmpeg.org/). On Windows: `winget install ffmpeg` or download from ffmpeg.org.

### 2. Run the server

```bash
uvicorn api:app --reload --host 0.0.0.0
```

### 3. Open in browser

Go to **http://localhost:8000** for the web UI.

## API Usage

### POST /detect

**Request body (JSON):**

```json
{
  "audio_base64": "<base64-encoded-mp3-string>",
  "language": "English"
}
```

- `audio_base64` (required): Base64-encoded MP3 audio
- `language` (optional): One of Tamil, English, Hindi, Malayalam, Telugu

**Response:**

```json
{
  "classification": "AI-generated",
  "confidence_score": 0.9234,
  "explanation": "The audio sample shows strong characteristics of AI-generated speech synthesis...",
  "raw_scores": {
    "ai_generated": 0.9234,
    "human": 0.0766
  },
  "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
}
```

### cURL example

```bash
# Encode MP3 to base64 and call API (PowerShell)
$bytes = [System.IO.File]::ReadAllBytes("sample.mp3")
$b64 = [Convert]::ToBase64String($bytes)
Invoke-RestMethod -Uri "http://localhost:8000/detect" -Method POST -ContentType "application/json" -Body '{"audio_base64":"' + $b64 + '","language":"English"}'
```

## Files

| File | Description |
|------|-------------|
| `api.py` | FastAPI backend with detection logic |
| `index.html` | Web UI (upload, record, results) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Model

Uses [MelodyMachine/Deepfake-audio-detection-V2](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) (Wav2Vec2-based, 99.7% accuracy). First run downloads the model (~95MB).
