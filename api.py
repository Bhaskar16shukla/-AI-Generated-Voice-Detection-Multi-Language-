"""
AI-Generated Voice Detection API (Multi-Language)
Supports: Tamil, English, Hindi, Malayalam, Telugu
Accepts Base64-encoded MP3 and returns JSON with classification, confidence, explanation
"""

import base64
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="AI Voice Detection API", description="Detects AI-generated vs human voice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VoiceDetectionRequest(BaseModel):
    audio_base64: str  # Base64-encoded MP3
    language: Optional[str] = None  # Tamil, English, Hindi, Malayalam, Telugu


# Load model on startup (lazy loading for faster startup)
classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        from transformers import pipeline
        classifier = pipeline(
            "audio-classification",
            model="MelodyMachine/Deepfake-audio-detection-V2",
            return_all_scores=True
        )
    return classifier


def decode_audio(base64_audio: str) -> tuple:
    """Decode Base64 MP3/WebM/WAV to audio array and sample rate."""
    try:
        if "," in base64_audio:
            parts = base64_audio.split(",", 1)
            mime = parts[0].lower() if len(parts) > 1 else ""
            base64_audio = parts[1] if len(parts) > 1 else base64_audio
        else:
            mime = ""
        audio_bytes = base64.b64decode(base64_audio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 encoding: {str(e)}")

    import torchaudio
    import torch

    suffix = ".mp3"
    if "webm" in mime or audio_bytes[:4] == b"\x1aE\xdf\xa3":
        suffix = ".webm"
    elif "wav" in mime or (len(audio_bytes) >= 4 and audio_bytes[:4] == b"RIFF"):
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        try:
            waveform, sample_rate = torchaudio.load(tmp_path)
        except Exception:
            import librosa
            waveform_np, sample_rate = librosa.load(tmp_path, sr=None, mono=True)
            waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        return waveform.squeeze().numpy(), int(sample_rate)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio format. Please provide valid MP3/WebM/WAV: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def get_explanation(classification: str, confidence: float, language: Optional[str]) -> str:
    """Generate human-readable explanation based on classification and language context."""
    langs = {"tamil": "Tamil", "english": "English", "hindi": "Hindi", "malayalam": "Malayalam", "telugu": "Telugu"}
    lang_name = langs.get(language.lower(), "multilingual") if language else "multilingual"

    if classification == "AI-generated":
        if confidence >= 0.9:
            return f"The audio sample (detected as {lang_name}) shows strong characteristics of AI-generated speech synthesis. The high confidence score indicates acoustic artifacts typical of text-to-speech systems."
        else:
            return f"The audio exhibits features commonly associated with AI-generated voice ({lang_name} context). Further verification recommended for borderline cases."
    else:
        if confidence >= 0.9:
            return f"The audio sample (detected as {lang_name}) displays natural human speech patterns. Phonetic variations and prosody are consistent with genuine human recording."
        else:
            return f"The audio appears to be human-generated ({lang_name} context). Confidence is moderate—consider additional verification for critical applications."


@app.post("/detect")
async def detect_voice(request: VoiceDetectionRequest):
    """
    Detect whether voice sample is AI-generated or human-generated.
    Accepts Base64-encoded MP3. Returns classification, confidence, explanation.
    """
    try:
        waveform, sample_rate = decode_audio(request.audio_base64)
        pipe = get_classifier()

        result = pipe({"array": waveform, "sampling_rate": sample_rate})
        scores = {item["label"]: item["score"] for item in result[0]}

        fake_score = scores.get("fake", scores.get("LABEL_0", 0))
        real_score = scores.get("real", scores.get("LABEL_1", 0))

        if fake_score >= real_score:
            classification = "AI-generated"
            confidence = float(fake_score)
        else:
            classification = "human-generated"
            confidence = float(real_score)

        explanation = get_explanation(classification, confidence, request.language)

        return {
            "classification": classification,
            "confidence_score": round(confidence, 4),
            "explanation": explanation,
            "raw_scores": {"ai_generated": fake_score, "human": real_score},
            "supported_languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api-info")
async def api_info():
    return {
        "api": "AI Voice Detection",
        "version": "1.0",
        "endpoints": {"POST /detect": "Analyze Base64 MP3 for AI vs human voice"},
        "languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
