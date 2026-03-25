"""
FastAPI Entry Point for Multi-Modal Biometric Liveness Detection.
Handles incoming video/audio payloads and routes them to the fusion pipeline.
"""

import os
import cv2
import librosa
import numpy as np
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import logging

from src.face_module import FaceLivenessDetector
from src.voice_module import VoiceLivenessDetector
from src.fusion import fuse_scores

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Biometric Liveness API",
    description="Multi-Modal Liveness Detection evaluating Face and Voice parameters.",
    version="1.0.0"
)

# Initialize AI Modules globally to load weights onto RAM
logger.info("Initializing Face and Voice Modules...")
face_detector = FaceLivenessDetector()
voice_detector = VoiceLivenessDetector()
logger.info("Modules Active.")

class VerificationResponse(BaseModel):
    liveness_score: float
    status: str
    message: str


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Biometric Liveness API is running."}


@app.post("/verify", response_model=VerificationResponse)
async def verify_liveness(
    video_file: UploadFile = File(..., description="Video stream containing face data"),
    audio_file: UploadFile = File(..., description="Audio stream containing voice data")
):
    """
    Endpoint to process video and audio streams for liveness verification.
    Handles temporal face execution over video frames, and spectogram analysis over audio file.
    """
    if not video_file.filename or not audio_file.filename:
        raise HTTPException(status_code=400, detail="Missing video or audio payload.")

    logger.info(f"Received verification request. Video: {video_file.filename}, Audio: {audio_file.filename}")

    # Create temporary payload IDs
    session_id = str(uuid.uuid4())
    temp_video_path = f"temp_{session_id}_{video_file.filename}"
    temp_audio_path = f"temp_{session_id}_{audio_file.filename}"

    try:
        # AIO save files
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
            
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        # --- 1. Face Execution ---
        logger.info("Processing Vision Pipeline...")
        cap = cv2.VideoCapture(temp_video_path)
        frame_scores = []
        
        # Limit to 30 frames to ensure fast API response time
        frame_count = 0
        while cap.isOpened() and frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
                
            score = face_detector.get_liveness_score(frame)
            frame_scores.append(score)
            frame_count += 1
            
        cap.release()
        
        if not frame_scores:
            raise ValueError("Video extraction failed or video is empty.")
            
        # Overall face score is median of valid frames to remove blink outliers
        overall_face_score = float(np.median(frame_scores))

        # --- 2. Voice Execution ---
        logger.info("Processing Acoustic Pipeline...")
        # sr=None preserves the original sampling rate for HF checks
        y, sr = librosa.load(temp_audio_path, sr=None)
        overall_voice_score = voice_detector.analyze_audio(y, original_sr=sr)

        # --- 3. Score Fusion ---
        logger.info(f"Fusing scores -> Face: {overall_face_score:.2f} | Voice: {overall_voice_score:.2f}")
        final_score, status = fuse_scores(overall_face_score, overall_voice_score)

        return VerificationResponse(
            liveness_score=final_score,
            status=status,
            message=f"Analyzed {frame_count} video frames and {len(y)/sr:.1f}s of audio."
        )

    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
    finally:
        # File System Cleanup
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
