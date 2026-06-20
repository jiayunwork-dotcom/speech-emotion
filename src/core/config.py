import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    TARGET_SAMPLE_RATE: int = 16000
    TARGET_CHANNELS: int = 1
    MAX_FILE_SIZE_MB: int = 50
    MAX_DURATION_MINUTES: int = 30
    
    VAD_SILENCE_THRESHOLD_RATIO: float = 0.1
    VAD_MIN_SILENCE_DURATION_MS: int = 500
    VAD_MIN_SPEECH_GAP_MS: int = 300
    
    SPEAKER_WINDOW_DURATION: float = 1.5
    SPEAKER_WINDOW_OVERLAP: float = 0.5
    SPEAKER_MFCC_COEFFS: int = 20
    SPEAKER_CLUSTERING_THRESHOLD: float = 0.6
    MAX_SPEAKERS: int = 10
    
    EMOTION_CLASSES: list = ["neutral", "happy", "angry", "sad", "fear", "surprise"]
    EMOTION_MFCC_COEFFS: int = 13
    
    BATCH_MAX_FILES: int = 10
    BATCH_MAX_PARALLEL: int = 3
    
    INTERRUPTION_THRESHOLD_MS: int = 200
    
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    EMOTION_MODEL_PATH: Path = MODELS_DIR / "emotion_rf_model.pkl"
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501

settings = Settings()
