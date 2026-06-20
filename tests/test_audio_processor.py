import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

from src.audio.processor import AudioProcessor
from src.core.config import settings


def test_convert_to_target_format():
    sr_original = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(sr_original * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t)
    
    y_2ch = np.vstack([y, y])
    
    y_converted, sr_converted = AudioProcessor.convert_to_target_format(y_2ch, sr_original)
    
    assert sr_converted == settings.TARGET_SAMPLE_RATE
    assert y_converted.ndim == 1
    assert len(y_converted) == int(duration * settings.TARGET_SAMPLE_RATE)


def test_vad_split():
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    y = np.zeros_like(t)
    
    speech1_start = int(0.5 * sr)
    speech1_end = int(1.5 * sr)
    y[speech1_start:speech1_end] = 0.5 * np.sin(2 * np.pi * 440 * t[speech1_start:speech1_end])
    
    speech2_start = int(2.5 * sr)
    speech2_end = int(3.5 * sr)
    y[speech2_start:speech2_end] = 0.5 * np.sin(2 * np.pi * 660 * t[speech2_start:speech2_end])
    
    segments = AudioProcessor.vad_split(y, sr)
    
    assert len(segments) >= 2
    for seg in segments:
        assert seg.start_ms < seg.end_ms
        assert seg.end_ms - seg.start_ms > 100


def test_save_and_load_processed_audio():
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t)
    
    task_id = "test_task_123"
    saved_path = AudioProcessor.save_processed_audio(y, sr, task_id)
    
    assert saved_path.exists()
    assert saved_path.suffix == '.wav'
    
    y_loaded, sr_loaded = AudioProcessor.load_processed_audio(task_id)
    
    assert sr_loaded == sr
    assert len(y_loaded) == len(y)
    
    saved_path.unlink()


def test_get_audio_duration():
    sr = 16000
    duration = 3.5
    y = np.zeros(int(sr * duration))
    
    duration_ms = AudioProcessor.get_audio_duration(y, sr)
    
    assert abs(duration_ms - int(duration * 1000)) < 100
