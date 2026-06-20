import pytest
import numpy as np

from src.speaker.separation import SpeakerSeparator
from src.core.models import AudioSegment
from src.core.config import settings


def test_extract_mfcc_features():
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t)
    
    features = SpeakerSeparator.extract_mfcc_features(y, sr, n_mfcc=20)
    
    assert features.shape[0] == 40


def test_extract_speaker_embedding():
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    embedding = SpeakerSeparator.extract_speaker_embedding(y, sr)
    
    assert embedding.ndim == 1
    expected_dim = 2 * 40 * int((settings.SPEAKER_WINDOW_DURATION * sr) / 512 + 1) * 2
    assert len(embedding) > 0


def test_cluster_speakers_single():
    embeddings = np.random.rand(1, 100)
    labels = SpeakerSeparator.cluster_speakers(embeddings)
    assert len(labels) == 1
    assert labels[0] == 0


def test_cluster_speakers_multiple():
    rng = np.random.default_rng(42)
    embeddings = np.vstack([
        rng.normal(0, 0.1, (5, 100)),
        rng.normal(1, 0.1, (5, 100)),
        rng.normal(2, 0.1, (5, 100))
    ])
    
    labels = SpeakerSeparator.cluster_speakers(embeddings)
    
    assert len(labels) == 15
    unique_labels = np.unique(labels)
    assert len(unique_labels) <= settings.MAX_SPEAKERS


def test_assign_speaker_labels():
    sr = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    y = np.zeros_like(t)
    
    freq1 = 440
    freq2 = 660
    for i in range(5):
        start = int(i * 2 * sr)
        end = int((i * 2 + 1) * sr)
        if i % 2 == 0:
            y[start:end] = 0.5 * np.sin(2 * np.pi * freq1 * t[start:end])
        else:
            y[start:end] = 0.5 * np.sin(2 * np.pi * freq2 * t[start:end])
    
    segments = [
        AudioSegment(start_ms=0, end_ms=1000),
        AudioSegment(start_ms=2000, end_ms=3000),
        AudioSegment(start_ms=4000, end_ms=5000),
        AudioSegment(start_ms=6000, end_ms=7000),
        AudioSegment(start_ms=8000, end_ms=9000),
    ]
    
    labeled_segments = SpeakerSeparator.assign_speaker_labels(segments, y, sr)
    
    assert len(labeled_segments) == 5
    for seg in labeled_segments:
        assert seg.speaker is not None
        assert seg.speaker.startswith("Speaker_")
