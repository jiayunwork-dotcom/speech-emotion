import pytest
import numpy as np

from src.emotion.recognizer import EmotionRecognizer
from src.core.config import settings


def test_emotion_recognizer_initialization():
    recognizer = EmotionRecognizer()
    assert recognizer.emotion_classes == settings.EMOTION_CLASSES
    assert len(recognizer.emotion_classes) == 6


def test_extract_f0_features():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 220 * t)
    
    features = EmotionRecognizer.extract_f0_features(y, sr)
    
    assert 'f0_mean' in features
    assert 'f0_std' in features
    assert 'f0_range' in features
    assert features['f0_mean'] > 0


def test_extract_emotion_features():
    recognizer = EmotionRecognizer()
    
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    
    features = recognizer.extract_emotion_features(y, sr)
    
    assert len(features) >= 34


def test_rule_based_predict():
    recognizer = EmotionRecognizer()
    
    angry_features = np.array([150, 80, 100, 3, 0.15, 0.2, 1000, 2000] + [0]*26)
    emotion, probs = recognizer.rule_based_predict(angry_features)
    assert emotion == 'angry'
    assert abs(sum(probs.values()) - 1.0) < 0.01
    
    happy_features = np.array([250, 50, 80, 4, 0.08, 0.1, 1500, 3000] + [0]*26)
    emotion, probs = recognizer.rule_based_predict(happy_features)
    assert emotion == 'happy'
    
    sad_features = np.array([100, 10, 20, 1, 0.02, 0.05, 500, 1000] + [0]*26)
    emotion, probs = recognizer.rule_based_predict(sad_features)
    assert emotion == 'sad'
    
    neutral_features = np.array([150, 20, 30, 2, 0.05, 0.08, 800, 1500] + [0]*26)
    emotion, probs = recognizer.rule_based_predict(neutral_features)
    assert emotion == 'neutral'


def test_predict():
    recognizer = EmotionRecognizer()
    
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    
    emotion, confidence, probs = recognizer.predict(y, sr)
    
    assert emotion in settings.EMOTION_CLASSES
    assert 0 <= confidence <= 1
    assert abs(sum(probs.values()) - 1.0) < 0.01
    assert set(probs.keys()) == set(settings.EMOTION_CLASSES)
