import os
import pickle
import numpy as np
import librosa
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.core.config import settings
from src.core.models import AudioSegment, EmotionLabel
from src.audio.processor import AudioProcessor


class EmotionRecognizer:
    def __init__(self):
        self.model = self._load_model()
        self.emotion_classes = settings.EMOTION_CLASSES
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotion_classes)}

    def _load_model(self):
        model_path = settings.EMOTION_MODEL_PATH
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    @staticmethod
    def extract_f0_features(y_segment: np.ndarray, sr: int) -> Dict[str, float]:
        f0, voiced_flag, _ = librosa.pyin(
            y_segment,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        f0_voiced = f0[voiced_flag & (f0 > 0)]
        if len(f0_voiced) == 0:
            return {
                'f0_mean': 0.0,
                'f0_std': 0.0,
                'f0_range': 0.0
            }
        
        return {
            'f0_mean': np.mean(f0_voiced),
            'f0_std': np.std(f0_voiced),
            'f0_range': np.max(f0_voiced) - np.min(f0_voiced)
        }

    @staticmethod
    def extract_speech_rate(y_segment: np.ndarray, sr: int) -> float:
        zcr = librosa.feature.zero_crossing_rate(y_segment, frame_length=512, hop_length=256)[0]
        duration = len(y_segment) / sr
        avg_zcr = np.mean(zcr)
        speech_rate = avg_zcr * 5.0
        return speech_rate if duration > 0 else 0.0

    @staticmethod
    def extract_energy_features(y_segment: np.ndarray) -> Dict[str, float]:
        energy = librosa.feature.rms(y=y_segment, frame_length=512, hop_length=256)[0]
        if len(energy) == 0:
            return {'energy_mean': 0.0, 'energy_range': 0.0}
        return {
            'energy_mean': np.mean(energy),
            'energy_range': np.max(energy) - np.min(energy)
        }

    @staticmethod
    def extract_mfcc_features(y_segment: np.ndarray, sr: int, n_mfcc: int) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    @staticmethod
    def extract_spectral_features(y_segment: np.ndarray, sr: int) -> Dict[str, float]:
        spectral_centroid = librosa.feature.spectral_centroid(y=y_segment, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)[0]
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_rolloff_mean': np.mean(spectral_rolloff)
        }

    def extract_emotion_features(self, y_segment: np.ndarray, sr: int) -> np.ndarray:
        f0_feats = EmotionRecognizer.extract_f0_features(y_segment, sr)
        speech_rate = EmotionRecognizer.extract_speech_rate(y_segment, sr)
        energy_feats = EmotionRecognizer.extract_energy_features(y_segment)
        mfcc_feats = EmotionRecognizer.extract_mfcc_features(y_segment, sr, settings.EMOTION_MFCC_COEFFS)
        spectral_feats = EmotionRecognizer.extract_spectral_features(y_segment, sr)
        
        features = np.array([
            f0_feats['f0_mean'],
            f0_feats['f0_std'],
            f0_feats['f0_range'],
            speech_rate,
            energy_feats['energy_mean'],
            energy_feats['energy_range'],
            spectral_feats['spectral_centroid_mean'],
            spectral_feats['spectral_rolloff_mean'],
            *mfcc_feats
        ])
        
        return features

    def rule_based_predict(self, features: np.ndarray) -> Tuple[str, Dict[str, float]]:
        f0_mean, f0_std, f0_range, speech_rate, energy_mean, energy_range = features[:6]
        
        probs = {e: 0.1 for e in self.emotion_classes}
        
        if f0_std > 50 and energy_mean > 0.1:
            probs['angry'] = 0.6
            probs['surprise'] = 0.2
        elif f0_mean > 200 and f0_std > 30:
            probs['happy'] = 0.6
            probs['surprise'] = 0.2
        elif f0_mean < 120 and energy_mean < 0.05:
            probs['sad'] = 0.6
            probs['fear'] = 0.1
        elif f0_range > 80 and speech_rate > 3:
            probs['fear'] = 0.5
            probs['surprise'] = 0.3
        elif f0_std > 40 and f0_mean > 180:
            probs['surprise'] = 0.5
            probs['happy'] = 0.2
        else:
            probs['neutral'] = 0.7
        
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        
        emotion = max(probs, key=probs.get)
        return emotion, probs

    def predict(self, y_segment: np.ndarray, sr: int) -> Tuple[str, float, Dict[str, float]]:
        if len(y_segment) < int(0.1 * sr):
            y_segment = np.pad(y_segment, (0, int(0.1 * sr) - len(y_segment)))
        
        features = self.extract_emotion_features(y_segment, sr)
        
        if self.model is not None and hasattr(self.model, 'predict_proba'):
            try:
                probs_array = self.model.predict_proba(features.reshape(1, -1))[0]
                probs = {self.emotion_classes[i]: float(probs_array[i]) for i in range(len(self.emotion_classes))}
                emotion_idx = np.argmax(probs_array)
                emotion = self.emotion_classes[emotion_idx]
                confidence = float(probs_array[emotion_idx])
                return emotion, confidence, probs
            except Exception:
                pass
        
        emotion, probs = self.rule_based_predict(features)
        confidence = probs[emotion]
        return emotion, confidence, probs

    def recognize_emotions(self, segments: List[AudioSegment], y: np.ndarray, sr: int) -> List[AudioSegment]:
        for segment in segments:
            try:
                segment_audio = AudioProcessor.extract_segment_audio(y, sr, segment)
                if len(segment_audio) > 0:
                    emotion, confidence, probs = self.predict(segment_audio, sr)
                    segment.emotion = emotion
                    segment.emotion_confidence = confidence
                    segment.emotion_probabilities = probs
                else:
                    segment.emotion = EmotionLabel.NEUTRAL
                    segment.emotion_confidence = 0.5
                    segment.emotion_probabilities = {e: 1/6 for e in self.emotion_classes}
            except Exception as e:
                segment.emotion = EmotionLabel.NEUTRAL
                segment.emotion_confidence = 0.3
                segment.emotion_probabilities = {e: 1/6 for e in self.emotion_classes}
        
        return segments
