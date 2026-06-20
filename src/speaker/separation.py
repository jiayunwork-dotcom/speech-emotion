import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.core.config import settings
from src.core.models import AudioSegment
from src.audio.processor import AudioProcessor


class SpeakerSeparator:
    @staticmethod
    def extract_mfcc_features(segment_audio: np.ndarray, sr: int, n_mfcc: int) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        combined = np.vstack([mfcc, delta_mfcc])
        return combined

    @staticmethod
    def extract_speaker_embedding(segment_audio: np.ndarray, sr: int) -> np.ndarray:
        window_duration = settings.SPEAKER_WINDOW_DURATION
        overlap = settings.SPEAKER_WINDOW_OVERLAP
        n_mfcc = settings.SPEAKER_MFCC_COEFFS
        
        window_samples = int(window_duration * sr)
        hop_samples = int(window_samples * (1 - overlap))
        
        if len(segment_audio) < window_samples:
            segment_audio = np.pad(segment_audio, (0, window_samples - len(segment_audio)))
        
        window_features = []
        
        for start in range(0, len(segment_audio) - window_samples + 1, hop_samples):
            window = segment_audio[start:start + window_samples]
            features = SpeakerSeparator.extract_mfcc_features(window, sr, n_mfcc)
            feature_mean = np.mean(features, axis=1)
            window_features.append(feature_mean)
        
        if not window_features:
            window = segment_audio[:window_samples]
            features = SpeakerSeparator.extract_mfcc_features(window, sr, n_mfcc)
            feature_mean = np.mean(features, axis=1)
            window_features.append(feature_mean)
        
        window_features = np.array(window_features)
        mean_features = np.mean(window_features, axis=0)
        std_features = np.std(window_features, axis=0)
        
        embedding = np.concatenate([mean_features, std_features])
        return embedding

    @staticmethod
    def cluster_speakers(embeddings: np.ndarray) -> List[int]:
        if len(embeddings) == 0:
            return []
        
        if len(embeddings) == 1:
            return [0]
        
        n_samples = len(embeddings)
        max_clusters = min(settings.MAX_SPEAKERS, n_samples)
        
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = cosine(embeddings[i], embeddings[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=settings.SPEAKER_CLUSTERING_THRESHOLD
        )
        
        labels = clusterer.fit_predict(distance_matrix)
        
        unique_labels = np.unique(labels)
        if len(unique_labels) > settings.MAX_SPEAKERS:
            clusterer2 = AgglomerativeClustering(
                n_clusters=settings.MAX_SPEAKERS,
                metric='precomputed',
                linkage='average'
            )
            labels = clusterer2.fit_predict(distance_matrix)
        
        return labels.tolist()

    @staticmethod
    def assign_speaker_labels(segments: List[AudioSegment], y: np.ndarray, sr: int) -> List[AudioSegment]:
        embeddings = []
        valid_segments = []
        
        for segment in segments:
            try:
                segment_audio = AudioProcessor.extract_segment_audio(y, sr, segment)
                if len(segment_audio) > 0:
                    embedding = SpeakerSeparator.extract_speaker_embedding(segment_audio, sr)
                    embeddings.append(embedding)
                    valid_segments.append(segment)
                    segment.speaker_embedding = embedding.tolist()
                else:
                    valid_segments.append(segment)
            except Exception as e:
                valid_segments.append(segment)
        
        if len(embeddings) == 0:
            for segment in segments:
                segment.speaker = "Speaker_1"
            return segments
        
        embeddings_array = np.array(embeddings)
        cluster_labels = SpeakerSeparator.cluster_speakers(embeddings_array)
        
        speaker_order = {}
        current_order = 0
        for i, segment in enumerate(valid_segments):
            if i < len(cluster_labels):
                label = cluster_labels[i]
                if label not in speaker_order:
                    speaker_order[label] = current_order
                    current_order += 1
                speaker_idx = speaker_order[label] + 1
                segment.speaker = f"Speaker_{speaker_idx}"
            else:
                segment.speaker = "Speaker_1"
        
        for segment in segments:
            if segment.speaker is None:
                segment.speaker = "Speaker_1"
        
        return segments
