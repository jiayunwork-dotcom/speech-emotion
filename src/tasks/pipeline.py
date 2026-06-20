import numpy as np
from typing import List, Tuple
from pathlib import Path

from src.core.config import settings
from src.core.models import AudioAnalysisResult, AudioSegment
from src.audio.processor import AudioProcessor
from src.speaker.separation import SpeakerSeparator
from src.emotion.recognizer import EmotionRecognizer
from src.output.formatter import OutputFormatter


class AnalysisPipeline:
    def __init__(self):
        self.emotion_recognizer = EmotionRecognizer()

    def process_audio_file(self, file_path: Path, task_id: str) -> AudioAnalysisResult:
        y, sr = AudioProcessor.load_audio(file_path)
        
        y, sr = AudioProcessor.convert_to_target_format(y, sr)
        duration_ms = AudioProcessor.get_audio_duration(y, sr)
        AudioProcessor.save_processed_audio(y, sr, task_id)
        
        segments = AudioProcessor.vad_split(y, sr)
        segments = SpeakerSeparator.assign_speaker_labels(segments, y, sr)
        segments = self.emotion_recognizer.recognize_emotions(segments, y, sr)
        
        result = AudioAnalysisResult(
            task_id=task_id,
            original_filename=file_path.name,
            duration_ms=duration_ms,
            sample_rate=sr,
            segments=segments
        )
        
        json_path = settings.PROCESSED_DATA_DIR / f"{task_id}_result.json"
        OutputFormatter.save_json(result, json_path)
        
        srt_path = settings.PROCESSED_DATA_DIR / f"{task_id}_result.srt"
        OutputFormatter.save_srt(result, srt_path)
        
        return result

    def process_segment(self, y: np.ndarray, sr: int, segment: AudioSegment) -> AudioSegment:
        segment_audio = AudioProcessor.extract_segment_audio(y, sr, segment)
        
        if segment.speaker is None:
            embedding = SpeakerSeparator.extract_speaker_embedding(segment_audio, sr)
            segment.speaker_embedding = embedding.tolist()
        
        if segment.emotion is None:
            emotion, confidence, probs = self.emotion_recognizer.predict(segment_audio, sr)
            segment.emotion = emotion
            segment.emotion_confidence = confidence
            segment.emotion_probabilities = probs
        
        return segment
