import os
import io
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional
import uuid

from src.core.config import settings
from src.core.models import AudioSegment


class AudioProcessor:
    @staticmethod
    def load_audio(file_path: Path) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(str(file_path), sr=None, mono=False)
        return y, sr

    @staticmethod
    def convert_to_target_format(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        
        if sr != settings.TARGET_SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=settings.TARGET_SAMPLE_RATE)
            sr = settings.TARGET_SAMPLE_RATE
        
        return y, sr

    @staticmethod
    def validate_audio(file_path: Path) -> Tuple[bool, str]:
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            return False, f"File size {file_size_mb:.1f}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB"
        
        try:
            y, sr = AudioProcessor.load_audio(file_path)
            duration = librosa.get_duration(y=y, sr=sr)
            if duration > settings.MAX_DURATION_MINUTES * 60:
                return False, f"Duration {duration/60:.1f}min exceeds limit of {settings.MAX_DURATION_MINUTES}min"
        except Exception as e:
            return False, f"Invalid audio file: {str(e)}"
        
        return True, "OK"

    @staticmethod
    def save_processed_audio(y: np.ndarray, sr: int, task_id: str) -> Path:
        output_path = settings.PROCESSED_DATA_DIR / f"{task_id}.wav"
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), y, sr)
        return output_path

    @staticmethod
    def load_processed_audio(task_id: str) -> Tuple[np.ndarray, int]:
        audio_path = settings.PROCESSED_DATA_DIR / f"{task_id}.wav"
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        return y, sr

    @staticmethod
    def get_audio_duration(y: np.ndarray, sr: int) -> int:
        return int(librosa.get_duration(y=y, sr=sr) * 1000)

    @staticmethod
    def vad_split(y: np.ndarray, sr: int) -> List[AudioSegment]:
        energy = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        energy_threshold = np.mean(energy) * settings.VAD_SILENCE_THRESHOLD_RATIO
        
        frame_duration_ms = (256 / sr) * 1000
        
        is_speech = energy > energy_threshold
        
        segments = []
        in_speech = False
        speech_start = 0
        silence_start = 0
        
        for i, speech_flag in enumerate(is_speech):
            current_time_ms = int(i * frame_duration_ms)
            
            if speech_flag and not in_speech:
                silence_duration = current_time_ms - silence_start
                if silence_start > 0 and silence_duration < settings.VAD_MIN_SILENCE_DURATION_MS:
                    pass
                else:
                    in_speech = True
                    speech_start = current_time_ms
            elif not speech_flag and in_speech:
                silence_start = current_time_ms
                silence_duration = 0
                in_speech = False
                if segments:
                    last_segment = segments[-1]
                    gap = speech_start - last_segment.end_ms
                    if gap < settings.VAD_MIN_SPEECH_GAP_MS:
                        last_segment.end_ms = int(i * frame_duration_ms)
                        continue
                
                segments.append(AudioSegment(
                    start_ms=speech_start,
                    end_ms=int(i * frame_duration_ms)
                ))
        
        if in_speech:
            segments.append(AudioSegment(
                start_ms=speech_start,
                end_ms=int(len(is_speech) * frame_duration_ms)
            ))
        
        merged_segments = []
        for seg in segments:
            if merged_segments:
                last = merged_segments[-1]
                if seg.start_ms - last.end_ms < settings.VAD_MIN_SPEECH_GAP_MS:
                    last.end_ms = seg.end_ms
                    continue
            merged_segments.append(seg)
        
        min_segment_duration = 200
        merged_segments = [s for s in merged_segments if (s.end_ms - s.start_ms) > min_segment_duration]
        
        return merged_segments

    @staticmethod
    def extract_segment_audio(y: np.ndarray, sr: int, segment: AudioSegment) -> np.ndarray:
        start_sample = int(segment.start_ms / 1000 * sr)
        end_sample = int(segment.end_ms / 1000 * sr)
        return y[start_sample:end_sample]
