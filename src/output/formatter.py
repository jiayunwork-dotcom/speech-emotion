import json
from datetime import datetime, timedelta
from typing import List
from pathlib import Path

from src.core.config import settings
from src.core.models import AudioAnalysisResult, AudioSegment


class OutputFormatter:
    @staticmethod
    def format_timestamp(ms: int) -> str:
        seconds = ms / 1000.0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def to_json(result: AudioAnalysisResult) -> str:
        result_dict = {
            "task_id": result.task_id,
            "original_filename": result.original_filename,
            "duration_ms": result.duration_ms,
            "sample_rate": result.sample_rate,
            "created_at": result.created_at.isoformat(),
            "segments": []
        }
        
        for seg in result.segments:
            seg_dict = {
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "speaker": seg.speaker,
                "emotion": seg.emotion.value if seg.emotion else None,
                "emotion_confidence": seg.emotion_confidence,
                "emotion_probabilities": seg.emotion_probabilities
            }
            result_dict["segments"].append(seg_dict)
        
        return json.dumps(result_dict, ensure_ascii=False, indent=2)

    @staticmethod
    def to_srt(result: AudioAnalysisResult) -> str:
        srt_lines = []
        
        for i, seg in enumerate(result.segments, 1):
            start_time = OutputFormatter.format_timestamp(seg.start_ms)
            end_time = OutputFormatter.format_timestamp(seg.end_ms)
            
            speaker = seg.speaker or "Unknown"
            emotion = seg.emotion or "neutral"
            confidence = f"{seg.emotion_confidence:.2f}" if seg.emotion_confidence else "N/A"
            
            text = f"[{speaker}] [{emotion}] (confidence: {confidence})"
            
            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")
        
        return "\n".join(srt_lines)

    @staticmethod
    def save_json(result: AudioAnalysisResult, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_str = OutputFormatter.to_json(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        return output_path

    @staticmethod
    def save_srt(result: AudioAnalysisResult, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        srt_str = OutputFormatter.to_srt(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_str)
        return output_path

    @staticmethod
    def load_json(task_id: str) -> AudioAnalysisResult:
        json_path = settings.PROCESSED_DATA_DIR / f"{task_id}_result.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        for seg_data in data["segments"]:
            segments.append(AudioSegment(**seg_data))
        
        return AudioAnalysisResult(
            task_id=data["task_id"],
            original_filename=data["original_filename"],
            duration_ms=data["duration_ms"],
            sample_rate=data["sample_rate"],
            created_at=datetime.fromisoformat(data["created_at"]),
            segments=segments
        )
