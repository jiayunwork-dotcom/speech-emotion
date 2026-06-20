from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
import uuid
from datetime import datetime


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EmotionLabel(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    FEAR = "fear"
    SURPRISE = "surprise"


class AudioSegment(BaseModel):
    start_ms: int
    end_ms: int
    speaker: Optional[str] = None
    emotion: Optional[EmotionLabel] = None
    emotion_confidence: Optional[float] = None
    emotion_probabilities: Optional[Dict[str, float]] = None
    speaker_embedding: Optional[List[float]] = None


class AudioAnalysisResult(BaseModel):
    task_id: str
    original_filename: str
    duration_ms: int
    sample_rate: int
    segments: List[AudioSegment]
    created_at: datetime = Field(default_factory=datetime.now)


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    error_message: Optional[str] = None


class BatchTaskCreate(BaseModel):
    file_names: List[str]


class BatchTaskResponse(BaseModel):
    batch_id: str
    tasks: List[TaskResponse]


class BatchStatusResponse(BaseModel):
    batch_id: str
    tasks: List[TaskStatusResponse]


class SpeakerStats(BaseModel):
    speaker: str
    total_duration_ms: int
    percentage: float
    segment_count: int


class EmotionDistribution(BaseModel):
    emotion: str
    count: int


class SpeakerEmotionDistribution(BaseModel):
    speaker: str
    emotions: List[EmotionDistribution]


class InterruptionEvent(BaseModel):
    interrupter: str
    interrupted: str
    time_ms: int


class StatisticsSummary(BaseModel):
    task_id: str
    total_duration_ms: int
    total_segments: int
    speaker_stats: List[SpeakerStats]
    emotion_distribution: List[EmotionDistribution]
    speaker_emotions: List[SpeakerEmotionDistribution]
    interruptions: List[InterruptionEvent]
    interruption_count: int
    avg_segment_duration_ms: float


class EmotionTimelinePoint(BaseModel):
    time_ms: int
    emotion_score: float
    speaker: str
    emotion: str
