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


class QualityGrade(str, Enum):
    EXCELLENT = "优秀"
    GOOD = "良好"
    FAIR = "一般"
    POOR = "较差"


class QualityDimensionScore(BaseModel):
    name: str
    score: float
    raw_value: Optional[float] = None
    unit: Optional[str] = None


class QualitySuggestion(BaseModel):
    dimension: str
    problem: str
    suggestion: str


class QualityAssessment(BaseModel):
    task_id: str
    snr: QualityDimensionScore
    clipping: QualityDimensionScore
    speech_ratio: QualityDimensionScore
    sample_rate_fitness: QualityDimensionScore
    overall_score: float
    grade: QualityGrade
    suggestions: List[QualitySuggestion] = []
    created_at: datetime = Field(default_factory=datetime.now)


class AudioMetaInfo(BaseModel):
    task_id: str
    original_sample_rate: int
    original_channels: int
    original_duration_ms: int
    processed_sample_rate: int
    processed_channels: int
    processed_duration_ms: int
    created_at: datetime = Field(default_factory=datetime.now)


class AudioAnalysisResult(BaseModel):
    task_id: str
    original_filename: str
    duration_ms: int
    sample_rate: int
    segments: List[AudioSegment]
    quality_assessment: Optional[QualityAssessment] = None
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


class ComparisonLabel(str, Enum):
    TASK1 = "task1占优"
    TASK2 = "task2占优"
    TIE = "持平"


class SpeakerDominanceItem(BaseModel):
    speaker: str
    percentage_task1: Optional[float] = None
    percentage_task2: Optional[float] = None


class SpeakerDominanceComparison(BaseModel):
    speakers: List[SpeakerDominanceItem]
    max_speaker_task1: Optional[str] = None
    max_percentage_task1: float = 0.0
    max_speaker_task2: Optional[str] = None
    max_percentage_task2: float = 0.0
    conclusion: ComparisonLabel


class SentimentComparison(BaseModel):
    sentiment_score_task1: float
    sentiment_score_task2: float
    conclusion: ComparisonLabel


class ActivityComparison(BaseModel):
    avg_segment_duration_ms_task1: float
    avg_segment_duration_ms_task2: float
    total_segments_task1: int
    total_segments_task2: int
    conclusion: ComparisonLabel


class InterruptionComparison(BaseModel):
    interruption_rate_task1: float
    interruption_rate_task2: float
    interruption_count_task1: int
    interruption_count_task2: int
    total_segments_task1: int
    total_segments_task2: int
    conclusion: ComparisonLabel


class ComparisonReport(BaseModel):
    task1_id: str
    task2_id: str
    task1_filename: str
    task2_filename: str
    speaker_dominance: SpeakerDominanceComparison
    sentiment: SentimentComparison
    activity: ActivityComparison
    interruption: InterruptionComparison


class TaskListItem(BaseModel):
    task_id: str
    filename: str
    status: TaskStatus
    created_at: Optional[datetime] = None

