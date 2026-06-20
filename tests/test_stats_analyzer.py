import pytest
from datetime import datetime

from src.stats.analyzer import StatisticsAnalyzer
from src.core.models import AudioSegment, AudioAnalysisResult, EmotionLabel


def test_calculate_speaker_stats():
    segments = [
        AudioSegment(start_ms=0, end_ms=3000, speaker="Speaker_1"),
        AudioSegment(start_ms=3500, end_ms=5000, speaker="Speaker_2"),
        AudioSegment(start_ms=5500, end_ms=8000, speaker="Speaker_1"),
        AudioSegment(start_ms=8500, end_ms=10000, speaker="Speaker_2"),
    ]
    
    stats = StatisticsAnalyzer.calculate_speaker_stats(segments)
    
    assert len(stats) == 2
    
    speaker1 = next(s for s in stats if s.speaker == "Speaker_1")
    speaker2 = next(s for s in stats if s.speaker == "Speaker_2")
    
    assert speaker1.total_duration_ms == 5500
    assert speaker2.total_duration_ms == 3000
    assert speaker1.segment_count == 2
    assert speaker2.segment_count == 2
    assert abs(speaker1.percentage + speaker2.percentage - 100.0) < 0.1


def test_calculate_emotion_distribution():
    segments = [
        AudioSegment(start_ms=0, end_ms=1000, emotion=EmotionLabel.HAPPY),
        AudioSegment(start_ms=1000, end_ms=2000, emotion=EmotionLabel.ANGRY),
        AudioSegment(start_ms=2000, end_ms=3000, emotion=EmotionLabel.HAPPY),
        AudioSegment(start_ms=3000, end_ms=4000, emotion=EmotionLabel.NEUTRAL),
        AudioSegment(start_ms=4000, end_ms=5000, emotion=EmotionLabel.SAD),
    ]
    
    dist = StatisticsAnalyzer.calculate_emotion_distribution(segments)
    
    happy = next(d for d in dist if d.emotion == "happy")
    angry = next(d for d in dist if d.emotion == "angry")
    neutral = next(d for d in dist if d.emotion == "neutral")
    
    assert happy.count == 2
    assert angry.count == 1
    assert neutral.count == 1


def test_detect_interruptions():
    segments = [
        AudioSegment(start_ms=0, end_ms=3000, speaker="Speaker_1"),
        AudioSegment(start_ms=3100, end_ms=5000, speaker="Speaker_2"),
        AudioSegment(start_ms=5300, end_ms=7000, speaker="Speaker_1"),
        AudioSegment(start_ms=7500, end_ms=9000, speaker="Speaker_2"),
    ]
    
    interruptions = StatisticsAnalyzer.detect_interruptions(segments)
    
    assert len(interruptions) == 1
    assert interruptions[0].interrupter == "Speaker_2"
    assert interruptions[0].interrupted == "Speaker_1"


def test_calculate_avg_segment_duration():
    segments = [
        AudioSegment(start_ms=0, end_ms=2000),
        AudioSegment(start_ms=2500, end_ms=4500),
        AudioSegment(start_ms=5000, end_ms=7000),
    ]
    
    avg = StatisticsAnalyzer.calculate_avg_segment_duration(segments)
    
    assert avg == pytest.approx(2000.0, 0.1)


def test_generate_summary():
    segments = [
        AudioSegment(start_ms=0, end_ms=3000, speaker="Speaker_1", emotion=EmotionLabel.HAPPY, emotion_confidence=0.9),
        AudioSegment(start_ms=3100, end_ms=5000, speaker="Speaker_2", emotion=EmotionLabel.ANGRY, emotion_confidence=0.85),
        AudioSegment(start_ms=5300, end_ms=8000, speaker="Speaker_1", emotion=EmotionLabel.NEUTRAL, emotion_confidence=0.7),
    ]
    
    result = AudioAnalysisResult(
        task_id="test_123",
        original_filename="test.wav",
        duration_ms=8000,
        sample_rate=16000,
        segments=segments,
        created_at=datetime.now()
    )
    
    summary = StatisticsAnalyzer.generate_summary(result)
    
    assert summary.task_id == "test_123"
    assert summary.total_duration_ms == 8000
    assert summary.total_segments == 3
    assert len(summary.speaker_stats) == 2
    assert summary.interruption_count == 1
    assert summary.avg_segment_duration_ms > 0
