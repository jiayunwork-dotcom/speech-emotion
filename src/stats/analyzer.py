from collections import defaultdict
from typing import List, Dict
import numpy as np

from src.core.config import settings
from src.core.models import (
    AudioAnalysisResult,
    AudioSegment,
    StatisticsSummary,
    SpeakerStats,
    EmotionDistribution,
    SpeakerEmotionDistribution,
    InterruptionEvent,
    EmotionTimelinePoint
)


class StatisticsAnalyzer:
    EMOTION_SCORES = {
        "neutral": 0,
        "happy": 2,
        "angry": -2,
        "sad": -1,
        "fear": -1.5,
        "surprise": 1.5
    }

    @staticmethod
    def calculate_speaker_stats(segments: List[AudioSegment]) -> List[SpeakerStats]:
        speaker_durations = defaultdict(int)
        speaker_counts = defaultdict(int)
        total_duration = 0
        
        for seg in segments:
            speaker = seg.speaker or "Unknown"
            duration = seg.end_ms - seg.start_ms
            speaker_durations[speaker] += duration
            speaker_counts[speaker] += 1
            total_duration += duration
        
        result = []
        for speaker, duration in speaker_durations.items():
            percentage = (duration / total_duration * 100) if total_duration > 0 else 0
            result.append(SpeakerStats(
                speaker=speaker,
                total_duration_ms=duration,
                percentage=round(percentage, 2),
                segment_count=speaker_counts[speaker]
            ))
        
        result.sort(key=lambda x: x.total_duration_ms, reverse=True)
        return result

    @staticmethod
    def calculate_emotion_distribution(segments: List[AudioSegment]) -> List[EmotionDistribution]:
        emotion_counts = defaultdict(int)
        
        for seg in segments:
            emotion = seg.emotion or "neutral"
            emotion_counts[emotion] += 1
        
        result = []
        for emotion in settings.EMOTION_CLASSES:
            result.append(EmotionDistribution(
                emotion=emotion,
                count=emotion_counts.get(emotion, 0)
            ))
        
        return result

    @staticmethod
    def calculate_speaker_emotions(segments: List[AudioSegment]) -> List[SpeakerEmotionDistribution]:
        speaker_emotions = defaultdict(lambda: defaultdict(int))
        
        for seg in segments:
            speaker = seg.speaker or "Unknown"
            emotion = seg.emotion or "neutral"
            speaker_emotions[speaker][emotion] += 1
        
        result = []
        speakers = sorted(speaker_emotions.keys())
        for speaker in speakers:
            emotions = []
            for emotion in settings.EMOTION_CLASSES:
                emotions.append(EmotionDistribution(
                    emotion=emotion,
                    count=speaker_emotions[speaker].get(emotion, 0)
                ))
            result.append(SpeakerEmotionDistribution(
                speaker=speaker,
                emotions=emotions
            ))
        
        return result

    @staticmethod
    def detect_interruptions(segments: List[AudioSegment]) -> List[InterruptionEvent]:
        interruptions = []
        threshold = settings.INTERRUPTION_THRESHOLD_MS
        
        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]
            
            prev_speaker = prev_seg.speaker
            curr_speaker = curr_seg.speaker
            
            if prev_speaker != curr_speaker and prev_speaker and curr_speaker:
                gap = curr_seg.start_ms - prev_seg.end_ms
                if gap >= 0 and gap < threshold:
                    interruptions.append(InterruptionEvent(
                        interrupter=curr_speaker,
                        interrupted=prev_speaker,
                        time_ms=prev_seg.end_ms
                    ))
        
        return interruptions

    @staticmethod
    def calculate_avg_segment_duration(segments: List[AudioSegment]) -> float:
        if not segments:
            return 0.0
        durations = [seg.end_ms - seg.start_ms for seg in segments]
        return float(np.mean(durations))

    @staticmethod
    def calculate_emotion_timeline(segments: List[AudioSegment]) -> List[EmotionTimelinePoint]:
        timeline = []
        
        for seg in segments:
            emotion = seg.emotion or "neutral"
            score = StatisticsAnalyzer.EMOTION_SCORES.get(emotion, 0)
            speaker = seg.speaker or "Unknown"
            
            mid_time = (seg.start_ms + seg.end_ms) // 2
            
            timeline.append(EmotionTimelinePoint(
                time_ms=mid_time,
                emotion_score=score,
                speaker=speaker,
                emotion=emotion
            ))
        
        return timeline

    @staticmethod
    def generate_summary(result: AudioAnalysisResult) -> StatisticsSummary:
        segments = result.segments
        
        speaker_stats = StatisticsAnalyzer.calculate_speaker_stats(segments)
        emotion_dist = StatisticsAnalyzer.calculate_emotion_distribution(segments)
        speaker_emotions = StatisticsAnalyzer.calculate_speaker_emotions(segments)
        interruptions = StatisticsAnalyzer.detect_interruptions(segments)
        avg_duration = StatisticsAnalyzer.calculate_avg_segment_duration(segments)
        
        return StatisticsSummary(
            task_id=result.task_id,
            total_duration_ms=result.duration_ms,
            total_segments=len(segments),
            speaker_stats=speaker_stats,
            emotion_distribution=emotion_dist,
            speaker_emotions=speaker_emotions,
            interruptions=interruptions,
            interruption_count=len(interruptions),
            avg_segment_duration_ms=round(avg_duration, 2)
        )
