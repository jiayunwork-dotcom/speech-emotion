from collections import defaultdict
from typing import List, Dict, Optional
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
    EmotionTimelinePoint,
    ComparisonReport,
    ComparisonLabel,
    SpeakerDominanceComparison,
    SpeakerDominanceItem,
    SentimentComparison,
    ActivityComparison,
    InterruptionComparison
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

    @staticmethod
    def _compare_values(val1: float, val2: float, is_higher_better: bool = True, percentage_threshold: float = 5.0, absolute_threshold: float = 0.05, use_percentage: bool = True) -> ComparisonLabel:
        diff = abs(val1 - val2)
        if use_percentage:
            max_val = max(abs(val1), abs(val2))
            if max_val > 0:
                pct_diff = (diff / max_val) * 100
            else:
                pct_diff = 0
            is_tie = pct_diff <= percentage_threshold or diff <= absolute_threshold
        else:
            is_tie = diff <= absolute_threshold
        if is_tie:
            return ComparisonLabel.TIE
        if is_higher_better:
            return ComparisonLabel.TASK1 if val1 > val2 else ComparisonLabel.TASK2
        else:
            return ComparisonLabel.TASK1 if val1 < val2 else ComparisonLabel.TASK2

    @staticmethod
    def calculate_sentiment_score(segments: List[AudioSegment]) -> float:
        if not segments:
            return 0.0
        positive_emotions = {"happy", "surprise"}
        negative_emotions = {"angry", "sad", "fear"}
        total = len(segments)
        positive_count = 0
        negative_count = 0
        for seg in segments:
            emotion = seg.emotion or "neutral"
            if emotion in positive_emotions:
                positive_count += 1
            elif emotion in negative_emotions:
                negative_count += 1
        if total == 0:
            return 0.0
        score = (positive_count - negative_count) / total
        return round(score, 4)

    @staticmethod
    def compare_speaker_dominance(result1: AudioAnalysisResult, result2: AudioAnalysisResult) -> SpeakerDominanceComparison:
        stats1 = StatisticsAnalyzer.calculate_speaker_stats(result1.segments)
        stats2 = StatisticsAnalyzer.calculate_speaker_stats(result2.segments)

        speaker_map1 = {s.speaker: s.percentage for s in stats1}
        speaker_map2 = {s.speaker: s.percentage for s in stats2}

        all_speakers = sorted(set(speaker_map1.keys()) | set(speaker_map2.keys()))

        speakers = []
        for sp in all_speakers:
            speakers.append(SpeakerDominanceItem(
                speaker=sp,
                percentage_task1=speaker_map1.get(sp),
                percentage_task2=speaker_map2.get(sp)
            ))

        max_speaker1 = stats1[0].speaker if stats1 else None
        max_pct1 = stats1[0].percentage if stats1 else 0.0
        max_speaker2 = stats2[0].speaker if stats2 else None
        max_pct2 = stats2[0].percentage if stats2 else 0.0

        conclusion = StatisticsAnalyzer._compare_values(max_pct1, max_pct2, is_higher_better=True)

        return SpeakerDominanceComparison(
            speakers=speakers,
            max_speaker_task1=max_speaker1,
            max_percentage_task1=max_pct1,
            max_speaker_task2=max_speaker2,
            max_percentage_task2=max_pct2,
            conclusion=conclusion
        )

    @staticmethod
    def compare_sentiment(result1: AudioAnalysisResult, result2: AudioAnalysisResult) -> SentimentComparison:
        score1 = StatisticsAnalyzer.calculate_sentiment_score(result1.segments)
        score2 = StatisticsAnalyzer.calculate_sentiment_score(result2.segments)
        conclusion = StatisticsAnalyzer._compare_values(score1, score2, is_higher_better=True, absolute_threshold=0.05, use_percentage=False)
        return SentimentComparison(
            sentiment_score_task1=score1,
            sentiment_score_task2=score2,
            conclusion=conclusion
        )

    @staticmethod
    def compare_activity(result1: AudioAnalysisResult, result2: AudioAnalysisResult) -> ActivityComparison:
        avg_dur1 = StatisticsAnalyzer.calculate_avg_segment_duration(result1.segments)
        avg_dur2 = StatisticsAnalyzer.calculate_avg_segment_duration(result2.segments)
        count1 = len(result1.segments)
        count2 = len(result2.segments)

        score1 = count1 / max(avg_dur1, 1) if avg_dur1 > 0 else 0
        score2 = count2 / max(avg_dur2, 1) if avg_dur2 > 0 else 0

        pct_diff = abs(count1 - count2) / max(count1, count2) * 100 if max(count1, count2) > 0 else 0
        avg_diff_pct = abs(avg_dur1 - avg_dur2) / max(avg_dur1, avg_dur2) * 100 if max(avg_dur1, avg_dur2) > 0 else 0

        if pct_diff <= 5 and avg_diff_pct <= 5:
            conclusion = ComparisonLabel.TIE
        elif score1 > score2:
            conclusion = ComparisonLabel.TASK1
        else:
            conclusion = ComparisonLabel.TASK2

        return ActivityComparison(
            avg_segment_duration_ms_task1=round(avg_dur1, 2),
            avg_segment_duration_ms_task2=round(avg_dur2, 2),
            total_segments_task1=count1,
            total_segments_task2=count2,
            conclusion=conclusion
        )

    @staticmethod
    def compare_interruptions(result1: AudioAnalysisResult, result2: AudioAnalysisResult) -> InterruptionComparison:
        ints1 = StatisticsAnalyzer.detect_interruptions(result1.segments)
        ints2 = StatisticsAnalyzer.detect_interruptions(result2.segments)
        count1 = len(ints1)
        count2 = len(ints2)
        total1 = len(result1.segments)
        total2 = len(result2.segments)
        rate1 = count1 / total1 if total1 > 0 else 0.0
        rate2 = count2 / total2 if total2 > 0 else 0.0

        conclusion = StatisticsAnalyzer._compare_values(rate1, rate2, is_higher_better=True, absolute_threshold=0.05, use_percentage=False)

        return InterruptionComparison(
            interruption_rate_task1=round(rate1, 4),
            interruption_rate_task2=round(rate2, 4),
            interruption_count_task1=count1,
            interruption_count_task2=count2,
            total_segments_task1=total1,
            total_segments_task2=total2,
            conclusion=conclusion
        )

    @staticmethod
    def generate_comparison_report(result1: AudioAnalysisResult, result2: AudioAnalysisResult) -> ComparisonReport:
        speaker_dominance = StatisticsAnalyzer.compare_speaker_dominance(result1, result2)
        sentiment = StatisticsAnalyzer.compare_sentiment(result1, result2)
        activity = StatisticsAnalyzer.compare_activity(result1, result2)
        interruption = StatisticsAnalyzer.compare_interruptions(result1, result2)

        return ComparisonReport(
            task1_id=result1.task_id,
            task2_id=result2.task_id,
            task1_filename=result1.original_filename,
            task2_filename=result2.original_filename,
            speaker_dominance=speaker_dominance,
            sentiment=sentiment,
            activity=activity,
            interruption=interruption
        )
