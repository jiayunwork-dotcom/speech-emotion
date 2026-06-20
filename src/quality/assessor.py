import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional

from src.core.config import settings
from src.core.models import (
    QualityAssessment,
    QualityDimensionScore,
    QualitySuggestion,
    QualityGrade,
    AudioMetaInfo,
    QualityHistoryRecord,
    QualityTrend,
    OverallTrend,
    DimensionTrend,
    DimensionTrendDetail
)
from src.audio.processor import AudioProcessor


class QualityAssessor:
    SNR_WEIGHT = 0.40
    CLIPPING_WEIGHT = 0.20
    SPEECH_RATIO_WEIGHT = 0.25
    SAMPLE_RATE_WEIGHT = 0.15

    @staticmethod
    def _linear_interpolate(value: float, low: float, high: float,
                            low_score: float, high_score: float) -> float:
        if value <= low:
            return low_score
        if value >= high:
            return high_score
        ratio = (value - low) / (high - low)
        return low_score + ratio * (high_score - low_score)

    @staticmethod
    def _calculate_segment_energy(y: np.ndarray, sr: int,
                                  start_ms: int, end_ms: int) -> float:
        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        start_sample = max(0, min(start_sample, len(y)))
        end_sample = max(0, min(end_sample, len(y)))
        if end_sample <= start_sample:
            return 0.0
        segment = y[start_sample:end_sample]
        if len(segment) == 0:
            return 0.0
        return float(np.mean(segment ** 2))

    @staticmethod
    def _evaluate_snr(y: np.ndarray, sr: int,
                      speech_segments: list,
                      silence_segments: list) -> Tuple[QualityDimensionScore, float]:
        speech_energies = []
        for seg in speech_segments:
            e = QualityAssessor._calculate_segment_energy(
                y, sr, seg.start_ms, seg.end_ms
            )
            if e > 0:
                speech_energies.append(e)

        silence_energies = []
        for seg in silence_segments:
            e = QualityAssessor._calculate_segment_energy(
                y, sr, seg.start_ms, seg.end_ms
            )
            if e > 0:
                silence_energies.append(e)

        if not speech_energies or not silence_energies:
            snr_db = 15.0
        else:
            avg_speech = np.mean(speech_energies)
            avg_silence = np.mean(silence_energies)
            if avg_silence <= 0:
                avg_silence = 1e-10
            ratio = avg_speech / avg_silence
            if ratio <= 0:
                ratio = 1e-10
            snr_db = 10 * np.log10(ratio)

        if snr_db < 10:
            score = QualityAssessor._linear_interpolate(snr_db, 0, 10, 0, 50)
        elif snr_db < 20:
            score = QualityAssessor._linear_interpolate(snr_db, 10, 20, 50, 100)
        else:
            score = 100.0

        score = float(max(0, min(100, score)))
        dim_score = QualityDimensionScore(
            name="信噪比(SNR)",
            score=score,
            raw_value=float(snr_db),
            unit="dB"
        )
        return dim_score, score

    @staticmethod
    def _evaluate_clipping(y: np.ndarray) -> Tuple[QualityDimensionScore, float]:
        if len(y) == 0:
            dim_score = QualityDimensionScore(
                name="削波检测",
                score=100.0,
                raw_value=0.0,
                unit="%"
            )
            return dim_score, 100.0

        threshold = 0.95
        clip_count = np.sum(np.abs(y) >= threshold)
        clip_ratio = clip_count / len(y) * 100

        if clip_ratio <= 0:
            score = 100.0
        elif clip_ratio >= 5:
            score = 0.0
        else:
            score = QualityAssessor._linear_interpolate(clip_ratio, 0, 5, 100, 0)

        score = float(max(0, min(100, score)))
        dim_score = QualityDimensionScore(
            name="削波检测",
            score=score,
            raw_value=float(clip_ratio),
            unit="%"
        )
        return dim_score, score

    @staticmethod
    def _evaluate_speech_ratio(speech_segments: list,
                               total_duration_ms: int) -> Tuple[QualityDimensionScore, float]:
        if total_duration_ms <= 0:
            dim_score = QualityDimensionScore(
                name="有效语音占比",
                score=0.0,
                raw_value=0.0,
                unit="%"
            )
            return dim_score, 0.0

        speech_duration = sum(
            seg.end_ms - seg.start_ms for seg in speech_segments
        )
        ratio = speech_duration / total_duration_ms * 100

        if ratio < 20:
            score = QualityAssessor._linear_interpolate(ratio, 0, 20, 0, 0)
        elif ratio > 80:
            score = 100.0
        else:
            score = QualityAssessor._linear_interpolate(ratio, 20, 80, 0, 100)

        score = float(max(0, min(100, score)))
        dim_score = QualityDimensionScore(
            name="有效语音占比",
            score=score,
            raw_value=float(ratio),
            unit="%"
        )
        return dim_score, score

    @staticmethod
    def _evaluate_sample_rate_fitness(original_sr: int) -> Tuple[QualityDimensionScore, float]:
        target_sr = settings.TARGET_SAMPLE_RATE

        if original_sr == target_sr:
            score = 100.0
        elif original_sr > target_sr:
            score = 80.0
        elif original_sr >= 8000:
            score = 40.0
        else:
            score = 0.0

        score = float(max(0, min(100, score)))
        dim_score = QualityDimensionScore(
            name="采样率适配度",
            score=score,
            raw_value=float(original_sr),
            unit="Hz"
        )
        return dim_score, score

    @staticmethod
    def _generate_suggestions(
        snr_dim: QualityDimensionScore,
        clipping_dim: QualityDimensionScore,
        speech_ratio_dim: QualityDimensionScore,
        sample_rate_dim: QualityDimensionScore,
        overall_score: float
    ) -> list:
        suggestions = []

        if snr_dim.score < 70:
            snr_val = f"{snr_dim.raw_value:.1f}dB" if snr_dim.raw_value is not None else "未知"
            suggestions.append(QualitySuggestion(
                dimension="信噪比(SNR)",
                problem=f"信噪比较低（当前 {snr_val}），背景噪声明显，可能影响情感识别准确率",
                suggestion="建议在安静环境下重新录制，或使用降噪软件（如Audacity的噪声消除、Adobe Audition降噪功能）预处理音频"
            ))

        if clipping_dim.score < 70:
            clip_val = f"{clipping_dim.raw_value:.2f}%" if clipping_dim.raw_value is not None else "未知"
            suggestions.append(QualitySuggestion(
                dimension="削波检测",
                problem=f"音频存在削波失真（削波占比 {clip_val}），可能导致语音特征丢失",
                suggestion="建议重新录制时降低麦克风增益或减小声源距离，避免音量过载；也可使用音频修复工具尝试还原削波部分"
            ))

        if speech_ratio_dim.score < 70:
            ratio_val = f"{speech_ratio_dim.raw_value:.1f}%" if speech_ratio_dim.raw_value is not None else "未知"
            suggestions.append(QualitySuggestion(
                dimension="有效语音占比",
                problem=f"有效语音占比较低（当前 {ratio_val}），大部分内容为静音或非语音",
                suggestion="建议裁剪掉开头和结尾的长时间静音段，或重新录制包含更多有效语音内容的音频"
            ))

        if sample_rate_dim.score < 70:
            sr_val = f"{int(sample_rate_dim.raw_value)}Hz" if sample_rate_dim.raw_value is not None else "未知"
            suggestions.append(QualitySuggestion(
                dimension="采样率适配度",
                problem=f"原始采样率为 {sr_val}，与目标采样率16kHz不匹配，重采样会引入信息损失或失真",
                suggestion="建议使用支持16kHz采样率的录音设备重新录制，若无法重录，可尝试使用专业重采样算法（如SoX高音质模式）进行预处理"
            ))

        if overall_score < 50:
            suggestions.append(QualitySuggestion(
                dimension="综合评估",
                problem="音频整体质量较差，分析结果的可靠程度较低",
                suggestion="强烈建议按照上述各维度的建议改善音频质量后重新进行分析，以获得更准确的结果"
            ))

        return suggestions

    @staticmethod
    def _determine_grade(overall_score: float) -> QualityGrade:
        if overall_score >= 90:
            return QualityGrade.EXCELLENT
        elif overall_score >= 70:
            return QualityGrade.GOOD
        elif overall_score >= 50:
            return QualityGrade.FAIR
        else:
            return QualityGrade.POOR

    @staticmethod
    def assess(task_id: str,
               y: np.ndarray, sr: int,
               original_sample_rate: int,
               total_duration_ms: int) -> QualityAssessment:
        speech_segments, silence_segments = AudioProcessor.vad_split_with_silence(y, sr)

        snr_dim, snr_score = QualityAssessor._evaluate_snr(
            y, sr, speech_segments, silence_segments
        )
        clipping_dim, clipping_score = QualityAssessor._evaluate_clipping(y)
        speech_ratio_dim, speech_ratio_score = QualityAssessor._evaluate_speech_ratio(
            speech_segments, total_duration_ms
        )
        sample_rate_dim, sample_rate_score = QualityAssessor._evaluate_sample_rate_fitness(
            original_sample_rate
        )

        overall_score = (
            snr_score * QualityAssessor.SNR_WEIGHT
            + clipping_score * QualityAssessor.CLIPPING_WEIGHT
            + speech_ratio_score * QualityAssessor.SPEECH_RATIO_WEIGHT
            + sample_rate_score * QualityAssessor.SAMPLE_RATE_WEIGHT
        )
        overall_score = float(max(0, min(100, overall_score)))

        grade = QualityAssessor._determine_grade(overall_score)

        suggestions = QualityAssessor._generate_suggestions(
            snr_dim, clipping_dim, speech_ratio_dim, sample_rate_dim, overall_score
        )

        return QualityAssessment(
            task_id=task_id,
            snr=snr_dim,
            clipping=clipping_dim,
            speech_ratio=speech_ratio_dim,
            sample_rate_fitness=sample_rate_dim,
            overall_score=overall_score,
            grade=grade,
            suggestions=suggestions
        )

    @staticmethod
    def save_meta_info(meta_info: AudioMetaInfo) -> Path:
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = settings.PROCESSED_DATA_DIR / f"{meta_info.task_id}_meta.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta_info.model_dump(), f, ensure_ascii=False, indent=2, default=str)
        return path

    @staticmethod
    def load_meta_info(task_id: str) -> Optional[AudioMetaInfo]:
        path = settings.PROCESSED_DATA_DIR / f"{task_id}_meta.json"
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return AudioMetaInfo(**data)

    @staticmethod
    def save_assessment(assessment: QualityAssessment) -> Path:
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = settings.PROCESSED_DATA_DIR / f"{assessment.task_id}_quality.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(assessment.model_dump(), f, ensure_ascii=False, indent=2, default=str)
        return path

    @staticmethod
    def load_assessment(task_id: str) -> Optional[QualityAssessment]:
        path = settings.PROCESSED_DATA_DIR / f"{task_id}_quality.json"
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return QualityAssessment(**data)

    @staticmethod
    def _get_history_path() -> Path:
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return settings.PROCESSED_DATA_DIR / "quality_history.json"

    @staticmethod
    def load_history() -> list:
        path = QualityAssessor._get_history_path()
        if not path.exists():
            return []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = []
        for item in data:
            try:
                records.append(QualityHistoryRecord(**item))
            except Exception:
                continue
        return records

    @staticmethod
    def save_history(records: list) -> None:
        path = QualityAssessor._get_history_path()
        records_data = [r.model_dump(mode='json') for r in records]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records_data, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def append_history_record(assessment: QualityAssessment, filename: str) -> QualityHistoryRecord:
        history = QualityAssessor.load_history()
        record = QualityHistoryRecord(
            task_id=assessment.task_id,
            filename=filename,
            assessed_at=assessment.created_at,
            overall_score=assessment.overall_score,
            snr_score=assessment.snr.score,
            clipping_score=assessment.clipping.score,
            speech_ratio_score=assessment.speech_ratio.score,
            sample_rate_fitness_score=assessment.sample_rate_fitness.score
        )
        history.append(record)
        QualityAssessor.save_history(history)
        return record

    @staticmethod
    def _compute_dimension_trend(
        dimension_name: str,
        scores: list
    ) -> DimensionTrendDetail:
        if len(scores) < 3:
            return DimensionTrendDetail(
                dimension=dimension_name,
                trend=DimensionTrend.STABLE,
                latest_score=scores[-1] if scores else 0.0,
                previous_avg=0.0,
                diff=0.0
            )
        latest = scores[-1]
        previous_avg = (scores[-2] + scores[-3]) / 2
        diff = latest - previous_avg
        if diff > 15:
            trend = DimensionTrend.RISING
        elif diff < -15:
            trend = DimensionTrend.DECLINING
        else:
            trend = DimensionTrend.STABLE
        return DimensionTrendDetail(
            dimension=dimension_name,
            trend=trend,
            latest_score=latest,
            previous_avg=previous_avg,
            diff=diff
        )

    @staticmethod
    def compute_trend() -> QualityTrend:
        history = QualityAssessor.load_history()
        if len(history) < 3:
            return QualityTrend(
                overall_trend=OverallTrend.INSUFFICIENT,
                has_decline=False,
                history=history,
                warnings=[]
            )

        overall_scores = [r.overall_score for r in history]
        latest = overall_scores[-1]
        ma3 = (overall_scores[-1] + overall_scores[-2] + overall_scores[-3]) / 3
        diff = latest - ma3
        has_decline = diff < -10

        if diff > 10:
            overall_trend = OverallTrend.RISING
        elif diff < -10:
            overall_trend = OverallTrend.DECLINING
        else:
            overall_trend = OverallTrend.STABLE

        snr_scores = [r.snr_score for r in history]
        clipping_scores = [r.clipping_score for r in history]
        speech_ratio_scores = [r.speech_ratio_score for r in history]
        sample_rate_scores = [r.sample_rate_fitness_score for r in history]

        dimension_trends = [
            QualityAssessor._compute_dimension_trend("信噪比(SNR)", snr_scores),
            QualityAssessor._compute_dimension_trend("削波检测", clipping_scores),
            QualityAssessor._compute_dimension_trend("有效语音占比", speech_ratio_scores),
            QualityAssessor._compute_dimension_trend("采样率适配度", sample_rate_scores),
        ]

        warnings = []
        if has_decline:
            warnings.append(
                f"近期上传音频质量呈下降趋势(当前评分{latest:.1f}分,近3次平均{ma3:.1f}分),建议检查录音设备或环境是否发生变化"
            )

        dim_map = {
            "信噪比(SNR)": snr_scores,
            "削波检测": clipping_scores,
            "有效语音占比": speech_ratio_scores,
            "采样率适配度": sample_rate_scores,
        }
        for dim_name, scores in dim_map.items():
            if len(scores) >= 2 and scores[-1] < 50 and scores[-2] < 50:
                warnings.append(f"{dim_name}连续处于低分状态,建议重点关注")

        return QualityTrend(
            overall_trend=overall_trend,
            has_decline=has_decline,
            moving_average=ma3,
            latest_score=latest,
            diff_from_avg=diff,
            dimension_trends=dimension_trends,
            history=history,
            warnings=warnings
        )

    @staticmethod
    def apply_trend_suggestions(assessment: QualityAssessment) -> QualityAssessment:
        trend = QualityAssessor.compute_trend()
        for warning in trend.warnings:
            is_overall = "下降趋势" in warning
            assessment.suggestions.append(QualitySuggestion(
                dimension="综合评估" if is_overall else "维度监测",
                problem=warning,
                suggestion="请持续关注音频质量变化，如问题持续建议排查录音设备、环境噪声或录制方式"
            ))
        return assessment

    @staticmethod
    def assess_with_trend(
        task_id: str,
        y: np.ndarray, sr: int,
        original_sample_rate: int,
        total_duration_ms: int,
        filename: str
    ) -> Tuple[QualityAssessment, QualityTrend]:
        assessment = QualityAssessor.assess(
            task_id, y, sr, original_sample_rate, total_duration_ms
        )
        QualityAssessor.append_history_record(assessment, filename)
        trend = QualityAssessor.compute_trend()
        assessment = QualityAssessor.apply_trend_suggestions(assessment)
        return assessment, trend
