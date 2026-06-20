import asyncio
import uuid
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import threading
import logging

from src.core.config import settings
from src.core.models import TaskStatus, AudioAnalysisResult, AudioMetaInfo, QualityAssessment
from src.audio.processor import AudioProcessor
from src.speaker.separation import SpeakerSeparator
from src.emotion.recognizer import EmotionRecognizer
from src.output.formatter import OutputFormatter
from src.quality.assessor import QualityAssessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskInfo:
    def __init__(self, task_id: str, filename: str, file_path: Path):
        self.task_id = task_id
        self.filename = filename
        self.file_path = file_path
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.error_message: Optional[str] = None
        self.result: Optional[AudioAnalysisResult] = None
        self.batch_id: Optional[str] = None
        self.meta_info: Optional[AudioMetaInfo] = None
        self.quality_assessment: Optional[QualityAssessment] = None
        self.preprocessed: bool = False


class BatchInfo:
    def __init__(self, batch_id: str, task_ids: List[str]):
        self.batch_id = batch_id
        self.task_ids = task_ids


class TaskManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._tasks: Dict[str, TaskInfo] = {}
        self._batches: Dict[str, BatchInfo] = {}
        self._task_queue: deque = deque()
        self._active_tasks: set = set()
        self._max_parallel = settings.BATCH_MAX_PARALLEL
        self._emotion_recognizer = EmotionRecognizer()
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._initialized = True
        self._start_worker()

    def _start_worker(self):
        self._processing_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._processing_thread.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                while len(self._active_tasks) < self._max_parallel and self._task_queue:
                    task_id = self._task_queue.popleft()
                    if task_id in self._tasks and self._tasks[task_id].status == TaskStatus.PENDING:
                        self._active_tasks.add(task_id)
                        thread = threading.Thread(
                            target=self._process_task_wrapper,
                            args=(task_id,),
                            daemon=True
                        )
                        thread.start()
                
                threading.Event().wait(0.5)
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                threading.Event().wait(1.0)

    def _process_task_wrapper(self, task_id: str):
        try:
            asyncio.run(self._process_task(task_id))
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.FAILED
                self._tasks[task_id].error_message = str(e)
        finally:
            self._active_tasks.discard(task_id)

    async def _process_task(self, task_id: str):
        task_info = self._tasks.get(task_id)
        if not task_info:
            return

        try:
            task_info.status = TaskStatus.PROCESSING
            task_info.progress = 0.0
            logger.info(f"Starting task {task_id} for file {task_info.filename}")

            y_original, sr_original = AudioProcessor.load_audio(task_info.file_path)
            original_meta = AudioProcessor.get_original_meta_info(y_original, sr_original, task_id)
            task_info.progress = 0.1

            y, sr = AudioProcessor.convert_to_target_format(y_original, sr_original)
            duration_ms = AudioProcessor.get_audio_duration(y, sr)
            AudioProcessor.save_processed_audio(y, sr, task_id)

            meta_info = AudioMetaInfo(
                task_id=task_id,
                original_sample_rate=original_meta["original_sample_rate"],
                original_channels=original_meta["original_channels"],
                original_duration_ms=original_meta["original_duration_ms"],
                processed_sample_rate=sr,
                processed_channels=1,
                processed_duration_ms=duration_ms
            )
            task_info.meta_info = meta_info
            QualityAssessor.save_meta_info(meta_info)
            task_info.preprocessed = True
            task_info.progress = 0.20

            quality_assessment = QualityAssessor.assess(
                task_id=task_id,
                y=y,
                sr=sr,
                original_sample_rate=original_meta["original_sample_rate"],
                total_duration_ms=duration_ms
            )
            QualityAssessor.append_history_record(quality_assessment, task_info.filename)
            quality_assessment = QualityAssessor.apply_trend_suggestions(quality_assessment)
            task_info.quality_assessment = quality_assessment
            QualityAssessor.save_assessment(quality_assessment)
            task_info.progress = 0.28

            segments = AudioProcessor.vad_split(y, sr)
            task_info.progress = 0.4
            logger.info(f"Task {task_id}: VAD detected {len(segments)} segments")

            segments = SpeakerSeparator.assign_speaker_labels(segments, y, sr)
            task_info.progress = 0.65
            logger.info(f"Task {task_id}: Speaker separation completed")

            segments = self._emotion_recognizer.recognize_emotions(segments, y, sr)
            task_info.progress = 0.85
            logger.info(f"Task {task_id}: Emotion recognition completed")

            result = AudioAnalysisResult(
                task_id=task_id,
                original_filename=task_info.filename,
                duration_ms=duration_ms,
                sample_rate=sr,
                segments=segments,
                quality_assessment=quality_assessment
            )
            task_info.result = result

            json_path = settings.PROCESSED_DATA_DIR / f"{task_id}_result.json"
            OutputFormatter.save_json(result, json_path)

            srt_path = settings.PROCESSED_DATA_DIR / f"{task_id}_result.srt"
            OutputFormatter.save_srt(result, srt_path)

            task_info.progress = 1.0
            task_info.status = TaskStatus.COMPLETED
            logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task_id} error: {e}")
            task_info.status = TaskStatus.FAILED
            task_info.error_message = str(e)
            raise

    def create_task(self, filename: str, file_content: bytes) -> Tuple[str, TaskStatus]:
        task_id = str(uuid.uuid4())
        settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        file_path = settings.RAW_DATA_DIR / f"{task_id}_{filename}"
        
        with open(file_path, 'wb') as f:
            f.write(file_content)

        valid, msg = AudioProcessor.validate_audio(file_path)
        if not valid:
            file_path.unlink(missing_ok=True)
            raise ValueError(msg)

        task_info = TaskInfo(task_id, filename, file_path)
        self._tasks[task_id] = task_info
        self._task_queue.append(task_id)
        
        return task_id, TaskStatus.PENDING

    def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    def get_task_result(self, task_id: str) -> Optional[AudioAnalysisResult]:
        task_info = self._tasks.get(task_id)
        if task_info and task_info.status == TaskStatus.COMPLETED:
            return task_info.result
        return None

    def get_task_quality_assessment(self, task_id: str) -> Optional[QualityAssessment]:
        task_info = self._tasks.get(task_id)
        if task_info and task_info.quality_assessment:
            return task_info.quality_assessment
        cached = QualityAssessor.load_assessment(task_id)
        if cached:
            if task_info:
                task_info.quality_assessment = cached
            return cached
        return None

    def get_task_meta_info(self, task_id: str) -> Optional[AudioMetaInfo]:
        task_info = self._tasks.get(task_id)
        if task_info and task_info.meta_info:
            return task_info.meta_info
        cached = QualityAssessor.load_meta_info(task_id)
        if cached:
            if task_info:
                task_info.meta_info = cached
            return cached
        return None

    def is_task_preprocessed(self, task_id: str) -> bool:
        task_info = self._tasks.get(task_id)
        if task_info and task_info.preprocessed:
            return True
        cached_meta = QualityAssessor.load_meta_info(task_id)
        if cached_meta and task_info:
            task_info.preprocessed = True
            return True
        return cached_meta is not None

    def create_batch(self, files: List[Tuple[str, bytes]]) -> Tuple[str, List[Tuple[str, TaskStatus]]]:
        if len(files) > settings.BATCH_MAX_FILES:
            raise ValueError(f"Maximum {settings.BATCH_MAX_FILES} files per batch")

        batch_id = str(uuid.uuid4())
        task_results = []
        
        for filename, content in files:
            try:
                task_id, status = self.create_task(filename, content)
                self._tasks[task_id].batch_id = batch_id
                task_results.append((task_id, status))
            except Exception as e:
                task_results.append((str(uuid.uuid4()), TaskStatus.FAILED))

        task_ids = [tr[0] for tr in task_results]
        self._batches[batch_id] = BatchInfo(batch_id, task_ids)
        
        return batch_id, task_results

    def get_batch_status(self, batch_id: str) -> Optional[BatchInfo]:
        return self._batches.get(batch_id)

    def get_batch_task_statuses(self, batch_id: str) -> List[TaskInfo]:
        batch_info = self._batches.get(batch_id)
        if not batch_info:
            return []
        return [self._tasks.get(tid) for tid in batch_info.task_ids if self._tasks.get(tid)]

    def get_all_tasks(self) -> List[TaskInfo]:
        return list(self._tasks.values())

    def get_task_filename(self, task_id: str) -> Optional[str]:
        task_info = self._tasks.get(task_id)
        if task_info:
            return task_info.filename
        return None


task_manager = TaskManager()
