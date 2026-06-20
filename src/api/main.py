from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from typing import List
import os
from pathlib import Path

from src.core.config import settings
from src.core.models import (
    TaskResponse,
    TaskStatusResponse,
    TaskStatus,
    BatchTaskResponse,
    BatchStatusResponse,
    AudioAnalysisResult,
    StatisticsSummary,
    ComparisonReport,
    TaskListItem,
    QualityAssessment,
    QualityTrend
)
from src.tasks.manager import task_manager
from src.stats.analyzer import StatisticsAnalyzer
from src.output.formatter import OutputFormatter
from src.quality.assessor import QualityAssessor
from src.audio.processor import AudioProcessor

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="语音情感识别与说话人分离后端分析平台",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Speech Emotion Recognition API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload", response_model=TaskResponse)
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.wav', '.mp3']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Only WAV and MP3 are supported."
        )
    
    content = await file.read()
    
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB"
        )
    
    try:
        task_id, status = task_manager.create_task(file.filename, content)
        return TaskResponse(
            task_id=task_id,
            status=status,
            message="Task created successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/task/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task_info = task_manager.get_task_status(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info.status,
        progress=task_info.progress,
        error_message=task_info.error_message
    )


@app.get("/task/{task_id}/result", response_model=AudioAnalysisResult)
async def get_task_result(task_id: str):
    result = task_manager.get_task_result(task_id)
    if not result:
        task_info = task_manager.get_task_status(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="Task not found")
        if task_info.status != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Task not completed. Current status: {task_info.status}"
            )
        raise HTTPException(status_code=404, detail="Result not found")
    
    return result


@app.get("/task/{task_id}/summary", response_model=StatisticsSummary)
async def get_task_summary(task_id: str):
    result = task_manager.get_task_result(task_id)
    if not result:
        task_info = task_manager.get_task_status(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="Task not found")
        if task_info.status != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Task not completed. Current status: {task_info.status}"
            )
        raise HTTPException(status_code=404, detail="Result not found")
    
    summary = StatisticsAnalyzer.generate_summary(result)
    return summary


@app.get("/task/{task_id}/srt")
async def get_task_srt(task_id: str):
    result = task_manager.get_task_result(task_id)
    if not result:
        task_info = task_manager.get_task_status(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="Task not found")
        if task_info.status != TaskStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Task not completed. Current status: {task_info.status}"
            )
        raise HTTPException(status_code=404, detail="Result not found")
    
    srt_content = OutputFormatter.to_srt(result)
    return Response(content=srt_content, media_type="text/plain")


@app.get("/task/{task_id}/audio")
async def get_task_audio(task_id: str):
    audio_path = settings.PROCESSED_DATA_DIR / f"{task_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=f"{task_id}.wav"
    )


@app.post("/batch/upload", response_model=BatchTaskResponse)
async def upload_batch(files: List[UploadFile] = File(...)):
    if len(files) > settings.BATCH_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.BATCH_MAX_FILES} files per batch"
        )
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    file_contents = []
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File without filename")
        
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.wav', '.mp3']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file.filename}. Only WAV and MP3 are supported."
            )
        
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        file_contents.append((file.filename, content))
    
    try:
        batch_id, task_results = task_manager.create_batch(file_contents)
        
        tasks = [
            TaskResponse(
                task_id=tid,
                status=status,
                message="Task created successfully" if status == TaskStatus.PENDING else "Task creation failed"
            )
            for tid, status in task_results
        ]
        
        return BatchTaskResponse(batch_id=batch_id, tasks=tasks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/batch/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    batch_info = task_manager.get_batch_status(batch_id)
    if not batch_info:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    task_statuses = task_manager.get_batch_task_statuses(batch_id)
    
    tasks = [
        TaskStatusResponse(
            task_id=ts.task_id,
            status=ts.status,
            progress=ts.progress,
            error_message=ts.error_message
        )
        for ts in task_statuses
    ]
    
    return BatchStatusResponse(batch_id=batch_id, tasks=tasks)


@app.get("/tasks", response_model=list[TaskListItem])
async def list_tasks(status: TaskStatus = None):
    all_tasks = task_manager.get_all_tasks()
    result = []
    for t in all_tasks:
        if status is None or t.status == status:
            created_at = None
            if t.result:
                created_at = t.result.created_at
            result.append(TaskListItem(
                task_id=t.task_id,
                filename=t.filename,
                status=t.status,
                created_at=created_at
            ))
    return result


@app.get("/compare", response_model=ComparisonReport)
async def compare_tasks(task1_id: str, task2_id: str):
    task1_info = task_manager.get_task_status(task1_id)
    task2_info = task_manager.get_task_status(task2_id)

    if not task1_info:
        raise HTTPException(status_code=404, detail=f"Task {task1_id} not found")
    if not task2_info:
        raise HTTPException(status_code=404, detail=f"Task {task2_id} not found")

    if task1_info.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task1_id} is not completed. Current status: {task1_info.status}"
        )
    if task2_info.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task {task2_id} is not completed. Current status: {task2_info.status}"
        )

    result1 = task_manager.get_task_result(task1_id)
    result2 = task_manager.get_task_result(task2_id)

    if not result1:
        raise HTTPException(status_code=404, detail=f"Result for task {task1_id} not found")
    if not result2:
        raise HTTPException(status_code=404, detail=f"Result for task {task2_id} not found")

    report = StatisticsAnalyzer.generate_comparison_report(result1, result2)
    return report


@app.get("/task/{task_id}/quality")
async def get_task_quality(task_id: str):
    task_info = task_manager.get_task_status(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    quality = task_manager.get_task_quality_assessment(task_id)
    trend = QualityAssessor.compute_trend()

    if quality:
        result = quality.model_dump()
        result["trend"] = trend.model_dump()
        return result

    if not task_manager.is_task_preprocessed(task_id):
        if task_info.status not in [TaskStatus.PROCESSING, TaskStatus.COMPLETED]:
            raise HTTPException(
                status_code=400,
                detail="Task preprocessing not yet started. Please wait for processing to begin."
            )
        raise HTTPException(
            status_code=400,
            detail=f"Task not yet preprocessed. Current status: {task_info.status}"
        )

    meta_info = task_manager.get_task_meta_info(task_id)
    if not meta_info:
        raise HTTPException(status_code=404, detail="Audio meta info not found")

    try:
        y, sr = AudioProcessor.load_processed_audio(task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load processed audio: {str(e)}")

    filename = task_manager.get_task_filename(task_id) or "unknown"

    quality, trend = QualityAssessor.assess_with_trend(
        task_id=task_id,
        y=y,
        sr=sr,
        original_sample_rate=meta_info.original_sample_rate,
        total_duration_ms=meta_info.processed_duration_ms,
        filename=filename
    )

    QualityAssessor.save_assessment(quality)

    result = quality.model_dump()
    result["trend"] = trend.model_dump()
    return result


@app.get("/quality/trend", response_model=QualityTrend)
async def get_quality_trend():
    return QualityAssessor.compute_trend()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
