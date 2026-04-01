from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class PipelineFilter(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    rawId: str
    referenceId: Optional[str] = None
    pipeline: List[PipelineFilter]
    segment: Optional[Segment] = None


class TaskStatus(BaseModel):
    taskId: str
    status: Literal['running', 'done', 'failed']
    processedId: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class OptimizeRequest(BaseModel):
    rawId: str
    referenceId: str
    pipeline: List[PipelineFilter]
    segment: Optional[Segment] = None


class SearchRequest(BaseModel):
    rawId: str
    referenceId: str
    filterPool: List[str]
    maxDepth: int = Field(ge=1, le=6)


class DatasetEvaluateRequest(BaseModel):
    rawFolderId: str
    refFolderId: str
    pipeline: List[PipelineFilter]


class WerEvaluateRequest(BaseModel):
    audioId: str
    transcriptFolderId: str
    segment: Optional[Segment] = None


class SavePipelineRequest(BaseModel):
    name: str
    pipeline: List[PipelineFilter]
