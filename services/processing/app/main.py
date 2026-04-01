from __future__ import annotations

import math
import uuid
from typing import Callable

import numpy as np
from fastapi import FastAPI, HTTPException
from scipy.signal import butter, filtfilt, medfilt, wiener

from services.common.contracts import ProcessRequest, TaskStatus

app = FastAPI(title='Processing Service')
TASKS: dict[str, TaskStatus] = {}


class FilterPlugin:
    registry: dict[str, Callable[[np.ndarray, int, dict], np.ndarray]] = {}

    @classmethod
    def register(cls, name: str):
        def deco(fn):
            cls.registry[name] = fn
            return fn
        return deco


@FilterPlugin.register('lowpass')
def lowpass(data: np.ndarray, sr: int, params: dict) -> np.ndarray:
    cutoff = float(params.get('cutoff_hz', 3400))
    order = int(params.get('order', 4))
    b, a = butter(order, cutoff / (sr / 2), btype='low')
    return filtfilt(b, a, data)


@FilterPlugin.register('highpass')
def highpass(data: np.ndarray, sr: int, params: dict) -> np.ndarray:
    cutoff = float(params.get('cutoff_hz', 120))
    order = int(params.get('order', 2))
    b, a = butter(order, cutoff / (sr / 2), btype='high')
    return filtfilt(b, a, data)


@FilterPlugin.register('median')
def median_filter(data: np.ndarray, sr: int, params: dict) -> np.ndarray:
    return medfilt(data, kernel_size=int(params.get('kernel', 7)) | 1)


@FilterPlugin.register('wiener')
def wiener_filter(data: np.ndarray, sr: int, params: dict) -> np.ndarray:
    return wiener(data, mysize=int(params.get('mysize', 9)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    err = y_true - y_pred
    mse = float(np.mean(np.square(err)))
    snr = 10 * math.log10((np.mean(y_true ** 2) + eps) / (mse + eps))
    seg = 10 * math.log10((np.mean(np.abs(y_true)) + eps) / (np.mean(np.abs(err)) + eps))
    lsd = float(np.sqrt(np.mean((20 * np.log10(np.abs(np.fft.rfft(y_true)) + eps) - 20 * np.log10(np.abs(np.fft.rfft(y_pred)) + eps)) ** 2)))
    return {'mse': mse, 'snr': snr, 'segmental_snr': seg, 'lsd': lsd}


@app.post('/process')
def process(req: ProcessRequest):
    # demo signal path; real implementation reads from storage and broker queue
    task_id = str(uuid.uuid4())
    TASKS[task_id] = TaskStatus(taskId=task_id, status='running')
    sr = 16000
    t = np.linspace(0, 3, 3 * sr, endpoint=False)
    raw = np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.randn(t.shape[0])
    ref = np.sin(2 * np.pi * 220 * t)
    if req.segment:
        raw = raw[int(req.segment.start * sr): int(req.segment.end * sr)]
        ref = ref[int(req.segment.start * sr): int(req.segment.end * sr)]
    processed = raw.copy()
    for step in req.pipeline:
        fn = FilterPlugin.registry.get(step.name)
        if not fn:
            TASKS[task_id] = TaskStatus(taskId=task_id, status='failed', error=f'Unknown filter: {step.name}')
            return TASKS[task_id]
        processed = fn(processed, sr, step.params)
    TASKS[task_id] = TaskStatus(taskId=task_id, status='done', processedId=f'processed-{task_id}', metrics=_metrics(ref, processed))
    return TASKS[task_id]


@app.get('/process/{task_id}')
def task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail='task not found')
    return TASKS[task_id]


@app.get('/process/{task_id}/audio')
def processed_audio(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail='task not found')
    return {'processedId': TASKS[task_id].processedId, 'message': 'Connect to Audio Storage Service in production'}
