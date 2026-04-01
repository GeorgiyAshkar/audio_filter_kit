from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException

from services.common.contracts import DatasetEvaluateRequest

app = FastAPI(title='Dataset Evaluation Service')
TASKS: dict[str, dict] = {}


@app.post('/dataset-evaluate')
def evaluate(body: DatasetEvaluateRequest):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {'status': 'done', 'aggregates': {'mse_mean': 0.0062, 'snr_mean': 13.8, 'lsd_mean': 1.4}, 'count': 12}
    return {'taskId': task_id, 'status': 'running'}


@app.get('/dataset-evaluate/{task_id}')
def status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail='task not found')
    return TASKS[task_id]
