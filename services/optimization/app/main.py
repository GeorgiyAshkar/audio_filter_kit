from __future__ import annotations

import random
import uuid

from fastapi import FastAPI, HTTPException

from services.common.contracts import OptimizeRequest, SearchRequest

app = FastAPI(title='Optimization Service')
TASKS: dict[str, dict] = {}


@app.post('/optimize')
def optimize(body: OptimizeRequest):
    task_id = str(uuid.uuid4())
    best = {'pipeline': [p.model_dump() for p in body.pipeline], 'metrics': {'mse': round(random.uniform(0.0001, 0.01), 6), 'snr': round(random.uniform(8, 24), 2)}}
    TASKS[task_id] = {'status': 'done', 'best': best, 'progress': 100}
    return {'taskId': task_id, 'status': 'running'}


@app.post('/search')
def search(body: SearchRequest):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {'status': 'done', 'best': {'sequence': body.filterPool[:body.maxDepth], 'metrics': {'mse': 0.0041, 'snr': 15.1}}, 'progress': 100}
    return {'taskId': task_id}


@app.get('/optimize/{task_id}')
def get_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail='task not found')
    return TASKS[task_id]
