from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException

from services.common.contracts import WerEvaluateRequest

app = FastAPI(title='Speech Recognition Service')
TASKS: dict[str, dict] = {}


@app.post('/wer-evaluate')
def wer_eval(body: WerEvaluateRequest):
    task_id = str(uuid.uuid4())
    TASKS[task_id] = {'status': 'done', 'wer': 0.12, 'asrText': 'test decoded speech', 'referenceText': 'test decoded speech'}
    return {'taskId': task_id}


@app.get('/wer-evaluate/{task_id}')
def status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail='task not found')
    return TASKS[task_id]
