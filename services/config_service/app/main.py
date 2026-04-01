from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException

from services.common.contracts import SavePipelineRequest

app = FastAPI(title='Config Service')
PIPELINES: dict[str, dict] = {}


@app.post('/pipelines')
def save_pipeline(body: SavePipelineRequest):
    cfg_id = str(uuid.uuid4())
    PIPELINES[cfg_id] = body.model_dump()
    return {'id': cfg_id, **PIPELINES[cfg_id]}


@app.get('/pipelines/{cfg_id}')
def load(cfg_id: str):
    if cfg_id not in PIPELINES:
        raise HTTPException(status_code=404, detail='config not found')
    return {'id': cfg_id, **PIPELINES[cfg_id]}
