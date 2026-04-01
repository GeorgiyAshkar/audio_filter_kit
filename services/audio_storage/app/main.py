from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

BASE = Path(os.getenv('AUDIO_STORAGE_PATH', '/tmp/das-audio'))
BASE.mkdir(parents=True, exist_ok=True)
META: dict[str, dict] = {}

app = FastAPI(title='Audio Storage Service')


def _save(upload: UploadFile, kind: str):
    item_id = str(uuid.uuid4())
    suffix = Path(upload.filename or '.wav').suffix or '.wav'
    file_path = BASE / f'{item_id}{suffix}'
    with file_path.open('wb') as f:
        shutil.copyfileobj(upload.file, f)
    META[item_id] = {'id': item_id, 'kind': kind, 'filename': upload.filename or file_path.name, 'path': str(file_path), 'duration': None}
    return META[item_id]


@app.post('/audio/raw')
def upload_raw(file: UploadFile = File(...)):
    return _save(file, 'raw')


@app.post('/audio/reference')
def upload_reference(file: UploadFile = File(...)):
    return _save(file, 'reference')


@app.get('/audio/{item_id}')
def get_audio(item_id: str):
    item = META.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail='audio not found')
    return FileResponse(path=item['path'], filename=item['filename'])
