from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

app = FastAPI(title='Gateway Service')

ROUTES = {
    'audio': os.getenv('AUDIO_STORAGE_URL', 'http://audio-storage:8001'),
    'filters': os.getenv('FILTER_REGISTRY_URL', 'http://filter-registry:8002'),
    'processing': os.getenv('PROCESSING_URL', 'http://processing:8003'),
    'optimization': os.getenv('OPTIMIZATION_URL', 'http://optimization:8004'),
    'dataset': os.getenv('DATASET_URL', 'http://dataset-evaluation:8005'),
    'speech': os.getenv('SPEECH_URL', 'http://speech-recognition:8006'),
    'config': os.getenv('CONFIG_URL', 'http://config-service:8007'),
}


@app.get('/health')
def health():
    return {'status': 'ok', 'services': ROUTES}


@app.api_route('/{service}/{path:path}', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
async def proxy(service: str, path: str, request: Request):
    if service not in ROUTES:
        raise HTTPException(status_code=404, detail='unknown service prefix')
    body = await request.body()
    target = f"{ROUTES[service]}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() != 'host'}
    async with httpx.AsyncClient(timeout=60) as client:
        upstream = await client.request(request.method, target, params=request.query_params, content=body, headers=headers)
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=upstream.headers.get('content-type'))
