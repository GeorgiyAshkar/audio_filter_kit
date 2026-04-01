from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title='Filter Registry Service')

FILTERS = {
    'lowpass': {
        'description': 'Butterworth low-pass filter',
        'params': {'cutoff_hz': {'type': 'number', 'min': 50, 'max': 8000, 'default': 3400}, 'order': {'type': 'integer', 'min': 1, 'max': 8, 'default': 4}},
    },
    'highpass': {
        'description': 'Butterworth high-pass filter',
        'params': {'cutoff_hz': {'type': 'number', 'min': 20, 'max': 4000, 'default': 120}, 'order': {'type': 'integer', 'min': 1, 'max': 8, 'default': 2}},
    },
    'bandpass': {
        'description': 'Band-pass filter',
        'params': {'low_hz': {'type': 'number', 'min': 20, 'max': 7000, 'default': 200}, 'high_hz': {'type': 'number', 'min': 50, 'max': 8000, 'default': 3400}},
    },
    'wiener': {'description': 'Wiener denoising', 'params': {'mysize': {'type': 'integer', 'min': 3, 'max': 31, 'default': 9}}},
    'moving_average': {'description': 'Moving average smoothing', 'params': {'window': {'type': 'integer', 'min': 3, 'max': 201, 'default': 11}}},
    'nlms': {'description': 'Adaptive NLMS filter', 'params': {'mu': {'type': 'number', 'min': 0.001, 'max': 1.0, 'default': 0.05}, 'taps': {'type': 'integer', 'min': 4, 'max': 256, 'default': 32}}},
    'rls': {'description': 'Adaptive RLS filter', 'params': {'lambda': {'type': 'number', 'min': 0.9, 'max': 0.9999, 'default': 0.99}, 'taps': {'type': 'integer', 'min': 4, 'max': 256, 'default': 32}}},
    'spectral_subtraction': {'description': 'Spectral subtraction', 'params': {'alpha': {'type': 'number', 'min': 0.1, 'max': 6.0, 'default': 2.0}}},
    'energy_gate': {'description': 'Energy threshold gate', 'params': {'threshold_db': {'type': 'number', 'min': -80, 'max': 0, 'default': -40}}},
    'wavelet_denoise': {'description': 'Wavelet denoising', 'params': {'level': {'type': 'integer', 'min': 1, 'max': 8, 'default': 3}}},
    'kalman': {'description': 'Kalman-based denoising', 'params': {'process_noise': {'type': 'number', 'min': 1e-7, 'max': 1.0, 'default': 1e-3}}},
}

class ValidateBody(BaseModel):
    pipeline: list[dict]


@app.get('/filters')
def get_filters():
    return [{'name': k, **v} for k, v in FILTERS.items()]


@app.get('/filters/{name}')
def get_filter(name: str):
    if name not in FILTERS:
        raise HTTPException(status_code=404, detail='Filter not found')
    return {'name': name, **FILTERS[name]}


@app.post('/validate-pipeline')
def validate_pipeline(body: ValidateBody):
    issues: list[str] = []
    for i, item in enumerate(body.pipeline):
        name = item.get('name')
        if name not in FILTERS:
            issues.append(f'#{i}: unknown filter {name}')
            continue
        for param in item.get('params', {}):
            if param not in FILTERS[name]['params']:
                issues.append(f'#{i}: invalid param {param} for {name}')
    return {'valid': len(issues) == 0, 'issues': issues}
