from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
import soundfile as sf


def decode_audio_upload(data: bytes) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y.astype(np.float32), int(sr)


def encode_wav_base64(y: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")
