from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from typing import Any, Dict, List

import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.signal import welch

from backend.audio_io import decode_audio_upload, encode_wav_base64
from backend.filters import (
    FilterConfig,
    default_pipeline_item,
    pipeline_from_payload,
    run_pipeline,
    serialize_filter_specs,
)
from backend.metrics import composite_score, compute_metrics

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/filters")
def api_filters():
    return jsonify({"filters": serialize_filter_specs()})


def _selection_slice(y: np.ndarray, sr: int, selection: Dict[str, Any] | None) -> np.ndarray:
    if not selection:
        return y
    start = float(selection.get("start", 0.0))
    end = float(selection.get("end", 0.0))
    if end <= start:
        return y
    s = max(0, min(len(y), int(start * sr)))
    e = max(s + 1, min(len(y), int(end * sr)))
    return y[s:e]


def _psd(y: np.ndarray, sr: int) -> Dict[str, List[float]]:
    if len(y) < 64:
        return {"freq": [], "pow": []}
    f, p = welch(y, fs=sr, nperseg=min(2048, len(y)))
    return {"freq": f.tolist(), "pow": (p + 1e-12).tolist()}


@app.post("/api/process")
def api_process():
    raw_file = request.files.get("raw")
    if raw_file is None:
        return jsonify({"error": "raw audio is required"}), 400

    raw, sr = decode_audio_upload(raw_file.read())
    ref_file = request.files.get("reference")
    ref = None
    ref_sr = sr
    if ref_file:
        ref, ref_sr = decode_audio_upload(ref_file.read())
        if ref_sr != sr:
            return jsonify({"error": "reference sample rate must match raw"}), 400

    pipeline_payload = json.loads(request.form.get("pipeline", "[]"))
    pipeline = pipeline_from_payload(pipeline_payload)
    selection = json.loads(request.form.get("selection", "null"))

    raw_selected = _selection_slice(raw, sr, selection)
    processed = run_pipeline(raw_selected, sr, pipeline) if pipeline else raw_selected

    metrics = None
    if ref is not None:
        metrics = compute_metrics(_selection_slice(ref, sr, selection), processed, sr)

    return jsonify(
        {
            "sr": sr,
            "processedWav": encode_wav_base64(processed, sr),
            "processedSignal": processed.tolist(),
            "rawSignal": raw.tolist(),
            "referenceSignal": ref.tolist() if ref is not None else [],
            "metrics": metrics,
            "latencyMs": len(pipeline) * 1.2,
            "spectra": {
                "raw": _psd(_selection_slice(raw, sr, selection), sr),
                "processed": _psd(processed, sr),
                "reference": _psd(_selection_slice(ref, sr, selection), sr) if ref is not None else {"freq": [], "pow": []},
            },
        }
    )


@app.post("/api/search")
def api_search():
    raw_file = request.files.get("raw")
    ref_file = request.files.get("reference")
    if raw_file is None or ref_file is None:
        return jsonify({"error": "raw and reference are required"}), 400
    raw, sr = decode_audio_upload(raw_file.read())
    ref, ref_sr = decode_audio_upload(ref_file.read())
    if sr != ref_sr:
        return jsonify({"error": "sample rates must match"}), 400

    payload = request.get_json(force=True, silent=True) or {}
    selected = payload.get("selectedFilters", [])
    max_depth = int(payload.get("maxDepth", 2))
    top_k = int(payload.get("topK", 5))
    max_eval = int(payload.get("maxEval", 100))

    eval_count = 0
    results = []
    for depth in range(1, max_depth + 1):
        for seq in itertools.product(selected, repeat=depth):
            spaces = [serialize_filter_specs()[name] for name in seq]
            param_grid = []
            for spec in spaces:
                variants = []
                for k, (v, pmin, pmax, step) in spec.items():
                    if isinstance(v, float):
                        vals = [float(pmin), float(v), float(pmax)]
                    else:
                        vals = sorted({int(pmin), int(v), int(pmax)})
                    variants.append([(k, x) for x in vals])
                merged = []
                for combo in itertools.product(*variants):
                    merged.append(dict(combo))
                param_grid.append(merged)
            for params_combo in itertools.product(*param_grid):
                pipeline = [FilterConfig(name=n, params=p) for n, p in zip(seq, params_combo)]
                est = run_pipeline(raw, sr, pipeline)
                m = compute_metrics(ref, est, sr)
                results.append({"pipeline": [asdict(p) for p in pipeline], "desc": " -> ".join(seq), "metrics": m, "score": composite_score(m)})
                eval_count += 1
                if eval_count >= max_eval:
                    break
            if eval_count >= max_eval:
                break
        if eval_count >= max_eval:
            break

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"results": results[:top_k], "evaluated": eval_count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
