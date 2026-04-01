from fastapi.testclient import TestClient

from services.filter_registry.app.main import app as filter_app
from services.processing.app.main import app as processing_app


def test_filter_list_and_validation():
    c = TestClient(filter_app)
    r = c.get('/filters')
    assert r.status_code == 200
    assert any(x['name'] == 'lowpass' for x in r.json())
    bad = c.post('/validate-pipeline', json={'pipeline': [{'name': 'unknown', 'params': {}}]})
    assert bad.status_code == 200
    assert bad.json()['valid'] is False


def test_processing_pipeline_done():
    c = TestClient(processing_app)
    r = c.post('/process', json={'rawId': 'raw1', 'referenceId': 'ref1', 'pipeline': [{'name': 'wiener', 'params': {'mysize': 7}}], 'segment': {'start': 0, 'end': 1}})
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'done'
    assert 'metrics' in body
