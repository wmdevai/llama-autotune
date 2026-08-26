"""Tests for the Web UI HTTP and SSE endpoints."""

import json
import logging
import queue

from fastapi.testclient import TestClient

from llama_autotune import web


def _client() -> TestClient:
    return TestClient(web.app)


def test_optimize_requires_model_path():
    client = _client()
    response = client.post("/api/optimize", json={"model_path": "   "})
    assert response.status_code == 400


def test_optimize_unknown_job_events_404():
    client = _client()
    response = client.get("/api/optimize/events/does-not-exist")
    assert response.status_code == 404


def test_queue_log_handler_forwards_records():
    sink: queue.Queue = queue.Queue()
    handler = web._QueueLogHandler(sink)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger = logging.getLogger("test_queue_handler")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    try:
        logger.info("hello world")
        event = sink.get(timeout=1)
        assert event["type"] == "log"
        assert event["message"] == "hello world"
    finally:
        logger.removeHandler(handler)


def test_optimize_streams_progress_and_result(monkeypatch, tmp_path):
    fake_model = tmp_path / "model.gguf"
    fake_model.write_bytes(b"GGUF")

    opt_logger = logging.getLogger("llama_autotune.optimizer")
    opt_logger.setLevel(logging.INFO)

    def fake_run(request):
        opt_logger.info("stage A complete")
        return {
            "objective": "balanced",
            "best_config": {"threads": 4},
            "best_score": 1.5,
            "total_evaluations": 1,
            "baseline_result": None,
            "final_result": {
                "success": True,
                "generation_tps": 10.0,
            },
        }

    monkeypatch.setattr(web, "_run_optimization", fake_run)

    client = _client()

    response = client.post(
        "/api/optimize",
        json={
            "model_path": str(fake_model),
            "trials_b": 1,
            "trials_c": 1,
        },
    )
    assert response.status_code == 200

    job_id = response.json()["job_id"]

    events = []

    with client.stream(
        "GET",
        f"/api/optimize/events/{job_id}",
    ) as stream_response:
        assert stream_response.status_code == 200

        for line in stream_response.iter_lines():
            if line.startswith("data: "):
                events.append(
                    json.loads(line[len("data: "):])
                )

    assert any(event["type"] == "log" for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["best_score"] == 1.5
    assert events[-1]["error"] is None
