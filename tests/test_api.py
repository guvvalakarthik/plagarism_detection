from fastapi.testclient import TestClient

from plagiarism_detection.api import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "2.0.0"}


def test_analyze_contract() -> None:
    text = "A production model needs monitoring, versioning, and evaluation evidence."
    response = client.post("/v1/analyze", json={"source": text, "candidate": text})

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "high_overlap"
    assert body["similarity_score"] == 1.0
    assert body["score_interpretation"].startswith("Similarity score")
    assert body["evidence"][0]["source_start"] == 0


def test_rejects_unknown_fields() -> None:
    text = "A sufficiently long piece of text for contract validation."
    response = client.post(
        "/v1/analyze",
        json={"source": text, "candidate": text, "probability": True},
    )

    assert response.status_code == 422


def test_rejects_oversized_documents() -> None:
    response = client.post(
        "/v1/analyze",
        json={"source": "word " * 20_001, "candidate": "three valid words"},
    )

    assert response.status_code == 422
