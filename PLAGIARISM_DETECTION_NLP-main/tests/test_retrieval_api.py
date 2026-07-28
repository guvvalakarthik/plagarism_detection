from fastapi.testclient import TestClient

from plagiarism_detection.api import app

client = TestClient(app)


def test_index_and_search_corpus() -> None:
    index_response = client.post(
        "/v1/corpus/index",
        json={
            "documents": [
                {
                    "document_id": "ml-guide",
                    "text": (
                        "Production machine learning requires monitoring, "
                        "representative evaluation, and clear ownership."
                    ),
                },
                {
                    "document_id": "gardening",
                    "text": (
                        "Healthy garden soil requires compost, sunlight, "
                        "and careful watering throughout the season."
                    ),
                },
            ]
        },
    )

    assert index_response.status_code == 200
    assert index_response.json()["documents"] == 2
    assert index_response.json()["passages"] == 2

    search_response = client.post(
        "/v1/corpus/search",
        json={"query": "machine learning evaluation", "top_k": 1},
    )

    assert search_response.status_code == 200
    assert search_response.json()["hits"][0]["document_id"] == "ml-guide"
