from fastapi.testclient import TestClient

from plagiarism_detection.api import app

client = TestClient(app)


def test_dashboard_exposes_compare_and_workspace_review_modes() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert 'id="analysis-form"' in response.text
    assert 'id="workspace-panel"' in response.text
    assert 'id="upload-form"' in response.text
    assert 'id="search-form"' in response.text
    assert 'id="workspace-results"' in response.text
    assert "/static/dashboard.css" in response.text


def test_dashboard_assets_use_workspace_workflow_contract() -> None:
    script = client.get("/static/app.js")
    stylesheet = client.get("/static/dashboard.css")

    assert script.status_code == 200
    assert stylesheet.status_code == 200
    assert 'workspaceRequest("/documents"' in script.text
    assert 'workspaceRequest("/search"' in script.text
    assert 'workspaceRequest("/feedback"' in script.text
    assert "innerHTML" not in script.text
    assert ".review-layout" in stylesheet.text
