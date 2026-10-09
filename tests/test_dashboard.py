"""Tests for the Flask dashboard endpoints."""

from __future__ import annotations

import io

import pytest

pytest.importorskip("flask")
pytest.importorskip("plotly")

from automl import AutoMLResult, create_app  # noqa: E402
from automl.data_upload import SAMPLE_DATASET_FILENAME  # noqa: E402


@pytest.fixture
def client(tmp_path, iris_result: AutoMLResult):
    app = create_app(iris_result, upload_dir=tmp_path / "uploads")
    app.config["TESTING"] = True
    return app.test_client()


def test_start_page_and_dashboard_render(client) -> None:
    assert client.get("/").status_code == 200
    response = client.get("/?view=dashboard")
    assert response.status_code == 200
    assert b"/vendor/plotly.min.js" in response.data


def test_plotly_bundle_is_served(client) -> None:
    response = client.get("/vendor/plotly.min.js")
    assert response.status_code == 200
    assert response.mimetype == "application/javascript"


@pytest.mark.parametrize("endpoint", ["/api/charts/leaderboard", "/api/charts/metrics", "/api/charts/resources"])
def test_chart_endpoints_return_figures(client, endpoint: str) -> None:
    payload = client.get(endpoint).get_json()
    assert "error" not in payload
    assert payload["data"]


def test_sample_dataset_is_listed(client) -> None:
    names = [item["name"] for item in client.get("/api/datasets").get_json()["datasets"]]
    assert SAMPLE_DATASET_FILENAME in names


def test_upload_does_not_overwrite_existing_file(client) -> None:
    csv = b"a,b,target\n" + b"".join(f"{i},{i * 2},{i % 2}\n".encode() for i in range(20))

    first = client.post("/api/upload", data={"file": (io.BytesIO(csv), "data.csv")})
    second = client.post("/api/upload", data={"file": (io.BytesIO(csv), "data.csv")})

    assert first.get_json()["filename"] == "data.csv"
    assert second.get_json()["filename"] == "data_1.csv"


def test_run_on_regression_upload_with_default_settings(client) -> None:
    rows = "".join(f"{i},{(i * 7) % 13},{i * 3.5 + 1.25}\n" for i in range(60))
    csv = ("x1,x2,price\n" + rows).encode()
    client.post("/api/upload", data={"file": (io.BytesIO(csv), "houses.csv")})

    response = client.post(
        "/api/run",
        json={"dataset": "houses.csv", "target": "price", "constraints": {"max_training_time": 10}},
    )

    body = response.get_json()
    assert response.status_code == 200, body
    assert body["best_result"]["primary_metric"] == "rmse"


def test_settings_file_is_not_a_dataset(client) -> None:
    client.get("/")  # settings.json is written on startup
    names = [item["name"] for item in client.get("/api/datasets").get_json()["datasets"]]
    assert "settings.json" not in names
    assert client.delete("/api/datasets/settings.json").status_code == 404


def test_results_page_shows_selected_model_and_leaderboard(client, iris_result: AutoMLResult) -> None:
    page = client.get("/?view=dashboard").get_data(as_text=True)
    assert 'id="leaderboard"' in page
    assert iris_result.best_result is not None
    assert "Selected model" in page
