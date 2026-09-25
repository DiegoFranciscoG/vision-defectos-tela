import base64
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from fabric_inspection.api.app import create_app
from fabric_inspection.config import ApiSettings
from fabric_inspection.inference.inspector import FabricInspector
from fabric_inspection.registry import ModelIntegrityError, ModelRegistry
from tests.integration.conftest import API_KEY, SIZE, api_settings, png_bytes
from tests.synthetic import make_image


def _memory_image() -> Image.Image:
    """First image of the tiny detector's memory bank (same RNG as the fixture)."""
    return make_image(np.random.default_rng(1), SIZE, None)[0]


def _good_png() -> bytes:
    return png_bytes(_memory_image())


def _hole_png() -> bytes:
    image = np.asarray(_memory_image()).copy()
    image[20:44, 20:44] = 0
    return png_bytes(Image.fromarray(image))


class TestSecurity:
    def test_health_is_public_and_has_security_headers(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert "default-src 'none'" in response.headers["content-security-policy"]
        assert "access-control-allow-origin" not in response.headers

    @pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key-that-is-long-enough-000000"}])
    def test_api_routes_deny_by_default(self, client: TestClient, headers: dict[str, str]) -> None:
        for method, url in (
            ("get", "/api/v1/predictions"),
            ("get", "/api/v1/rolls"),
            ("get", "/api/v1/models/current"),
            ("get", "/api/v1/monitoring/drift"),
        ):
            response = getattr(client, method)(url, headers=headers)
            assert response.status_code == 401, url
            assert response.headers["www-authenticate"] == "ApiKey"

    def test_docs_are_served_with_their_own_csp(self, client: TestClient) -> None:
        response = client.get("/docs")
        assert response.status_code == 200
        assert "cdn.jsdelivr.net" in response.headers["content-security-policy"]

    def test_rate_limit_returns_429(
        self, tiny_models: tuple[ModelRegistry, Path], api_database: str
    ) -> None:
        _, model_dir = tiny_models
        settings = api_settings(api_database, model_dir, rate_limit_per_minute=3)
        app = create_app(settings, FabricInspector.from_settings(settings))
        with TestClient(app) as limited:
            codes = [
                limited.get("/api/v1/models/current", headers={"X-API-Key": API_KEY}).status_code
                for _ in range(4)
            ]
        assert codes == [200, 200, 200, 429]

    def test_body_over_the_limit_is_rejected_before_parsing(
        self, client: TestClient, auth: dict[str, str]
    ) -> None:
        big = b"0" * (6 * 1024 * 1024)
        response = client.post(
            "/api/v1/predictions", headers=auth, files={"file": ("x.png", big, "image/png")}
        )
        assert response.status_code == 413

    def test_settings_refuse_weak_keys_and_wildcard_cors(self) -> None:
        with pytest.raises(ValidationError, match="at least 32"):
            ApiSettings(api_keys="short")  # type: ignore[arg-type]
        with pytest.raises(ValidationError, match="explicit origins"):
            ApiSettings(api_keys=API_KEY, cors_allowed_origins="*")  # type: ignore[arg-type]

    def test_tampered_model_is_not_loaded(
        self, tiny_models: tuple[ModelRegistry, Path], tmp_path: Path
    ) -> None:
        registry, model_dir = tiny_models
        tampered = registry.model_copy(deep=True)
        tampered.detector.sha256 = "0" * 64
        tampered.save(tmp_path / "registry.json")
        settings = api_settings("sqlite://", model_dir, model_registry=tmp_path / "registry.json")
        with pytest.raises(ModelIntegrityError):
            FabricInspector.from_settings(settings.model_copy(update={"model_base_url": None}))


class TestPredictions:
    def test_good_and_defective_images(self, client: TestClient, auth: dict[str, str]) -> None:
        good = client.post(
            "/api/v1/predictions", headers=auth, files={"file": ("a.png", _good_png(), "image/png")}
        )
        assert good.status_code == 201, good.text
        body = good.json()
        assert body["is_defective"] is False
        assert body["defects"] == []
        assert body["latency_ms"] > 0
        assert "inference;dur=" in good.headers["server-timing"]

        hole = client.post(
            "/api/v1/predictions?include_heatmap=true",
            headers=auth,
            files={"file": ("b.png", _hole_png(), "image/png")},
            data={"roll_code": "RL-2026-0001", "position_m": "12.5"},
        )
        assert hole.status_code == 201, hole.text
        body = hole.json()
        assert body["is_defective"] is True
        assert body["score_ratio"] >= 1
        assert body["defects"]
        assert all(1 <= defect["points"] <= 4 for defect in body["defects"])
        assert body["roll_code"] == "RL-2026-0001"
        assert base64.b64decode(body["heatmap_png_base64"])[:4] == b"\x89PNG"
        assert base64.b64decode(body["cam_png_base64"])[:4] == b"\x89PNG"

        page = client.get("/api/v1/predictions?limit=2", headers=auth).json()
        assert page["total"] >= 202  # 200 seeded + 2 uploaded
        assert page["items"][0]["id"] == body["id"]

    def test_jpeg_is_accepted(self, client: TestClient, auth: dict[str, str]) -> None:
        jpeg = png_bytes(make_image(np.random.default_rng(3), SIZE, None)[0], fmt="JPEG")
        response = client.post(
            "/api/v1/predictions", headers=auth, files={"file": ("c.jpg", jpeg, "image/jpeg")}
        )
        assert response.status_code == 201

    @pytest.mark.parametrize(
        ("payload", "status"),
        [
            (b"not an image at all", 415),
            (b"", 415),
            (png_bytes(Image.new("RGB", (8, 8))), 415),
            (png_bytes(Image.new("RGB", (5000, 40))), 413),
            (png_bytes(Image.new("RGB", (64, 64)), fmt="GIF"), 415),
        ],
    )
    def test_invalid_uploads(
        self, client: TestClient, auth: dict[str, str], payload: bytes, status: int
    ) -> None:
        response = client.post(
            "/api/v1/predictions", headers=auth, files={"file": ("x.png", payload, "image/png")}
        )
        assert response.status_code == status, response.text
        assert "Traceback" not in response.text

    @pytest.mark.parametrize(
        ("data", "status"),
        [
            ({"roll_code": "RL-2026-0001"}, 422),
            ({"position_m": "3"}, 422),
            ({"roll_code": "RL-NOPE", "position_m": "3"}, 404),
            ({"roll_code": "RL-2026-0001", "position_m": "9999"}, 422),
            ({"roll_code": "bad code!", "position_m": "3"}, 422),
        ],
    )
    def test_roll_rules(
        self, client: TestClient, auth: dict[str, str], data: dict[str, str], status: int
    ) -> None:
        response = client.post(
            "/api/v1/predictions",
            headers=auth,
            files={"file": ("a.png", _good_png(), "image/png")},
            data=data,
        )
        assert response.status_code == status, response.text


class TestQualityAndOperations:
    def test_roll_endpoints(self, client: TestClient, auth: dict[str, str]) -> None:
        rolls = client.get("/api/v1/rolls", headers=auth).json()
        assert len(rolls) == 10
        quality = client.get("/api/v1/rolls/RL-2026-0006/quality", headers=auth).json()
        assert set(quality["textrack_payload"]) == {"inspectedLengthM", "widthCm", "defects"}
        assessment = client.post("/api/v1/rolls/RL-2026-0006/assessments", headers=auth)
        assert assessment.status_code == 201
        assert client.get("/api/v1/rolls/RL-NOPE/quality", headers=auth).status_code == 404
        alerts = client.get("/api/v1/quality/alerts", headers=auth)
        assert alerts.status_code == 200

    def test_aql_and_lot_inspection(self, client: TestClient, auth: dict[str, str]) -> None:
        plan = client.get(
            "/api/v1/quality/aql-plan?lot_size=500&aql=2.5&inspection_level=II", headers=auth
        ).json()
        assert (plan["plan_letter"], plan["sample_size"], plan["accept_number"]) == ("H", 50, 3)
        bad = client.get("/api/v1/quality/aql-plan?lot_size=500&aql=0.01", headers=auth)
        assert bad.status_code == 422
        lot = client.post(
            "/api/v1/lots/LT-2026-02/inspections",
            headers=auth,
            json={"aql": "2.5", "inspection_level": "II"},
        )
        assert lot.status_code == 201
        assert lot.json()["decision"] in {"ACCEPTED", "REJECTED"}

    def test_drift_models_and_savings(self, client: TestClient, auth: dict[str, str]) -> None:
        drift = client.get("/api/v1/monitoring/drift?window=100", headers=auth)
        assert drift.status_code == 200
        assert drift.json()["n_window"] == 100
        report = client.post("/api/v1/monitoring/drift-reports?window=100", headers=auth)
        assert report.status_code == 201
        models = client.get("/api/v1/models/current", headers=auth).json()
        assert models["weights_license"] == "CC-BY-NC-SA-4.0"
        assert len(models["detector"]["sha256"]) == 64
        savings = client.post(
            "/api/v1/savings/estimate", headers=auth, json={"meters_per_month": 20000}
        )
        assert savings.status_code == 200
        assert savings.json()["assumptions"]
        invalid = client.post(
            "/api/v1/savings/estimate", headers=auth, json={"manual_recall": 3, "extra": 1}
        )
        assert invalid.status_code == 422
        assert '"input"' not in invalid.text
