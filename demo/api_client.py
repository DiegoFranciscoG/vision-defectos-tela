"""Server-side client of the inspection API. The API key never reaches the browser."""

import os
from typing import Any

import httpx
import streamlit as st

TIMEOUT = httpx.Timeout(120.0, connect=20.0)


def _setting(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except FileNotFoundError:  # no secrets.toml: fall back to environment variables
        value = None
    return str(value) if value else os.environ.get(name)


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@st.cache_resource
def get_client() -> httpx.Client:
    base_url = _setting("API_BASE_URL") or "http://localhost:8000"
    api_key = _setting("API_KEY")
    if not api_key:
        st.error("Falta API_KEY (Streamlit secrets o variable de entorno).")
        st.stop()
    return httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=TIMEOUT)


def _check(response: httpx.Response) -> Any:
    if response.is_success:
        return response.json()
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise ApiError(response.status_code, str(detail))


def get(path: str, **params: Any) -> Any:
    return _check(get_client().get(path, params=params))


def post(path: str, *, json: Any = None, **params: Any) -> Any:
    return _check(get_client().post(path, json=json, params=params))


def predict(image: bytes, filename: str, roll_code: str | None, position_m: float | None) -> Any:
    data = {}
    if roll_code:
        data = {"roll_code": roll_code, "position_m": str(position_m or 0)}
    response = get_client().post(
        "/api/v1/predictions",
        params={"include_heatmap": "true"},
        files={"file": (filename, image, "application/octet-stream")},
        data=data,
    )
    return _check(response)


def wake_up() -> dict[str, Any]:
    """Render free sleeps after 15 minutes; the first request can take about a minute."""
    return _check(get_client().get("/health"))  # type: ignore[no-any-return]
