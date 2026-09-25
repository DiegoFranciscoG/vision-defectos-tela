"""Streamlit demo: inspect an image, grade rolls, watch drift and estimate savings.

Run: streamlit run demo/app.py (needs API_BASE_URL and API_KEY in secrets or environment).
"""

import base64
import json
from pathlib import Path

import api_client as api
import pandas as pd
import streamlit as st

SAMPLES = Path(__file__).parent / "samples"
DECISION_ICON = {"ACCEPTED": "🟢 Aceptado", "WARNING": "🟡 Advertencia", "REJECTED": "🔴 Rechazado"}

st.set_page_config(page_title="Inspección de tela", page_icon="🧵", layout="wide")


def _png(data: str | None) -> bytes | None:
    return base64.b64decode(data) if data else None


def _error(error: api.ApiError) -> None:
    st.error(f"La API respondió {error.status}: {error.detail}")


def inspection_page() -> None:
    st.header("Inspección de una imagen")
    st.caption(
        "PatchCore decide si hay defecto y dónde; ResNet-18 nombra el tipo. La longitud se "
        "convierte a milímetros con la escala de cámara (supuesto S1) y a puntos ASTM D5430."
    )
    samples = sorted(SAMPLES.glob("*.jpg"))
    left, right = st.columns([1, 1])
    with left:
        source = st.radio("Imagen", ["Muestra MVTec AD", "Subir archivo"], horizontal=True)
        image_bytes, filename = None, "upload.png"
        if source == "Muestra MVTec AD" and samples:
            choice = st.selectbox("Muestra", samples, format_func=lambda p: p.stem)
            image_bytes, filename = choice.read_bytes(), choice.name
        else:
            upload = st.file_uploader("PNG o JPEG (máx. 5 MB)", type=["png", "jpg", "jpeg"])
            if upload is not None:
                image_bytes, filename = upload.getvalue(), upload.name
        attach = st.checkbox("Asociar a un rollo (sistema de 4 puntos)")
        roll_code, position = None, None
        if attach:
            try:
                rolls = api.get("/api/v1/rolls")
            except api.ApiError as error:
                _error(error)
                return
            roll = st.selectbox(
                "Rollo", rolls, format_func=lambda r: f"{r['code']} · {r['length_m']} m"
            )
            roll_code = roll["code"]
            position = st.number_input("Posición (m)", 0.0, float(roll["length_m"]), 1.0, 0.5)
        run = st.button("Inspeccionar", type="primary", disabled=image_bytes is None)
    if image_bytes is not None:
        right.image(image_bytes, caption="Imagen original", use_container_width=True)
    if not run or image_bytes is None:
        return
    with st.spinner("Analizando…"):
        try:
            result = api.predict(image_bytes, filename, roll_code, position)
        except api.ApiError as error:
            _error(error)
            return
    verdict = "🔴 Defecto" if result["is_defective"] else "🟢 Sin defecto"
    cols = st.columns(4)
    cols[0].metric("Decisión", verdict)
    cols[1].metric("Puntaje / umbral", f"{result['score_ratio']:.2f}")
    cols[2].metric("Tipo", result["predicted_class"] or "—")
    cols[3].metric("Latencia (ms)", f"{result['latency_ms']:.0f}")
    images = st.columns(2)
    images[0].image(
        _png(result["heatmap_png_base64"]),
        caption="Mapa de anomalía (PatchCore) y regiones",
        use_container_width=True,
    )
    images[1].image(
        _png(result["cam_png_base64"]),
        caption="CAM del clasificador (= Grad-CAM)",
        use_container_width=True,
    )
    if result["defects"]:
        st.subheader("Defectos y puntos ASTM D5430")
        st.dataframe(pd.DataFrame(result["defects"]), hide_index=True, use_container_width=True)
    with st.expander("Probabilidades por clase y detalle"):
        st.bar_chart(pd.Series(result["class_probabilities"]))
        st.json({k: result[k] for k in ("image_sha256", "score", "threshold", "models")})


def rolls_page() -> None:
    st.header("Rollos · sistema de 4 puntos y AQL")
    try:
        rolls = api.get("/api/v1/rolls")
    except api.ApiError as error:
        _error(error)
        return
    table = pd.DataFrame(rolls)
    table["decision"] = table["decision"].map(DECISION_ICON)
    st.dataframe(table, hide_index=True, use_container_width=True)
    code = st.selectbox("Detalle del rollo", [r["code"] for r in rolls])
    quality = api.get(f"/api/v1/rolls/{code}/quality")
    cols = st.columns(4)
    cols[0].metric("Decisión", DECISION_ICON[quality["decision"]])
    cols[1].metric("Puntos (tope 4/yarda)", quality["total_points"])
    cols[2].metric("Puntos / 100 yd²", quality["points_per_100_sq_yd"])
    cols[3].metric("Límite", quality["max_points_allowed"])
    st.dataframe(pd.DataFrame(quality["defects"]), hide_index=True, use_container_width=True)
    if st.button("Registrar evaluación y alerta"):
        assessment = api.post(f"/api/v1/rolls/{code}/assessments")
        st.success("Alerta registrada" if assessment["alert"] else "Rollo aceptado: sin alerta")
    with st.expander("Payload para textrack (POST /fabric-rolls/{id}/inspection)"):
        st.code(json.dumps(quality["textrack_payload"], indent=2), language="json")

    st.subheader("Aceptación del lote (ISO 2859-1, muestreo simple normal)")
    lots = sorted({r["lot_code"] for r in rolls})
    lot_cols = st.columns(3)
    lot = lot_cols[0].selectbox("Lote", lots)
    aql = lot_cols[1].selectbox("AQL", ["0.65", "1.0", "1.5", "2.5", "4.0", "6.5"], index=3)
    level = lot_cols[2].selectbox("Nivel", ["I", "II", "III", "S-1", "S-2", "S-3", "S-4"], index=1)
    if st.button("Evaluar lote"):
        try:
            result = api.post(
                f"/api/v1/lots/{lot}/inspections", json={"aql": aql, "inspection_level": level}
            )
        except api.ApiError as error:
            _error(error)
            return
        st.metric("Decisión del lote", DECISION_ICON[result["decision"]])
        st.write(
            f"Letra **{result['code_letter']}** · muestra n = **{result['sample_size']}** · "
            f"Ac = {result['accept_number']} / Re = {result['reject_number']} · "
            f"rollos rechazados en la muestra: **{result['rejected_rolls']}**"
        )
        st.dataframe(pd.DataFrame(result["sampled_rolls"]), hide_index=True)


def monitoring_page() -> None:
    st.header("Monitoreo de drift")
    st.caption(
        "KS de dos muestras entre los puntajes de tela normal recientes y la referencia de "
        "validación (defectos reales excluidos), más la tasa de alertas."
    )
    window = st.slider("Ventana (predicciones)", 30, 1000, 200, 10)
    try:
        drift = api.get("/api/v1/monitoring/drift", window=window)
    except api.ApiError as error:
        _error(error)
        return
    cols = st.columns(4)
    cols[0].metric("Estado", "⚠️ Drift" if drift["drift_detected"] else "✅ Estable")
    cols[1].metric("p-valor KS", "—" if drift["p_value"] is None else f"{drift['p_value']:.3f}")
    cols[2].metric(
        "Tasa de alertas", "—" if drift["alert_rate"] is None else f"{drift['alert_rate']:.1%}"
    )
    cols[3].metric("Ventana", drift["n_window"])
    st.caption(f"Motivo: {drift['reason']}")
    scores = pd.concat(
        [
            pd.DataFrame({"score": drift["window_scores"], "serie": "ventana"}),
            pd.DataFrame({"score": drift["reference_scores"], "serie": "referencia (val)"}),
        ]
    )
    st.vega_lite_chart(
        scores,
        {
            "mark": {"type": "bar", "opacity": 0.6},
            "encoding": {
                "x": {"field": "score", "bin": {"maxbins": 40}, "title": "Puntaje de anomalía"},
                "y": {"aggregate": "count", "stack": None, "title": "Imágenes"},
                "color": {"field": "serie", "type": "nominal"},
            },
        },
        use_container_width=True,
    )
    recent = api.get("/api/v1/predictions", limit=20)
    st.subheader("Últimas predicciones")
    st.dataframe(
        pd.DataFrame(recent["items"])[
            [
                "id",
                "created_at",
                "roll_code",
                "position_m",
                "score_ratio",
                "is_defective",
                "predicted_class",
                "latency_ms",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )


def savings_page() -> None:
    st.header("Ahorro estimado frente a inspección manual")
    st.caption("Todos los parámetros son editables; los supuestos están documentados (S7-S9).")
    cols = st.columns(3)
    body = {
        "meters_per_month": cols[0].number_input(
            "Metros inspeccionados al mes", 1000, 2_000_000, 50_000, 1000
        ),
        "manual_speed_m_min": cols[1].slider("Velocidad manual (m/min)", 8, 20, 15),
        "automated_speed_m_min": cols[2].slider("Velocidad automática (m/min)", 10, 120, 30),
        "defects_per_100_m": cols[0].slider("Defectos por 100 m", 0.0, 10.0, 2.0, 0.1),
        "manual_recall": cols[1].slider("Detección del inspector (S7)", 0.3, 1.0, 0.7, 0.05),
        "price_per_meter_usd": cols[2].number_input("Precio por metro (USD)", 0.5, 50.0, 4.0, 0.5),
        "hourly_cost_usd": cols[0].number_input("Costo hora inspector (USD)", 0.0, 50.0, 2.01, 0.1),
        "price_reduction": cols[1].slider("Pérdida de precio por defecto", 0.45, 0.65, 0.45, 0.01),
    }
    try:
        result = api.post("/api/v1/savings/estimate", json=body)
    except api.ApiError as error:
        _error(error)
        return
    metrics = st.columns(4)
    metrics[0].metric("Ahorro neto / mes", f"USD {result['net_saving_usd']:,.0f}")
    metrics[1].metric(
        "Metros defectuosos que ya no escapan", f"{result['escaped_meters_avoided']:,.0f} m"
    )
    metrics[2].metric(
        "Horas de inspección",
        f"{result['automated_hours']:.0f} h",
        f"{result['automated_hours'] - result['manual_hours']:.0f} h",
    )
    metrics[3].metric("Falsas alarmas / mes", f"{result['false_alarms_per_month']:.0f}")
    st.write(
        f"Recall del modelo usado: **{result['inputs']['model_recall']:.0%}** (medido en test)."
    )
    with st.expander("Supuestos"):
        for item in result["assumptions"]:
            st.write(f"- {item}")


def model_page() -> None:
    st.header("Modelos servidos")
    info = api.get("/api/v1/models/current")
    st.write(
        f"Paquete **{info['bundle_version']}** · pesos bajo "
        f"**{info['weights_license']}** (uso no comercial)"
    )
    for key in ("detector", "classifier"):
        model = info[key]
        st.subheader(model["name"])
        st.write(
            f"Archivo `{model['file']}` · {model['size_bytes'] / 1e6:.1f} MB · "
            f"{'INT8' if model['quantized'] else 'FP32'} · SHA-256 `{model['sha256'][:16]}…`"
        )
        if model["threshold"] is not None:
            st.write(f"Umbral {model['threshold']:.4f} ({model['threshold_policy']})")
        st.json(model["test_metrics"], expanded=False)


with st.spinner("Conectando con la API (Render free puede tardar ~1 min en despertar)…"):
    try:
        health = api.wake_up()
    except Exception as error:
        st.error(f"No se pudo conectar con la API: {error}")
        st.stop()
st.sidebar.success(f"API {health['status']} · {health['detector_version']}")
st.sidebar.caption("Imágenes de ejemplo: MVTec AD (CC BY-NC-SA 4.0). Datos de rollos: ficticios.")
pages = st.navigation(
    [
        st.Page(inspection_page, title="Inspección", icon="🔍", default=True),
        st.Page(rolls_page, title="Rollos y lotes", icon="🧵", url_path="rolls"),
        st.Page(monitoring_page, title="Monitoreo", icon="📈", url_path="monitoring"),
        st.Page(savings_page, title="Ahorro", icon="💰", url_path="savings"),
        st.Page(model_page, title="Modelo", icon="🧠", url_path="model"),
    ]
)
pages.run()
