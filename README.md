# vision-defectos-tela

Detecta defectos en tela a partir de una foto: dice si hay defecto, dónde está y qué tipo es, y convierte esa detección en la decisión de calidad que usa la industria textil (sistema de 4 puntos ASTM D5430 por rollo y muestreo AQL ISO 2859-1 por lote). Incluye entrenamiento reproducible, API de inferencia con ONNX Runtime y monitoreo de drift.

![CI](https://github.com/DiegoFranciscoG/vision-defectos-tela/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12-3776ab) ![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20(CPU)-ee4c2c) ![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.30-005ced) ![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688) ![Streamlit](https://img.shields.io/badge/Streamlit-1.64-ff4b4b) ![MLflow](https://img.shields.io/badge/MLflow-3.16-0194e2) ![License](https://img.shields.io/badge/c%C3%B3digo-MIT-blue) ![Weights](https://img.shields.io/badge/pesos-CC%20BY--NC--SA%204.0-lightgrey)

**Demo en vivo:** se publica en Streamlit Community Cloud (demo) + Render (API) + Neon (BD) siguiendo [Despliegue](#despliegue-gratis). Mientras tanto, `docker compose up --build` levanta exactamente la misma demo en local.
**Acceso de prueba:** la API usa `X-API-Key`. Las claves no se publican: se generan con el comando de [Variables de entorno](#variables-de-entorno) y la demo las envía desde el servidor, nunca desde el navegador.

![Recorrido por la demo](docs/img/demo.gif)

| Inspección (PatchCore + CAM) | Rollos · 4 puntos | Lote · ISO 2859-1 |
|---|---|---|
| ![Inspección](docs/img/inspection.png) | ![Rollos](docs/img/rolls.png) | ![Lote](docs/img/lot.png) |
| **Monitoreo de drift** | **Ahorro frente a inspección manual** | **API documentada (Swagger)** |
| ![Monitoreo](docs/img/monitoring.png) | ![Ahorro](docs/img/savings.png) | ![Swagger](docs/img/swagger.png) |

## Problema que resuelve
En una tejeduría o confección la tela se revisa a mano: se desenrolla en una mesa iluminada a **8–20 m/min** y el inspector para el motor en cada defecto ([Kumar, 2008](docs/investigacion.md)). Es lento, depende del cansancio de la persona y un defecto que llega al cliente le baja el precio a esa tela un **45–65 %**. Además, casi nunca hay suficientes fotos de defectos para entrenar un clasificador clásico. Este proyecto aprende cómo se ve la tela **sin** defecto, marca cualquier desviación, la mide en milímetros y aplica las mismas reglas que el comprador usa para aceptar o rechazar un rollo o un envío. Se integra con [textrack](https://github.com/DiegoFranciscoG/textrack) (MES de confección): la API produce el payload exacto de su endpoint de inspección de rollos.

## Funcionalidades
- **Detección y localización con PatchCore** (Roth et al., CVPR 2022): se entrena solo con tela buena y genera un mapa de anomalía con las regiones del defecto.
- **Baseline supervisado documentado** (ResNet-18 afinada con las 6 clases), evaluado sobre **el mismo test**: la comparación es honesta.
- **Explicabilidad**: mapa de PatchCore y **CAM = Grad-CAM** del clasificador. En una red GAP + lineal son iguales salvo una constante (Selvaraju et al., §3.1), así que el mapa se exporta como salida del ONNX y está cubierto por un test contra autograd.
- **Decisión de calidad**:
  - puntos ASTM D5430 por defecto, con tope de 4 por yarda lineal;
  - puntos/100 yd² y aceptación del rollo;
  - plan de muestreo ISO 2859-1 (Tablas I y II-A) para aceptar el lote;
  - alertas de calidad.
- **Pipeline reproducible con un comando**: descarga con SHA-256 → manifiesto y split sin fuga → entrenamiento → evaluación → ONNX + INT8, con semillas fijas y trazado en MLflow.
- **API FastAPI + ONNX Runtime (CPU)**:
  - validación de imagen por contenido y latencia medida (`Server-Timing`);
  - registro de cada predicción (solo el hash de la imagen);
  - drift de puntajes y estimación de ahorro.
- **Demo Streamlit**: subes o eliges una imagen y ves el mapa, la máscara, el puntaje, el tipo y los puntos. También simula rollos, evalúa lotes, muestra el drift y calcula el ahorro.

## Resultados (test compartido: 61 imágenes buenas + 35 con defecto)
Las métricas salen de [`reports/metrics.json`](reports/metrics.json), generado por `fabric-inspection reproduce` (semilla 42, CPU i5-1135G7, 21,7 min). Los umbrales se eligieron en **validación** (política `max_f1_val`); el test solo se leyó para reportar.

| Modelo | AUPRC test (IC 95 % bootstrap) | AUROC | Precisión | Recall | F1 | FPR | AUPRC val |
|---|---|---|---|---|---|---|---|
| Baseline: clasificador ResNet-18 (usa defectos de train) | 0,997 (0,989–1,000) | 0,998 | 0,971 | 0,971 | 0,971 | 1,6 % | 0,954 |
| PatchCore ResNet-18 (solo tela buena) | 1,000 (1,000–1,000) | 1,000 | 0,972 | 1,000 | 0,986 | 1,6 % | 0,985 |
| **PatchCore Wide-ResNet-50 · servido (ONNX FP32)** | **1,000 (1,000–1,000)** | **1,000** | **1,000** | **1,000** | **1,000** | **0 %** | **1,000** |

- **Recall por tipo:** PatchCore detecta el 100 % de cada tipo; el baseline deja escapar 1 de 7 hilos sueltos (`thread` 0,86). Sin haber visto ningún defecto, la detección de anomalías iguala o supera al clasificador supervisado.
- **Localización:** AUROC de píxel de 0,995 en test.
- **Comprobación con la literatura** (split oficial de MVTec, WRN-50): AUROC de imagen 0,986, AUPRC 0,996 y AUROC de píxel 0,990, del orden de lo que publica PatchCore para *carpet*.
- **Tipo de defecto** (clasificador servido en INT8): macro-F1 0,845 (good 0,98 · color 0,93 · cut 0,77 · **hole 0,67** · metal 1,00 · thread 0,71).
- **INT8 decidido en validación:**
  - Clasificador INT8: 11,4 MB, −0,003 de AUPRC → **se sirve INT8**.
  - PatchCore INT8: la AUPRC de validación cayó de 1,00 a 0,47 porque cuantizar las convoluciones deforma la escala de distancias → **se sirve FP32** (111 MB).
- **Paridad PyTorch vs ONNX:** diferencia absoluta máxima de 4·10⁻⁶ (PatchCore) y 4·10⁻⁵ (clasificador).
- **Latencia y memoria:** 348 ms p50 / 403 ms p95 por imagen (1 hilo); el contenedor de la API usa ~385 MiB, dentro de los 512 MB de Render free.

**Lectura honesta.**
- El test tiene pocos defectos (35), por eso reporto el intervalo de confianza.
- *Carpet* es una categoría en la que PatchCore rinde muy bien, y todas las imágenes vienen de una sola alfombra y una sola cámara: en planta habría que recalibrar el umbral y la escala mm/px (supuesto S1).
- El clasificador confunde a veces un agujero con una mancha. En ASTM eso importa (un agujero vale 4 puntos), así que está en el roadmap.

| Curvas PR (test) | Ejemplos por tipo: verdad de campo · PatchCore · CAM |
|---|---|
| ![PR](reports/figures/pr_curves.png) | ![Ejemplos](reports/figures/examples.png) |

## Arquitectura
```mermaid
flowchart LR
  subgraph Offline["Pipeline reproducible (CPU, semilla fija)"]
    D["MVTec AD carpet<br/>SHA-256 fijado"] --> M["Manifiesto + split por imagen<br/>chequeo de fuga (sha256 + dHash)"]
    M --> B["Baseline<br/>ResNet-18 + CAM"]
    M --> P["PatchCore<br/>R18 y WRN-50"]
    B --> E["Evaluación en test común<br/>AUPRC + IC 95 %"]
    P --> E
    E --> X["ONNX + INT8<br/>decisión en val"]
    X --> R[("registry.json<br/>+ GitHub Release")]
    B -.-> ML[("MLflow local")]
    P -.-> ML
  end
  subgraph Cloud["Nube gratuita"]
    S["Streamlit demo<br/>Community Cloud"] -- "X-API-Key (servidor)" --> A["FastAPI + ONNX Runtime<br/>Render free"]
    A --> N[("PostgreSQL<br/>Neon")]
  end
  R -- "descarga verificada por SHA-256" --> A
  A -- "FabricInspectionRequest" --> T["textrack<br/>(MES)"]
```
El código está por capas: `api/controller` → `service` → `repository` (+ `api/dto`, `api/mapper`, `exceptions`). Las reglas de negocio puras (4 puntos, AQL, drift, ahorro, regiones) viven en `domain/`, sin dependencias de frameworks. La API no depende de PyTorch: solo NumPy, Pillow y ONNX Runtime.

## Stack y por qué
| Capa | Tecnología | Motivo |
|---|---|---|
| Entrenamiento | PyTorch 2.14 (CPU), timm 1.0, scikit-learn | Licencias BSD/Apache. Se evitó Ultralytics (AGPL-3.0). PatchCore está implementado en ~150 líneas en vez de usar anomalib, para exportarlo como un solo ONNX y no arrastrar Lightning. |
| Servicio | ONNX Runtime 1.30, FastAPI 0.141 | Inferencia en CPU sin PyTorch, imagen ligera. Cuantización estática QDQ S8S8, como recomienda la documentación para CNN. |
| Experimentos | MLflow 3.16 (SQLite local) | Parámetros, métricas y artefactos por corrida; no se expone a la red. |
| Datos | SQLAlchemy 2.1 + Alembic, SQLite / PostgreSQL 18 | El mismo esquema en local y en Neon; migraciones versionadas con catálogos. |
| Demo | Streamlit 1.64 | Cliente delgado de la API, gratis en Community Cloud. |
| DevOps | uv, Docker multi-stage, GitHub Actions | Dependencias fijadas en `uv.lock`, imágenes con versión fija y usuario no root, CI con gitleaks. |

## Modelo de datos
12 tablas derivadas de la investigación. Detalle de campos, restricciones y reglas R1–R12 en [docs/modelo-datos.md](docs/modelo-datos.md) (fuentes en [docs/investigacion.md](docs/investigacion.md)).

```mermaid
erDiagram
  DATASETS ||--o{ IMAGES : contiene
  IMAGES ||--|| LABELS : etiqueta
  DEFECT_CLASSES ||--o{ LABELS : clase
  EXPERIMENTS ||--o{ MODEL_VERSIONS : produce
  MODEL_VERSIONS ||--o{ PREDICTIONS : genera
  ROLLS ||--o{ PREDICTIONS : "cuadro del rollo"
  PREDICTIONS ||--o{ DETECTED_DEFECTS : regiones
  DEFECT_CLASSES ||--o{ DETECTED_DEFECTS : tipo
  ROLLS ||--o{ QUALITY_ALERTS : "4 puntos"
  MODEL_VERSIONS ||--o{ DRIFT_REPORTS : monitoreo
  LOT_INSPECTIONS }o..o{ ROLLS : "lot_code (AQL)"
```

## Ejecutar en local
**Con Docker** (API + PostgreSQL + demo, con rollos ficticios):
```bash
cp .env.example .env            # completa POSTGRES_PASSWORD, API_KEYS y DEMO_API_KEY
docker compose up --build       # API http://127.0.0.1:8000/docs · demo http://127.0.0.1:8501
```
Los modelos ONNX se leen de `./models` (montado en solo lectura) si ejecutaste `reproduce`; si no están, la API los descarga del GitHub Release `v1.0.0` y los verifica por SHA-256. En Windows, si la ruta del proyecto tiene caracteres no ASCII, Compose (bake) falla al construir: usa `docker build -t vision-defectos-tela-api:1.0.0 .`, `docker build -f demo/Dockerfile -t vision-defectos-tela-demo:1.0.0 .` y luego `docker compose up -d --no-build`.

**Sin Docker** (Python 3.12 + [uv](https://docs.astral.sh/uv/)):
```bash
uv sync --all-extras                          # dependencias exactas de uv.lock
uv run fabric-inspection reproduce            # ~22 min en CPU: descarga, entrena, evalúa y exporta
uv run fabric-inspection init-db              # SQLite en data/app.db (migraciones Alembic)
uv run fabric-inspection seed-demo            # 10 rollos y 400 cuadros ficticios
API_KEYS=<tu-clave> uv run fabric-inspection serve                 # http://127.0.0.1:8000/docs
API_KEY=<tu-clave> uv run streamlit run demo/app.py                # http://localhost:8501
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db           # experimentos en 127.0.0.1:5000
```
El notebook [notebooks/train_on_kaggle_or_colab.ipynb](notebooks/train_on_kaggle_or_colab.ipynb) corre el mismo pipeline en Kaggle o Colab (opcional).

## Variables de entorno
Genera claves con `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

| Variable | Descripción | Obligatoria |
|---|---|---|
| `API_KEYS` | Claves aceptadas por la API, separadas por comas (≥ 32 caracteres cada una). Sin ella la API no arranca. | Sí (API) |
| `DATABASE_URL` | `sqlite:///data/app.db` en local; en Neon: `postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require` | No (default SQLite) |
| `POSTGRES_PASSWORD` | Contraseña de PostgreSQL en Docker Compose | Sí (compose) |
| `DEMO_API_KEY` / `API_KEY` | Clave que usa la demo (una de `API_KEYS`) | Sí (demo) |
| `API_BASE_URL` | URL de la API para la demo | Sí (demo) |
| `CORS_ALLOWED_ORIGINS` | Orígenes explícitos separados por comas (se rechaza `*`) | No |
| `MM_PER_PIXEL` | Escala de la cámara en mm/píxel (supuesto S1, 0,1 por defecto) | No |
| `MAX_POINTS_PER_100_SQ_YD` | Límite de aceptación del rollo (supuesto S3, 40 por defecto) | No |
| `ENABLE_DOCS` | Publica `/docs` y `/openapi.json` | No |
| `MODEL_BASE_URL` | Reemplaza la URL del Release desde la que se bajan los ONNX | No |
| `FORWARDED_ALLOW_IPS` | Proxies de confianza para `X-Forwarded-For` (`*` solo detrás de Render) | No |

## API
Swagger: `http://127.0.0.1:8000/docs` · ejemplos listos en [docs/api.http](docs/api.http). Todo `/api/v1` exige `X-API-Key`.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado de la API y la BD (público) |
| POST | `/api/v1/predictions` | Inspecciona una imagen (multipart). Opcional: `roll_code`, `position_m`, `?include_heatmap=true` |
| GET | `/api/v1/predictions` | Registro de predicciones (paginado) |
| GET | `/api/v1/rolls` | Rollos con su calificación de 4 puntos |
| GET | `/api/v1/rolls/{code}/quality` | Detalle del rollo + payload para textrack |
| POST | `/api/v1/rolls/{code}/assessments` | Califica y guarda una alerta si no se acepta |
| GET | `/api/v1/quality/alerts` | Últimas alertas de calidad |
| GET | `/api/v1/quality/aql-plan` | Plan ISO 2859-1 (letra, n, Ac, Re) |
| POST | `/api/v1/lots/{lot_code}/inspections` | Acepta o rechaza un lote de rollos |
| GET / POST | `/api/v1/monitoring/drift` · `/drift-reports` | Drift de puntajes (KS) y tasa de alertas |
| POST | `/api/v1/savings/estimate` | Ahorro mensual frente a inspección manual |
| GET | `/api/v1/models/current` | Modelos servidos, hashes, umbrales y métricas |

## Tests y cobertura
```bash
uv run pytest --cov      # unitarios + integración (SQLite, API con ONNX reales diminutos, pipeline sintético)
uv run ruff check . && uv run ruff format --check . && uv run mypy
```
Resultado actual: **106 tests en verde, cobertura 93,8 %** (mínimo exigido en CI: 70 %).
- Los tests no descargan nada: el pipeline completo se prueba sobre un dataset sintético con la estructura de MVTec.
- Se comprueba que la misma semilla da la misma partición y los mismos puntajes.
- Se verifica que el CAM exportado coincide con Grad-CAM calculado por autograd.
- Las tablas AQL se prueban contra celdas del estándar, y los 4 puntos en los límites exactos de 3, 6 y 9 pulgadas.
- La seguridad de la API se prueba de punta a punta: 401, 413, 415, 422, 429, hash de modelo alterado y headers.

## Despliegue (gratis)
| Servicio | Qué corre | Límites del plan gratuito (verificados en sep-2026) |
|---|---|---|
| GitHub Releases | `patchcore.fp32.onnx` (111 MB) y `classifier.int8.onnx` (11 MB) | < 2 GiB por archivo, sin límite de ancho de banda |
| Render (Docker, `render.yaml`) | API FastAPI + ONNX Runtime | 512 MB de RAM; duerme tras 15 min y tarda ~1 min en despertar; disco efímero (por eso la BD está en Neon) |
| Neon | PostgreSQL de predicciones, alertas y drift | 0,5 GB y 100 CU-h por proyecto |
| Streamlit Community Cloud | Demo (`demo/app.py`, `demo/requirements.txt`) | 690 MB–2,7 GB de RAM; duerme tras 12 h sin tráfico |

Pasos:
1. Publica el Release `v1.0.0` con los dos ONNX.
2. Crea la BD en Neon.
3. En Render, crea el Blueprint desde `render.yaml` e ingresa `DATABASE_URL` y `API_KEYS`.
4. En Streamlit Cloud, apunta a `demo/app.py` y en *Secrets* pon `API_BASE_URL` (la URL de Render) y `API_KEY`.

El entrenamiento no se despliega: se reproduce en local o en Kaggle/Colab.

## Seguridad aplicada
- **Deny by default:** todo `/api/v1` exige `X-API-Key`, comparada en tiempo constante. La API **no arranca** sin claves de al menos 32 caracteres.
- **Rate limiting:** límite por clave y límite por IP para intentos no autenticados. Inferencia concurrente acotada, con 503 y `Retry-After`.
- **Subidas (OWASP API4):**
  - cuerpo de hasta 5 MB, cortado antes de parsear;
  - formato verificado por contenido (PNG/JPEG), lados de 32 a 4096 px;
  - protección contra *decompression bombs*;
  - la imagen no se guarda, solo su SHA-256 (minimización, LOPDP).
- **Errores sin stack traces** y sin eco de los valores enviados. CSP, `X-Frame-Options`, `nosniff`, `no-store`. CORS cerrado por defecto y `*` rechazado.
- **Cadena de suministro (OWASP ML06):**
  - ONNX cargados solo si su SHA-256 coincide con `registry.json`, descargados por HTTPS;
  - pesos preentrenados en *safetensors*, fijados por revisión y hash;
  - dataset verificado por hash;
  - `uv.lock` y `--require-hashes` en la demo.
- **Contenedores y CI:** Docker multi-stage con versiones fijas, usuario uid 10001 y healthcheck. CI con gitleaks sobre el historial, acciones fijadas por SHA y `permissions: contents: read`; Dependabot activo.
- Detalle completo en [SECURITY.md](SECURITY.md).

## Decisiones técnicas
- **Detección de anomalías en vez de solo clasificación.** Hay pocos defectos (37 en train) y los tipos futuros son desconocidos. PatchCore aprende la tela buena. Aun así entrené un baseline supervisado y lo evalué en el mismo test: PatchCore gana en recall (1,00 vs 0,97) sin ver defectos.
- **Protocolo sin fuga.**
  - La partición es por imagen, estratificada y con semilla, antes de redimensionar o aumentar datos.
  - Un test falla si aparecen duplicados exactos (SHA-256) o casi duplicados (dHash) entre particiones.
  - La normalización usa estadísticas de ImageNet.
  - Umbral, época, backbone y la decisión INT8/FP32 se fijan en validación.
- **Regla de selección escrita antes de ver resultados.** Gana la mejor AUPRC de validación; con un empate dentro de 0,005, el modelo más liviano. Ganó WRN-50 (1,000 frente a 0,985).
- **Referencia en CPU.** PyTorch no garantiza el mismo resultado entre CPU y GPU; por eso un solo comando en CPU reproduce las métricas del README, y Kaggle/Colab quedan como opción.
- **Hashes + manifiesto en vez de DVC.** El dataset es público y tiene URL fija, así que basta con verificar el SHA-256 sin montar un remoto de almacenamiento. Los pesos van en Releases.
- **Grad-CAM exportable.** Aprovecho la equivalencia matemática con CAM para no necesitar gradientes ni PyTorch en producción.
- **Drift que no confunde defectos reales con drift.** KS sobre los puntajes de la tela que parece normal frente a la referencia de validación, más un límite aparte para la tasa de alertas.
- **Demo como cliente delgado.** La API es la única que escribe en la BD y guarda la clave; la demo no expone secretos al navegador.

## Roadmap
- [ ] Mejorar el reconocimiento de agujeros (F1 0,67): más ejemplos, o usar la forma de la región para marcarlo como agujero, que vale 4 puntos.
- [ ] Probar la cuantización INT8 de PatchCore sin cuantizar las capas que alimentan el banco de memoria, o con calibración por percentil.
- [ ] Añadir AU-PRO de localización y un segundo dataset con licencia permisiva (DAGM 2007, CC BY 4.0).
- [ ] Calibrar `MM_PER_PIXEL` con un patrón físico y enviar las inspecciones a textrack de forma automática.
- [ ] Registro de modelos en MLflow Model Registry y despliegue *canary* de nuevas versiones.

## Fuentes de datos y licencias
- **MVTec AD — carpet** · © MVTec Software GmbH · [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (uso **no comercial**) · https://www.mvtec.com/company/research/datasets/mvtec-ad. Las imágenes no se versionan: se descargan y se verifican por hash. Las 6 muestras de `demo/samples` y las figuras de `reports/figures` se redistribuyen bajo la misma licencia (ver [demo/samples/LICENSE.md](demo/samples/LICENSE.md)).
  Citas: Bergmann et al. (2021) *IJCV* 129, 1038–1059, DOI 10.1007/s11263-020-01400-4; Bergmann et al. (2019) *CVPR*.
- **Pesos entrenados** (`*.onnx` del Release): CC BY-NC-SA 4.0, al derivar de MVTec AD (supuesto S6). **Código:** MIT.
- **Pesos preentrenados** `resnet18.tv_in1k` y `wide_resnet50_2.tv_in1k` (timm / torchvision): BSD-3-Clause, entrenados en ImageNet-1k (uso de investigación no comercial).
- **Normas:** ASTM D5430-26 (4 puntos), ISO 2859-1:2026 (AQL) y MIL-STD-105E (dominio público, origen de las tablas). Las 38 fuentes y los 14 supuestos están en [docs/investigacion.md](docs/investigacion.md).

## Autor
**Diego Francisco Granda Zhingre** · [GitHub](https://github.com/DiegoFranciscoG)
