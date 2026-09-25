# Política de seguridad

## Reportar una vulnerabilidad
No abras un issue público. Escríbeme por GitHub (perfil DiegoFranciscoG) con los pasos para reproducir el problema. Respondo en un máximo de 7 días.

## Prácticas aplicadas en este proyecto
- **Secretos solo por variables de entorno.** `.env` no se versiona (ver `.env.example`). La API no arranca sin `API_KEYS`, y cada clave debe tener al menos 32 caracteres. Docker Compose exige `POSTGRES_PASSWORD`, `API_KEYS` y `DEMO_API_KEY` (`${VAR:?}`).
- **Deny by default.** Todas las rutas `/api/v1/*` exigen `X-API-Key` y la comparación es de tiempo constante. Solo `/health` y la documentación OpenAPI son públicas.
- **Rate limiting.** Hay un límite por clave (30/min por defecto) y otro por IP para los intentos sin clave válida, contra fuerza bruta.
- **Consumo de recursos acotado (OWASP API4).**
  - Cuerpo limitado a 5 MB (más el overhead de multipart) antes de parsear.
  - Imágenes validadas por contenido: solo PNG o JPEG, lados de 32 a 4096 px y protección contra *decompression bombs* de Pillow.
  - Como mucho 2 inferencias concurrentes; si se agota la espera, la API responde 503 con `Retry-After`.
- **Errores sin detalles internos.** No se exponen stack traces ni se devuelven los valores enviados en los errores de validación. Los errores inesperados solo llevan un `error_id`.
- **Cabeceras de seguridad.** CSP restrictiva, `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: no-referrer` y `Cache-Control: no-store`. CORS está cerrado por defecto y el valor `*` se rechaza.
- **Cadena de suministro del modelo (OWASP ML06).**
  - Los ONNX se cargan solo si su SHA-256 coincide con `models/registry.json` y se descargan únicamente por HTTPS.
  - Los pesos preentrenados son *safetensors* fijados por revisión y hash; nunca se deserializan *pickles*.
  - Las dependencias están fijadas en `uv.lock`, y las de la demo con `--require-hashes`.
- **Datos.** Las imágenes subidas no se guardan: solo su SHA-256 (minimización de datos, LOPDP). Los datos de rollos son ficticios y el dataset se verifica por hash.
- **Contenedores.** Las imágenes son multi-stage, con versiones fijas, usuario sin privilegios (uid 10001) y healthcheck.
- **CI.** gitleaks revisa el historial completo. Las acciones de GitHub están fijadas por SHA con `permissions: contents: read`, y Dependabot vigila las dependencias.
- **MLflow** se usa solo en local (`127.0.0.1`) y no se despliega.

## Alcance
Es un proyecto de portafolio. Los pesos entrenados se distribuyen bajo CC BY-NC-SA 4.0 (heredado de MVTec AD) y **no** deben usarse con fines comerciales.
