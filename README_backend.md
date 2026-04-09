# SkyOps AI — Backend API

FastAPI backend con datos reales de FlightRadar24, Open-Meteo y modelo IA.

## Estructura del repo

```
skyops-api/
├── main.py                    ← API completa
├── modelo_vuelos_final.joblib ← tu modelo IA (cópialo aquí)
├── requirements.txt
├── render.yaml
└── README.md
```

## Paso 1 — Preparar el repo en GitHub

1. Crea un repo nuevo en GitHub llamado `skyops-api`
2. Copia estos archivos dentro
3. **MUY IMPORTANTE**: copia también `modelo_vuelos_final.joblib` en la raíz del repo
4. Haz commit y push

```bash
git init
git add .
git commit -m "feat: skyops api inicial"
git remote add origin https://github.com/TU_USUARIO/skyops-api.git
git push -u origin main
```

## Paso 2 — Deploy en Render (gratis)

1. Ve a https://render.com y crea cuenta con GitHub
2. Click "New +" → "Web Service"
3. Conecta el repo `skyops-api`
4. Render detecta el `render.yaml` automáticamente
5. Click "Create Web Service"
6. Espera ~3 minutos → te da una URL tipo:
   `https://skyops-api.onrender.com`

## Paso 3 — Probar la API

Abre en el navegador:
- `https://skyops-api.onrender.com/` → status
- `https://skyops-api.onrender.com/health` → modelo cargado?
- `https://skyops-api.onrender.com/docs` → Swagger UI interactivo
- `https://skyops-api.onrender.com/flights?airport=ALL&hours=15`
- `https://skyops-api.onrender.com/weather/ATL`
- `https://skyops-api.onrender.com/metar/ATL`

## Endpoints disponibles

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Status de la API |
| GET | `/health` | Health check + modelo IA |
| GET | `/airports` | Lista de aeropuertos |
| GET | `/flights?airport=ALL&hours=15` | Vuelos en vivo + riesgo IA |
| GET | `/weather/{iata}` | Previsión 24h Open-Meteo |
| GET | `/metar/{iata}` | METAR + TAF reales |
| GET | `/airlines` | Lista aerolíneas FlightRadar24 |

## ⚠️ Nota sobre el plan gratuito de Render

El servicio gratuito se "duerme" tras 15 min de inactividad.
La primera petición puede tardar ~30s en despertar.
Para un proyecto universitario es suficiente.
