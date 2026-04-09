from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import math
import time
import requests
import joblib
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

# Diagnóstico de FlightRadar24
try:
    from FlightRadar24.api import FlightRadar24API
    print("✅ Import FlightRadar24.api OK")
except ImportError:
    try:
        from flightradar24.api import Api as FlightRadar24API
        print("✅ Import flightradar24.api OK")
    except ImportError:
        try:
            import FlightRadar24
            print(f"✅ FlightRadar24 encontrado, dir: {dir(FlightRadar24)}")
            FlightRadar24API = None
        except ImportError as e:
            print(f"❌ FlightRadar24 no encontrado: {e}")
            FlightRadar24API = None
MODEL = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    try:
        MODEL = joblib.load("modelo_vuelos_final.joblib")
        print("✅ Modelo IA cargado correctamente")
    except Exception as e:
        print(f"⚠️  Modelo no encontrado: {e}")
    yield

# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="SkyOps AI — API",
    description="Backend para el panel de operaciones de vuelos USA con predicción IA",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # En producción cambia esto a tu dominio Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────
AIRPORTS = {
    "ATL": {"name": "Atlanta Hartsfield-Jackson", "coords": [33.6407, -84.4277], "icao": "KATL"},
    "ORD": {"name": "Chicago O'Hare",             "coords": [41.9742, -87.9073], "icao": "KORD"},
    "LAX": {"name": "Los Angeles International",  "coords": [33.9416, -118.4085],"icao": "KLAX"},
    "JFK": {"name": "New York JFK",               "coords": [40.6413, -73.7781], "icao": "KJFK"},
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    R = 3440.065
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dLon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_weather(iatas: list[str]) -> dict:
    """Descarga previsión horaria de Open-Meteo para cada IATA."""
    result = {}
    url = "https://api.open-meteo.com/v1/forecast"
    for apt in iatas:
        lat, lon = AIRPORTS[apt]["coords"]
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m,visibility,cloudcover,temperature_2m,precipitation",
            "wind_speed_unit": "kn",
            "precipitation_unit": "mm",
            "timezone": "UTC",
        }
        try:
            r = requests.get(url, params=params, timeout=10).json()
            times = r["hourly"]["time"]
            result[apt] = {
                t: {
                    "wind":      r["hourly"]["wind_speed_10m"][i]      or 0,
                    "gusts":     r["hourly"]["wind_gusts_10m"][i]      or 0,
                    "direction": r["hourly"]["wind_direction_10m"][i]  or 0,
                    "visibility":r["hourly"]["visibility"][i]          or 10000,
                    "clouds":    r["hourly"]["cloudcover"][i]          or 0,
                    "temp":      r["hourly"]["temperature_2m"][i]      or 15,
                    "precip":    r["hourly"]["precipitation"][i]       or 0,
                }
                for i, t in enumerate(times)
            }
        except Exception as e:
            print(f"⚠️  Weather error {apt}: {e}")
            result[apt] = {}
    return result


def get_weather_at(iata: str, dt: datetime, weather_cache: dict) -> dict:
    fallback = {"wind": 0, "gusts": 0, "direction": 0, "visibility": 10000,
                "clouds": 0, "temp": 15, "precip": 0}
    if iata not in weather_cache:
        return fallback
    key = dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
    return weather_cache[iata].get(key, fallback)


def predict_risk(origin: str, dest: str, carrier_iata: str,
                 eta: datetime, weather_cache: dict) -> dict:
    """Devuelve probabilidad y nivel de riesgo usando el modelo IA."""
    if MODEL is None:
        return {"score": 0.0, "level": "LOW", "label": "Baja", "pct": "0%"}

    c_orig = get_weather_at(origin, eta, weather_cache)
    c_dest = get_weather_at(dest,   eta, weather_cache)

    try:
        enc_orig = MODEL["le_orig"].transform([origin])[0]    if origin    in MODEL["le_orig"].classes_    else 0
        enc_dest = MODEL["le_dest"].transform([dest])[0]      if dest      in MODEL["le_dest"].classes_    else 0
        enc_carr = MODEL["le_carrier"].transform([carrier_iata])[0] if carrier_iata in MODEL["le_carrier"].classes_ else 0
    except Exception:
        enc_orig = enc_dest = enc_carr = 0

    features = pd.DataFrame([[
        c_orig["wind"], c_orig["gusts"], c_orig["visibility"], c_orig["clouds"], c_orig["temp"],
        c_dest["wind"], c_dest["gusts"], c_dest["visibility"], c_dest["clouds"], c_dest["temp"],
        enc_orig, enc_dest, enc_carr,
    ]], columns=MODEL["features"])

    prob = float(MODEL["modelo"].predict_proba(features)[0][1])

    if prob < 0.25:
        level, label = "LOW",    "Baja"
    elif prob < 0.60:
        level, label = "MEDIUM", "Media"
    else:
        level, label = "HIGH",   "Alta"

    return {
        "score": round(prob, 4),
        "level": level,
        "label": label,
        "pct":   f"{prob:.1%}",
        "windAtDest":   round(c_dest["wind"],  1),
        "precipAtDest": round(c_dest["precip"], 2),
    }


def safe_iata(node) -> str:
    try:
        if isinstance(node, dict) and isinstance(node.get("code"), dict):
            return node["code"].get("iata", "N/A") or "N/A"
    except Exception:
        pass
    return "N/A"


def safe_str(val, fallback="N/A") -> str:
    return str(val) if val is not None else fallback


# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "SkyOps AI API", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/airports")
def get_airports():
    """Lista de aeropuertos monitorizados."""
    return AIRPORTS


@app.get("/flights")
def get_flights(
    airport: str = Query(default="ALL", description="IATA code o ALL"),
    hours:   int = Query(default=15,  ge=1, le=24),
):
    """
    Devuelve:
    - inAir:      vuelos en ruta acercándose a las bases (≤500 nm)
    - arrivals:   llegadas programadas en las próximas {hours}h
    - departures: salidas programadas en las próximas {hours}h
    """
    iatas = list(AIRPORTS.keys()) if airport == "ALL" else [airport.upper()]
    for code in iatas:
        if code not in AIRPORTS:
            raise HTTPException(400, f"Aeropuerto desconocido: {code}")

    now = datetime.now(timezone.utc)
    limit = now + timedelta(hours=hours)

    # ── Descargar meteorología ──────────────────
    weather = fetch_weather(iatas)

    # ── FlightRadar24 ───────────────────────────
    fr = FlightRadar24API()
    in_air_raw, arrivals_raw, departures_raw = [], [], []

    try:
        all_flights = fr.get_flights()
        for f in all_flights:
            if getattr(f, "ground_speed", 0) > 0:
                dest = safe_str(getattr(f, "destination_airport_iata", "")).upper()
                if dest in iatas:
                    in_air_raw.append(f)
                else:
                    for apt in iatas:
                        dist = haversine_nm(
                            f.latitude, f.longitude,
                            AIRPORTS[apt]["coords"][0], AIRPORTS[apt]["coords"][1],
                        )
                        if dist < 500:
                            in_air_raw.append(f)
                            break
    except Exception as e:
        print(f"⚠️  FlightRadar24 inAir error: {e}")

    for apt in iatas:
        try:
            details = fr.get_airport_details(apt)
            arr = details["airport"]["pluginData"]["schedule"]["arrivals"]["data"]
            dep = details["airport"]["pluginData"]["schedule"]["departures"]["data"]
            for v in arr: v["_target"] = apt
            for v in dep: v["_target"] = apt
            arrivals_raw.extend(arr)
            departures_raw.extend(dep)
        except Exception as e:
            print(f"⚠️  FlightRadar24 schedule error {apt}: {e}")

    # ── Serializar vuelos en aire ───────────────
    in_air_out = []
    for f in in_air_raw:
        dest   = safe_str(getattr(f, "destination_airport_iata", "N/A")).upper()
        origin = safe_str(getattr(f, "origin_airport_iata",      "N/A")).upper()
        carrier_iata = safe_str(getattr(f, "airline_iata", "N/A"))
        speed  = max(getattr(f, "ground_speed", 1), 1)

        if dest in AIRPORTS:
            dist_nm = haversine_nm(
                f.latitude, f.longitude,
                AIRPORTS[dest]["coords"][0], AIRPORTS[dest]["coords"][1],
            )
            eta_dt = now + timedelta(hours=dist_nm / speed)
        else:
            eta_dt = now + timedelta(hours=2)

        risk = predict_risk(origin, dest, carrier_iata, eta_dt, weather)

        in_air_out.append({
            "id":           safe_str(getattr(f, "id", "")),
            "callsign":     safe_str(getattr(f, "callsign", "N/A")),
            "airline":      safe_str(getattr(f, "airline_short_name", "N/A")),
            "aircraft":     safe_str(getattr(f, "aircraft_code", "N/A")),
            "registration": safe_str(getattr(f, "registration", "N/A")),
            "origin":       origin,
            "destination":  dest,
            "latitude":     round(f.latitude,  4),
            "longitude":    round(f.longitude, 4),
            "altitude":     getattr(f, "altitude",      0),
            "speed":        getattr(f, "ground_speed",  0),
            "heading":      getattr(f, "heading",       0),
            "verticalSpeed":getattr(f, "vertical_speed",0),
            "eta":          eta_dt.isoformat(),
            "scheduledTime":eta_dt.isoformat(),
            "estimatedTime":eta_dt.isoformat(),
            "riskLevel":    risk["level"],
            "riskScore":    risk["score"],
            "riskLabel":    risk["label"],
            "windAtDest":   risk.get("windAtDest",   0),
            "precipAtDest": risk.get("precipAtDest", 0),
        })

    # ── Serializar llegadas / salidas ───────────
    def serialize_schedule(raw_list: list, mode: str) -> list:
        out = []
        for v in raw_list:
            fdata = v.get("flight") or {}
            ts_key = "arrival" if mode == "arrival" else "departure"
            ts_sched = (fdata.get("time") or {}).get("scheduled", {})
            ts_est   = (fdata.get("time") or {}).get("estimated", {}) or (fdata.get("time") or {}).get("real", {})

            sched_ts = (ts_sched or {}).get(ts_key)
            est_ts   = (ts_est   or {}).get(ts_key) or sched_ts
            if not sched_ts:
                continue

            sched_dt = datetime.fromtimestamp(sched_ts, tz=timezone.utc)
            if not (now <= sched_dt <= limit):
                continue

            est_dt = datetime.fromtimestamp(est_ts, tz=timezone.utc) if est_ts else sched_dt

            apt_data    = fdata.get("airport") or {}
            origin_node = apt_data.get("origin")      or {}
            dest_node   = apt_data.get("destination") or {}
            origin      = safe_iata(origin_node)
            destination = safe_iata(dest_node)
            target      = v.get("_target", "N/A")

            airline_data  = fdata.get("airline") or {}
            airline_name  = airline_data.get("name", "N/A")
            carrier_iata  = (airline_data.get("code") or {}).get("iata", "N/A")
            aircraft_data = fdata.get("aircraft") or {}
            aircraft_code = (aircraft_data.get("model") or {}).get("code", "N/A")
            registration  = aircraft_data.get("registration", "N/A")

            num_data   = (fdata.get("identification") or {}).get("number") or {}
            callsign   = num_data.get("default", "N/A")

            o = origin if mode == "arrival" else target
            d = target if mode == "arrival" else destination

            risk = predict_risk(o, d, carrier_iata, sched_dt, weather)

            out.append({
                "id":           callsign + str(sched_ts),
                "callsign":     callsign,
                "airline":      airline_name,
                "aircraft":     aircraft_code,
                "registration": registration,
                "origin":       o,
                "destination":  d,
                "latitude":     0,
                "longitude":    0,
                "altitude":     0,
                "speed":        0,
                "heading":      0,
                "verticalSpeed":0,
                "eta":          est_dt.isoformat(),
                "scheduledTime":sched_dt.isoformat(),
                "estimatedTime":est_dt.isoformat(),
                "riskLevel":    risk["level"],
                "riskScore":    risk["score"],
                "riskLabel":    risk["label"],
                "windAtDest":   risk.get("windAtDest",   0),
                "precipAtDest": risk.get("precipAtDest", 0),
            })
        return out

    arrivals   = serialize_schedule(arrivals_raw,   "arrival")
    departures = serialize_schedule(departures_raw, "departure")

    return {
        "airport":    airport,
        "hours":      hours,
        "timestamp":  now.isoformat(),
        "inAir":      in_air_out,
        "arrivals":   arrivals,
        "departures": departures,
        "counts": {
            "inAir":      len(in_air_out),
            "arrivals":   len(arrivals),
            "departures": len(departures),
        },
    }


@app.get("/weather/{iata}")
def get_weather(iata: str):
    """Previsión horaria completa para un aeropuerto (24h)."""
    iata = iata.upper()
    if iata not in AIRPORTS:
        raise HTTPException(400, f"Aeropuerto desconocido: {iata}")

    weather = fetch_weather([iata])
    apt_weather = weather.get(iata, {})

    result = []
    for hour_str, data in apt_weather.items():
        result.append({"hour": hour_str + ":00", **data})

    return {
        "airport":   iata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forecast":  result[:24],
    }


@app.get("/metar/{iata}")
def get_metar_taf(iata: str):
    """METAR y TAF reales desde aviationweather.gov."""
    iata = iata.upper()
    if iata not in AIRPORTS:
        raise HTTPException(400, f"Aeropuerto desconocido: {iata}")

    icao = AIRPORTS[iata]["icao"]
    metar_text = taf_text = ""

    try:
        r = requests.get(
            f"https://aviationweather.gov/api/data/metar?ids={icao}&format=raw",
            timeout=8,
        )
        metar_text = r.text.strip() if r.status_code == 200 else f"No hay METAR para {icao}"
    except Exception:
        metar_text = f"Error obteniendo METAR para {icao}"

    try:
        r = requests.get(
            f"https://aviationweather.gov/api/data/taf?ids={icao}&format=raw",
            timeout=8,
        )
        taf_text = r.text.strip() if r.status_code == 200 else f"No hay TAF para {icao}"
    except Exception:
        taf_text = f"Error obteniendo TAF para {icao}"

    return {
        "airport":   iata,
        "icao":      icao,
        "metar":     metar_text,
        "taf":       taf_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/airlines")
def get_airlines():
    """Lista de aerolíneas desde FlightRadar24."""
    try:
        fr = FlightRadar24API()
        airlines = fr.get_airlines()
        return [
            {"icao": a.get("ICAO", ""), "iata": a.get("Code", ""), "name": a["Name"]}
            for a in airlines if "Name" in a
        ]
    except Exception as e:
        raise HTTPException(500, f"Error obteniendo aerolíneas: {e}")
