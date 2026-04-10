from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import math
import requests
import joblib
import pandas as pd
from datetime import datetime, timedelta, timezone
from FlightRadar24 import FlightRadar24API

# ─────────────────────────────────────────────
#  LIFESPAN
# ─────────────────────────────────────────────
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
app = FastAPI(title="SkyOps AI — API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    "LAX": {"name": "Los Angeles International",  "coords": [33.9416, -118.4085], "icao": "KLAX"},
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


def fetch_weather(iatas: list) -> dict:
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
                    "wind":       r["hourly"]["wind_speed_10m"][i]     or 0,
                    "gusts":      r["hourly"]["wind_gusts_10m"][i]     or 0,
                    "direction":  r["hourly"]["wind_direction_10m"][i] or 0,
                    "visibility": r["hourly"]["visibility"][i]         or 10000,
                    "clouds":     r["hourly"]["cloudcover"][i]         or 0,
                    "temp":       r["hourly"]["temperature_2m"][i]     or 15,
                    "precip":     r["hourly"]["precipitation"][i]      or 0,
                }
                for i, t in enumerate(times)
            }
        except Exception as e:
            print(f"⚠️  Weather error {apt}: {e}")
            result[apt] = {}
    return result


def get_weather_at(iata: str, dt: datetime, cache: dict) -> dict:
    fallback = {"wind": 0, "gusts": 0, "direction": 0,
                "visibility": 10000, "clouds": 0, "temp": 15, "precip": 0}
    if iata not in cache:
        return fallback
    key = dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00")
    return cache[iata].get(key, fallback)


def predict_risk(origin, dest, carrier_iata, eta, weather_cache) -> dict:
    if MODEL is None:
        return {"score": 0.0, "level": "LOW", "label": "Baja",
                "pct": "0%", "windAtDest": 0, "precipAtDest": 0}
    c_orig = get_weather_at(origin, eta, weather_cache)
    c_dest = get_weather_at(dest,   eta, weather_cache)
    try:
        enc_orig = MODEL["le_orig"].transform([origin])[0]          if origin       in MODEL["le_orig"].classes_    else 0
        enc_dest = MODEL["le_dest"].transform([dest])[0]            if dest         in MODEL["le_dest"].classes_    else 0
        enc_carr = MODEL["le_carrier"].transform([carrier_iata])[0] if carrier_iata in MODEL["le_carrier"].classes_ else 0
    except Exception:
        enc_orig = enc_dest = enc_carr = 0

    features = pd.DataFrame([[
        c_orig["wind"], c_orig["gusts"], c_orig["visibility"], c_orig["clouds"], c_orig["temp"],
        c_dest["wind"], c_dest["gusts"], c_dest["visibility"], c_dest["clouds"], c_dest["temp"],
        enc_orig, enc_dest, enc_carr,
    ]], columns=MODEL["features"])

    prob = float(MODEL["modelo"].predict_proba(features)[0][1])
    if prob < 0.25:   level, label = "LOW",    "Baja"
    elif prob < 0.60: level, label = "MEDIUM", "Media"
    else:             level, label = "HIGH",   "Alta"

    return {
        "score": round(prob, 4), "level": level, "label": label,
        "pct": f"{prob:.1%}",
        "windAtDest":   round(c_dest["wind"],   1),
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
    return {
        "status": "ok",
        "service": "SkyOps AI API",
        "version": "1.0.0",
        "model_loaded": MODEL is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/airports")
def get_airports():
    return AIRPORTS


@app.get("/flights")
def get_flights(
    airport: str = Query(default="ALL"),
    hours:   int = Query(default=15, ge=1, le=24),
):
    iatas = list(AIRPORTS.keys()) if airport == "ALL" else [airport.upper()]
    for code in iatas:
        if code not in AIRPORTS:
            raise HTTPException(400, f"Aeropuerto desconocido: {code}")

    now   = datetime.now(timezone.utc)
    limit = now + timedelta(hours=hours)
    weather = fetch_weather(iatas)

    fr = FlightRadar24API()
    in_air_raw, arrivals_raw, departures_raw = [], [], []

    # ── Vuelos en aire ──────────────────────────
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
        print(f"⚠️  get_flights error: {e}")

    # ── Schedules ──────────────────────────────
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
            print(f"⚠️  schedule error {apt}: {e}")

    # ── Serializar vuelos en aire ───────────────
    in_air_out = []
    for f in in_air_raw:
        dest         = safe_str(getattr(f, "destination_airport_iata", "N/A")).upper()
        origin       = safe_str(getattr(f, "origin_airport_iata",      "N/A")).upper()
        carrier_iata = safe_str(getattr(f, "airline_iata", "N/A"))
        speed        = max(getattr(f, "ground_speed", 1) or 1, 1)

        if dest in AIRPORTS:
            d_lat, d_lon = AIRPORTS[dest]["coords"]
        else:
            d_lat = f.latitude
            d_lon = f.longitude

        dist_nm = haversine_nm(f.latitude, f.longitude, d_lat, d_lon)
        eta_dt  = now + timedelta(hours=dist_nm / speed)
        risk    = predict_risk(origin, dest, carrier_iata, eta_dt, weather)

        in_air_out.append({
            "id":            safe_str(getattr(f, "id", "")),
            "callsign":      safe_str(getattr(f, "callsign",           "N/A")),
            "airline":       safe_str(getattr(f, "airline_short_name", "N/A")),
            "aircraft":      safe_str(getattr(f, "aircraft_code",      "N/A")),
            "registration":  safe_str(getattr(f, "registration",       "N/A")),
            "origin":        origin,
            "destination":   dest,
            "latitude":      round(float(f.latitude),  4),
            "longitude":     round(float(f.longitude), 4),
            "altitude":      getattr(f, "altitude",       0) or 0,
            "speed":         getattr(f, "ground_speed",   0) or 0,
            "heading":       getattr(f, "heading",        0) or 0,
            "verticalSpeed": getattr(f, "vertical_speed", 0) or 0,
            "eta":           eta_dt.isoformat(),
            "scheduledTime": eta_dt.isoformat(),
            "estimatedTime": eta_dt.isoformat(),
            "riskLevel":     risk["level"],
            "riskScore":     risk["score"],
            "riskLabel":     risk["label"],
            "windAtDest":    risk["windAtDest"],
            "precipAtDest":  risk["precipAtDest"],
        })

    # ── Serializar llegadas / salidas ───────────
    def serialize_schedule(raw_list: list, mode: str) -> list:
        out = []
        for v in raw_list:
            try:
                fdata   = v.get("flight") or {}
                ts_key  = "arrival" if mode == "arrival" else "departure"
                t_node  = fdata.get("time") or {}
                ts_sched = (t_node.get("scheduled") or {}).get(ts_key)
                ts_est   = (t_node.get("estimated") or {}).get(ts_key) or \
                           (t_node.get("real")      or {}).get(ts_key) or ts_sched
                if not ts_sched:
                    continue

                sched_dt = datetime.fromtimestamp(ts_sched, tz=timezone.utc)
                if not (now <= sched_dt <= limit):
                    continue

                est_dt = datetime.fromtimestamp(ts_est, tz=timezone.utc) if ts_est else sched_dt

                apt_data    = fdata.get("airport") or {}
                origin_nd   = apt_data.get("origin")      or {}
                dest_nd     = apt_data.get("destination") or {}
                origin      = safe_iata(origin_nd)
                destination = safe_iata(dest_nd)
                target      = v.get("_target", "N/A")

                al_data       = fdata.get("airline") or {}
                airline_name  = al_data.get("name", "N/A")
                carrier_iata  = (al_data.get("code") or {}).get("iata", "N/A")
                ac_data       = fdata.get("aircraft") or {}
                aircraft_code = (ac_data.get("model") or {}).get("code", "N/A")
                registration  = ac_data.get("registration", "N/A")
                num_data      = (fdata.get("identification") or {}).get("number") or {}
                callsign      = num_data.get("default", "N/A")

                o = origin if mode == "arrival" else target
                d = target if mode == "arrival" else destination

                risk = predict_risk(o, d, carrier_iata, sched_dt, weather)

                out.append({
                    "id":            callsign + str(ts_sched),
                    "callsign":      callsign,
                    "airline":       airline_name,
                    "aircraft":      aircraft_code,
                    "registration":  registration,
                    "origin":        o,
                    "destination":   d,
                    "latitude":      0,
                    "longitude":     0,
                    "altitude":      0,
                    "speed":         0,
                    "heading":       0,
                    "verticalSpeed": 0,
                    "eta":           est_dt.isoformat(),
                    "scheduledTime": sched_dt.isoformat(),
                    "estimatedTime": est_dt.isoformat(),
                    "riskLevel":     risk["level"],
                    "riskScore":     risk["score"],
                    "riskLabel":     risk["label"],
                    "windAtDest":    risk["windAtDest"],
                    "precipAtDest":  risk["precipAtDest"],
                })
            except Exception as e:
                print(f"⚠️  parse error: {e}")
                continue
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
    iata = iata.upper()
    if iata not in AIRPORTS:
        raise HTTPException(400, f"Aeropuerto desconocido: {iata}")
    weather     = fetch_weather([iata])
    apt_weather = weather.get(iata, {})
    result = [{"hour": k + ":00", **v} for k, v in apt_weather.items()]
    return {
        "airport":   iata,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "forecast":  result[:24],
    }


@app.get("/metar/{iata}")
def get_metar_taf(iata: str):
    iata = iata.upper()
    if iata not in AIRPORTS:
        raise HTTPException(400, f"Aeropuerto desconocido: {iata}")
    icao = AIRPORTS[iata]["icao"]
    try:
        r = requests.get(
            f"https://aviationweather.gov/api/data/metar?ids={icao}&format=raw",
            timeout=8,
        )
        metar = r.text.strip() if r.status_code == 200 else f"No hay METAR para {icao}"
    except Exception:
        metar = f"Error obteniendo METAR para {icao}"
    try:
        r = requests.get(
            f"https://aviationweather.gov/api/data/taf?ids={icao}&format=raw",
            timeout=8,
        )
        taf = r.text.strip() if r.status_code == 200 else f"No hay TAF para {icao}"
    except Exception:
        taf = f"Error obteniendo TAF para {icao}"
    return {
        "airport":   iata,
        "icao":      icao,
        "metar":     metar,
        "taf":       taf,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
