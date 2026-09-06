"""
EnsembleWetter Austria — FastAPI Backend
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from weather_core import (
    geocode_ort, lade_alle_daten, berechne_konsens,
    wende_trockenfenster_an, finde_bestes_fenster,
    berechne_bodenindex, ANWENDUNGSFAELLE, STUFEN,
    BODENARTEN, BODENART_STANDARD, hole_prognoseguete,
    finde_fall
)

app = FastAPI(
    title="EnsembleWetter Austria API",
    description="Ensemble-Wettervorhersage für Landwirtschaft und Alpinraum",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ensemblewetter.at",
                   "https://www.ensemblewetter.at",
                   "http://localhost:5173"],  # Vite Dev-Server
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Anfrage/Antwort-Modelle ──────────────────────────────────────────────────

class AuswertungAnfrage(BaseModel):
    ort: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    kategorie: str
    fall: str
    tage: int = 5
    bodenart: str = BODENART_STANDARD
    hoehe: Optional[float] = None
    parameter: Optional[dict] = None  # Nutzer-Schwellenwerte


class StundenBewertung(BaseModel):
    zeit: str
    stufe: int
    ampel: str
    sicherheit: str
    n_familien: int
    temp: Optional[float]
    feuchte: Optional[float]
    wind: Optional[float]
    ns_wahr: Optional[float]
    ns_p75: Optional[float]
    gruende: Optional[str]


class AuswertungAntwort(BaseModel):
    ort: str
    lat: float
    lon: float
    hoehe: Optional[float]
    kategorie: str
    fall: str
    tage: int
    bestes_fenster: Optional[dict]
    stunden: list[StundenBewertung]
    bodenindex: Optional[dict]
    gradient_info: Optional[str]
    inversion_anteil: float


# ── Endpunkte ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/anwendungsfaelle")
def liste_anwendungsfaelle():
    """Alle verfügbaren Kategorien und Fälle zurückgeben."""
    ergebnis = {}
    for kat, faelle in ANWENDUNGSFAELLE.items():
        ergebnis[kat] = {}
        for fall_name, cfg in faelle.items():
            ergebnis[kat][fall_name] = {
                "beschreibung": cfg.get("beschreibung", ""),
                "boden_relevant": cfg.get("boden_relevant", False),
                "trocknung_relevant": cfg.get("trocknung_relevant", False),
                "alpin": cfg.get("alpin_bewölkung", False),
            }
    return ergebnis


@app.get("/bodenarten")
def liste_bodenarten():
    return {
        name: {"hinweis": cfg["hinweis"],
               "nutzbare_fk": cfg.get("nutzbare_fk", 60.0)}
        for name, cfg in BODENARTEN.items()
    }


@app.get("/geocode")
def geocode(q: str = Query(..., min_length=2)):
    """Ortssuche — gibt bis zu 8 Treffer zurück."""
    treffer = geocode_ort(q)
    if not treffer:
        return []
    return treffer


@app.post("/auswertung", response_model=AuswertungAntwort)
def auswertung(anfrage: AuswertungAnfrage):
    """
    Kern-Endpunkt: Ensemble-Auswertung für einen Ort und Anwendungsfall.
    """
    # Koordinaten ermitteln
    lat, lon = anfrage.lat, anfrage.lon
    if lat is None or lon is None:
        treffer = geocode_ort(anfrage.ort)
        if not treffer:
            raise HTTPException(status_code=404,
                                detail=f"Ort nicht gefunden: {anfrage.ort}")
        lat = treffer[0]["lat"]
        lon = treffer[0]["lon"]

    # Anwendungsfall laden
    fall = finde_fall(f"{anfrage.kategorie} / {anfrage.fall}")
    if fall is None:
        # Direkte Suche
        kat = ANWENDUNGSFAELLE.get(anfrage.kategorie, {})
        fall = kat.get(anfrage.fall)
    if fall is None:
        raise HTTPException(status_code=404,
                            detail=f"Anwendungsfall nicht gefunden: "
                                   f"{anfrage.kategorie} / {anfrage.fall}")

    af = dict(fall)
    af["name"] = anfrage.fall
    af["modus"] = "standard"

    # Nutzer-Parameter überschreiben
    if anfrage.parameter:
        for k, v in anfrage.parameter.items():
            af[k] = v

    # Daten laden und auswerten
    try:
        alle_daten = lade_alle_daten(
            lat, lon, af, anfrage.tage,
            standorthoehe=anfrage.hoehe)
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"Datenabruf fehlgeschlagen: {e}")

    konsens = berechne_konsens(alle_daten, af, anfrage.tage)
    konsens = wende_trockenfenster_an(konsens, af)
    ziel, bestes, n_bloecke = finde_bestes_fenster(konsens)

    # Boden
    bodenindex = None
    if af.get("boden_relevant") and alle_daten.get("boden"):
        try:
            bi = berechne_bodenindex(alle_daten["boden"], anfrage.bodenart)
            bodenindex = {
                "zustand": bi["metriken"]["zustand_text"],
                "saettigung": round(bi["metriken"]["saettigung"] * 100, 1),
                "regen_7t": bi["metriken"]["regen_7t"],
                "tage_seit_regen": bi["metriken"]["tage_seit_regen"],
                "bodenart": anfrage.bodenart,
                "hinweis": bi["bodenart_hinweis"],
            }
        except Exception:
            pass

    # Fenster serialisieren
    fenster_dict = None
    if bestes:
        fenster_dict = {
            "stufe": ziel,
            "mindest_stufe": bestes.get("mindest_name"),
            "stunden": bestes["stunden"],
            "start": bestes["start"].isoformat(),
            "ende": bestes["ende"].isoformat(),
            "sicherheit": bestes["sicherheit"],
            "n_bloecke": n_bloecke,
        }

    # Stunden serialisieren
    def _f(x):
        try:
            v = float(x)
            return None if pd.isna(v) else round(v, 1)
        except (TypeError, ValueError):
            return None

    stunden = []
    for _, r in konsens.iterrows():
        stunden.append(StundenBewertung(
            zeit=r["time"].isoformat(),
            stufe=int(r["ampel_int"]),
            ampel=r["ampel"],
            sicherheit=r["sicherheit"],
            n_familien=int(r.get("n_familien", 0) or 0),
            temp=_f(r.get("temp_median")),
            feuchte=_f(r.get("feuchte_median")),
            wind=_f(r.get("wind_median")),
            ns_wahr=_f(r.get("ns_wahrscheinlichkeit")),
            ns_p75=_f(r.get("ns_p75")),
            gruende=str(r.get("gruende", "") or ""),
        ))

    # Gradient-Info
    gdf = alle_daten.get("gradient")
    grad_info = None
    if gdf is not None and len(gdf):
        g = float(pd.to_numeric(gdf["grad"], errors="coerce").median())
        inv = float(gdf["inversion"].mean())
        grad_info = (f"Gradient {g:.2f} °C/100 m"
                     + (" — Inversion erkannt" if inv > 0.15 else ""))

    return AuswertungAntwort(
        ort=anfrage.ort, lat=lat, lon=lon,
        hoehe=alle_daten.get("standorthoehe"),
        kategorie=anfrage.kategorie, fall=anfrage.fall,
        tage=anfrage.tage,
        bestes_fenster=fenster_dict,
        stunden=stunden,
        bodenindex=bodenindex,
        gradient_info=grad_info,
        inversion_anteil=round(
            float(alle_daten.get("inversion_anteil", 0)), 3),
    )


@app.get("/prognoseguete")
def prognoseguete(lat: float, lon: float,
                  tage: int = Query(default=7, ge=3, le=14)):
    """INCA-gestützte Verifikation der Modellgüte."""
    ergebnis = hole_prognoseguete(lat, lon, tage)
    if not ergebnis:
        return {"verfuegbar": False,
                "hinweis": "Keine Verifikationsdaten verfügbar."}
    return {"verfuegbar": True, **ergebnis}


@app.get("/stufen")
def stufen_info():
    """Erläuterung der Bewertungsstufen."""
    return {
        int(k): {"name": v["name"], "farbe": v["farbe"]}
        for k, v in STUFEN.items()
    }
