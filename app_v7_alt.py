# ============================================================
# EnsembleWetter — Präzisionsprognose für Landwirtschaft und Alpinraum (v7)
# ============================================================
# Starten:  streamlit run app_v7.py
# Neu in dieser Version:
#   • Bodenzustand-Übersicht (Vorgeschichte 10 Tage, gewichtet)
#   • Bodeneinfluss auf die Heatmap (moderat + Extremnässe = Rot)
#   • Ensemble-Zähler = echte Summe aller Läufe je Stunde
#   • Interaktive Plotly-Heatmap (24h, Hover + Klick für Details)
#   • Lange Zeitfenster-Liste entfernt
# ============================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# MODELL-KONFIGURATION
# ============================================================

HAUPTLAUFE = {
    'ECMWF IFS 9km': {'url': 'https://api.open-meteo.com/v1/ecmwf',
                      'max_tage': 10, 'aufloesung': '9 km'},
    'ICON EU 7km': {'url': 'https://api.open-meteo.com/v1/dwd-icon',
                    'max_tage': 7, 'aufloesung': '7 km'},
    'GFS 25km': {'url': 'https://api.open-meteo.com/v1/gfs',
                 'max_tage': 16, 'aufloesung': '25 km'},
    'ARPEGE 13km': {'url': 'https://api.open-meteo.com/v1/meteofrance',
                    'max_tage': 4, 'aufloesung': '13 km'},
    'GeoSphere Austria': {'url': 'https://api.open-meteo.com/v1/forecast',
                          'url_param': 'geosphere_austria',
                          'max_tage': 3, 'aufloesung': '~2 km'},
}

ENSEMBLE_MODELLE = {
    'ICON-D2-EPS 2km': {'model_key': 'icon_d2_eps', 'mitglieder': 20,
                        'max_stunden': 48, 'aufloesung': '2 km', 'fokus': 'kurzfrist'},
    'ICON-EU-EPS 13km': {'model_key': 'icon_eu_eps', 'mitglieder': 40,
                         'max_stunden': 120, 'aufloesung': '13 km', 'fokus': 'mittelfrist'},
    'ICON-EPS Global 26km': {'model_key': 'icon_seamless_eps', 'mitglieder': 40,
                             'max_stunden': 180, 'aufloesung': '26 km', 'fokus': 'mittelfrist'},
    'ECMWF IFS ENS 9km': {'model_key': 'ecmwf_ifs025', 'mitglieder': 51,
                          'max_stunden': 360, 'aufloesung': '9 km', 'fokus': 'langfrist'},
    'GFS ENS 25km': {'model_key': 'gfs025_ens', 'mitglieder': 31,
                     'max_stunden': 240, 'aufloesung': '25 km', 'fokus': 'langfrist'},
    'GEM ENS 25km': {'model_key': 'gem_global_ens', 'mitglieder': 21,
                     'max_stunden': 384, 'aufloesung': '25 km', 'fokus': 'langfrist'},
}

ENSEMBLE_API = 'https://ensemble-api.open-meteo.com/v1/ensemble'
ENS_VARIABLEN = ('temperature_2m,relative_humidity_2m,precipitation,'
                 'wind_speed_10m')
HAUPT_VARIABLEN = ('temperature_2m,relative_humidity_2m,precipitation,'
                   'wind_speed_10m,wind_gusts_10m,wind_direction_10m,'
                   'cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,'
                   'dew_point_2m')

# Höhengradienten für alpine Korrektur
TEMP_GRADIENT_PRO_100M = 0.65      # °C Abnahme je 100 m
WIND_FAKTOR_PRO_1000M = 0.35       # relative Windzunahme je 1000 m
ALPIN_SCHWELLE_M = 1200            # ab hier gilt ein Standort als alpin

# ============================================================
# BODEN-KONFIGURATION (kalibrierbar!)
# ============================================================
# Diese Werte sind Startwerte für Grünland/Normalboden und
# sollten mit Rückmeldung der Landwirte angepasst werden.

BODEN_VORGESCHICHTE_TAGE = 10       # wie weit zurück
BODEN_DRAINAGE_MM = 6.0             # tägl. Wasserverlust zusätzlich zur Verdunstung
                                    # (et0 ~4 + 6 ≈ 10 mm/Tag → ~100 mm in 10 Tagen weg)

# Schwellen für die Wasserbilanz (mm Überschuss) → Bodenzustand
BILANZ_FEUCHT = 12
BILANZ_NASS = 25
BILANZ_EXTREM = 40

# ============================================================
# ANWENDUNGSFÄLLE — nach Bereichen gruppiert
# ============================================================
# Jeder Eintrag enthält Richtwerte (vom Bediener anpassbar) sowie:
#   boden_relevant  – Bodennässe beeinflusst die Befahrbarkeit
#   duerre_relevant – Trockenheit ist ungünstig (Keimung)
#   trockenfenster_h – benötigte trockene Stunden nach Beginn (0 = egal)
#   modus – 'standard' oder 'frost'
# Die Werte sind fachliche Startwerte und über die Oberfläche editierbar.

ANWENDUNGSFAELLE = {
    'Grünland & Feldfutter': {
        'Heuernte (bodengetrocknet)': {
            'temp_min': 15, 'temp_max': 35, 'humidity_max': 65, 'wind_max': 25,
            'precip_max': 0.0, 'trockenfenster_h': 48, 'boden_relevant': True,
            'beschreibung': 'Mahd + 2–3 Tage Abtrocknung am Boden'},
        'Heuernte (Belüftung)': {
            'temp_min': 13, 'temp_max': 35, 'humidity_max': 75, 'wind_max': 25,
            'precip_max': 0.0, 'trockenfenster_h': 12, 'boden_relevant': True,
            'beschreibung': 'Anwelken, Trocknung dann per Belüftung im Stock'},
        'Silieren / Anwelksilage': {
            'temp_min': 10, 'temp_max': 35, 'humidity_max': 75, 'wind_max': 30,
            'precip_max': 0.2, 'trockenfenster_h': 8, 'boden_relevant': True,
            'beschreibung': 'Kurzes Anwelken, Wind unkritisch'},
        'Nachsaat / Übersaat Grünland': {
            'temp_min': 8, 'temp_max': 25, 'humidity_min': 40, 'humidity_max': 85,
            'wind_max': 20, 'precip_max': 0.5, 'boden_relevant': True,
            'duerre_relevant': True,
            'beschreibung': 'Mild, Boden nicht zu nass/trocken'},
    },
    'Ackerbau': {
        'Säen Getreide': {
            'temp_min': 8, 'temp_max': 25, 'humidity_max': 80, 'wind_max': 25,
            'precip_max': 0.5, 'boden_relevant': True, 'duerre_relevant': True,
            'beschreibung': 'Mild, abgetrockneter Boden'},
        'Säen Mais': {
            'temp_min': 10, 'temp_max': 28, 'humidity_max': 80, 'wind_max': 25,
            'precip_max': 0.5, 'boden_relevant': True, 'duerre_relevant': True,
            'beschreibung': 'Boden ausreichend warm (≥10°C)'},
        'Säen Raps': {
            'temp_min': 10, 'temp_max': 28, 'humidity_max': 80, 'wind_max': 25,
            'precip_max': 0.5, 'boden_relevant': True, 'duerre_relevant': True,
            'beschreibung': 'Feinsämerei, gleichmäßige Bodenfeuchte'},
        'Ernten / Dreschen': {
            'temp_min': 18, 'temp_max': 38, 'humidity_max': 55, 'wind_max': 30,
            'precip_max': 0.0, 'trockenfenster_h': 6, 'boden_relevant': True,
            'beschreibung': 'Heiß, trocken, niedrige Kornfeuchte'},
        'Pflügen / Bodenbearbeitung': {
            'temp_min': 3, 'temp_max': 35, 'wind_max': 40, 'precip_max': 1.0,
            'boden_relevant': True,
            'beschreibung': 'Boden abgetrocknet genug zum Befahren'},
        'Spritzen Fungizide': {
            'temp_min': 10, 'temp_max': 25, 'humidity_min': 60, 'humidity_max': 95,
            'wind_max': 15, 'precip_max': 0.0, 'boden_relevant': False,
            'beschreibung': 'Windstill, feucht, mild, kein Regen'},
        'Spritzen Herbizide': {
            'temp_min': 8, 'temp_max': 25, 'humidity_min': 50, 'humidity_max': 95,
            'wind_max': 12, 'precip_max': 0.0, 'boden_relevant': False,
            'beschreibung': 'Sehr windstill, kein Regen danach'},
        'Spritzen Insektizide': {
            'temp_min': 8, 'temp_max': 25, 'wind_max': 12, 'precip_max': 0.0,
            'boden_relevant': False,
            'beschreibung': 'Windstill, oft Abend/Morgen (Bienenschutz)'},
    },
    'Düngung': {
        'Güllefahren (flüssig)': {
            'temp_min': 5, 'temp_max': 30, 'humidity_max': 85, 'wind_max': 20,
            'precip_max': 0.5, 'boden_relevant': True,
            'beschreibung': 'Frostfrei, nicht zu nass, mäßiger Wind'},
        'Festmist ausbringen': {
            'temp_min': 0, 'temp_max': 30, 'wind_max': 30, 'precip_max': 1.0,
            'boden_relevant': True,
            'beschreibung': 'Boden tragfähig, frostfrei'},
        'Mineraldünger streuen': {
            'temp_min': 3, 'temp_max': 30, 'wind_max': 18, 'precip_max': 0.5,
            'boden_relevant': True,
            'beschreibung': 'Wenig Wind (gleichmäßige Streuung)'},
    },
    'Sonderkulturen': {
        'Weinbau Spritzung': {
            'temp_min': 10, 'temp_max': 28, 'humidity_min': 55, 'humidity_max': 95,
            'wind_max': 15, 'precip_max': 0.0, 'boden_relevant': False,
            'beschreibung': 'Windstill, kein Regen, mild'},
        'Obstbau Spritzung': {
            'temp_min': 10, 'temp_max': 28, 'humidity_min': 55, 'humidity_max': 95,
            'wind_max': 15, 'precip_max': 0.0, 'boden_relevant': False,
            'beschreibung': 'Windstill, kein Regen, mild'},
        'Obsternte': {
            'temp_min': 5, 'temp_max': 32, 'humidity_max': 85, 'wind_max': 30,
            'precip_max': 0.2, 'trockenfenster_h': 4, 'boden_relevant': False,
            'beschreibung': 'Trocken, Früchte nicht nass ernten'},
        'Gemüsebau Pflanzung': {
            'temp_min': 8, 'temp_max': 28, 'humidity_min': 40, 'wind_max': 20,
            'precip_max': 0.5, 'boden_relevant': True, 'duerre_relevant': True,
            'beschreibung': 'Mild, gleichmäßige Bodenfeuchte'},
    },
    'Bergsport & Outdoor': {
        'Hochtour / Gletscher': {
            'temp_min': -25, 'temp_max': 30, 'wind_max': 40, 'precip_max': 0.0,
            'trockenfenster_h': 0, 'boden_relevant': False,
            'beschreibung': 'Niederschlagsfrei, wenig Wind, früher Aufbruch. '
                            'HINWEIS: Werte auf Talniveau — in der Höhe deutlich '
                            'kälter/windiger. Höhenmodul folgt.'},
        'Klettern (Fels)': {
            'temp_min': 8, 'temp_max': 32, 'humidity_max': 90, 'wind_max': 30,
            'precip_max': 0.0, 'trockenfenster_h': 12, 'boden_relevant': False,
            'beschreibung': 'Fels muss abgetrocknet sein (12h trocken)'},
        'Wandern / Hüttentour': {
            'temp_min': 3, 'temp_max': 32, 'wind_max': 45, 'precip_max': 0.5,
            'boden_relevant': False,
            'beschreibung': 'Kein Dauerregen, vertretbarer Wind'},
        'Mountainbike / Trail': {
            'temp_min': 5, 'temp_max': 33, 'wind_max': 40, 'precip_max': 0.3,
            'trockenfenster_h': 6, 'boden_relevant': False,
            'beschreibung': 'Trails abgetrocknet, kein Regen'},
    },
    'Bau & Sonstiges': {
        'Baggerarbeiten / Erdarbeiten': {
            'temp_min': -2, 'temp_max': 38, 'wind_max': 45, 'precip_max': 1.0,
            'boden_relevant': True,
            'beschreibung': 'Boden nicht zu aufgeweicht, frostfrei'},
        'Betonieren': {
            'temp_min': 5, 'temp_max': 30, 'wind_max': 40, 'precip_max': 0.2,
            'trockenfenster_h': 12, 'boden_relevant': False,
            'beschreibung': 'Frostfrei, kein Regen während Abbinden'},
        'Malen / Fassade außen': {
            'temp_min': 8, 'temp_max': 30, 'humidity_max': 80, 'wind_max': 25,
            'precip_max': 0.0, 'trockenfenster_h': 12, 'boden_relevant': False,
            'beschreibung': 'Trocken, mäßige Feuchte zum Aushärten'},
        'Dachdecken / Außenarbeit': {
            'temp_min': 3, 'temp_max': 34, 'wind_max': 25, 'precip_max': 0.1,
            'boden_relevant': False,
            'beschreibung': 'Trocken, wenig Wind (Absturzgefahr)'},
        'Veranstaltung im Freien': {
            'temp_min': 12, 'temp_max': 33, 'wind_max': 30, 'precip_max': 0.1,
            'boden_relevant': False,
            'beschreibung': 'Trocken und angenehm'},
        'Rasen mähen (privat)': {
            'temp_min': 8, 'temp_max': 33, 'humidity_max': 90, 'wind_max': 35,
            'precip_max': 0.2, 'trockenfenster_h': 4, 'boden_relevant': False,
            'beschreibung': 'Gras trocken, kein Regen'},
    },
    'Frost': {
        'Frostüberwachung': {
            'modus': 'frost', 'temp_warn': 3, 'temp_frost': 0,
            'boden_relevant': False,
            'beschreibung': 'Überwachung Tiefsttemperaturen und Frostrisiko'},
    },
}


def finde_fall(name):
    """Sucht einen Anwendungsfall über alle Kategorien und gibt (kategorie, params)."""
    for kat, faelle in ANWENDUNGSFAELLE.items():
        if name in faelle:
            return kat, faelle[name]
    return None, None


# Fünfstufige Eignungsskala
STUFEN = {
    4: {'name': 'sehr gut',   'farbe': '#1b7a3d', 'kurz': 'sehr gut'},
    3: {'name': 'gut',        'farbe': '#6ab04c', 'kurz': 'gut'},
    2: {'name': 'bedingt',    'farbe': '#e8b62c', 'kurz': 'bedingt'},
    1: {'name': 'ungünstig',  'farbe': '#e07b2c', 'kurz': 'ungünstig'},
    0: {'name': 'ungeeignet', 'farbe': '#c0392b', 'kurz': 'ungeeignet'},
}
FARBEN = {v['name']: v['farbe'] for v in STUFEN.values()}
AMPEL_INT = {v['name']: k for k, v in STUFEN.items()}
INT_AMPEL = {k: v['name'] for k, v in STUFEN.items()}

# Bezugsgrößen für die Normierung der Grenzwertüberschreitung.
# Sie übersetzen eine absolute Überschreitung in einen Prozentwert,
# damit Temperatur, Feuchte, Wind und Niederschlag vergleichbar werden.
BEZUG_TEMP = 10.0        # °C entsprechen 100 % Überschreitung
BEZUG_FEUCHTE = 20.0     # Prozentpunkte entsprechen 100 %
BEZUG_WIND_MIN = 10.0    # km/h Untergrenze des Bezugs
BEZUG_NS_MIN = 1.5       # mm/h Untergrenze des Bezugs
WOCHENTAGE = {'Monday':'Mo','Tuesday':'Di','Wednesday':'Mi','Thursday':'Do',
              'Friday':'Fr','Saturday':'Sa','Sunday':'So'}

def tag_kurz(dt):
    return WOCHENTAGE.get(dt.strftime('%A'), dt.strftime('%a'))


# ============================================================
# DATEN ABRUFEN
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def geocode_ort(name):
    try:
        r = requests.get('https://geocoding-api.open-meteo.com/v1/search',
                         params={'name': name, 'count': 5, 'language': 'de'}, timeout=10)
        data = r.json()
    except Exception:
        return None
    if 'results' not in data or not data['results']:
        return None
    treffer = []
    for x in data['results']:
        hoehe = x.get('elevation')
        zusatz = f" · {hoehe:.0f} m" if hoehe is not None else ""
        treffer.append({
            'lat': x['latitude'], 'lon': x['longitude'],
            'hoehe': hoehe,
            'label': (f"{x['name']}, {x.get('admin1','')}, "
                      f"{x.get('country','')}{zusatz}")})
    return treffer


@st.cache_data(ttl=1800, show_spinner=False)
def hole_modellhoehe(lat, lon):
    """Seehöhe des Modellgitterpunkts — Referenz für die Höhenkorrektur."""
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast',
                         params={'latitude': lat, 'longitude': lon,
                                 'hourly': 'temperature_2m',
                                 'forecast_days': 1,
                                 'timezone': 'Europe/Vienna'}, timeout=10)
        return r.json().get('elevation')
    except Exception:
        return None


def hoehenkorrektur(df, delta_h, spalten=None):
    """
    Rechnet Modellwerte auf die tatsächliche Standorthöhe um.
    delta_h = Standorthöhe − Modellhöhe (positiv = Standort liegt höher).
    Temperatur und Taupunkt sinken mit der Höhe, Wind nimmt zu.
    """
    if not delta_h or abs(delta_h) < 100:
        return df
    df = df.copy()
    dt = delta_h / 100.0 * TEMP_GRADIENT_PRO_100M
    windfaktor = 1.0 + (delta_h / 1000.0) * WIND_FAKTOR_PRO_1000M
    windfaktor = float(np.clip(windfaktor, 0.6, 2.5))

    for sp in (spalten or ['temp', 'temp_median', 'temp_p10', 'temp_p90',
                           'taupunkt']):
        if sp in df.columns:
            df[sp] = pd.to_numeric(df[sp], errors='coerce') - dt
    for sp in ['wind', 'wind_median', 'boeen']:
        if sp in df.columns:
            df[sp] = pd.to_numeric(df[sp], errors='coerce') * windfaktor
    return df


def himmelsrichtung(grad):
    """Wandelt Gradzahl in Himmelsrichtungs-Kürzel."""
    if grad is None or (isinstance(grad, float) and np.isnan(grad)):
        return '–'
    richtungen = ['N', 'NNO', 'NO', 'ONO', 'O', 'OSO', 'SO', 'SSO',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return richtungen[int((float(grad) + 11.25) % 360 // 22.5)]


@st.cache_data(ttl=1800, show_spinner=False)
def hole_hauptlauf(lat, lon, modell_name, tage):
    cfg = HAUPTLAUFE[modell_name]
    params = {'latitude': lat, 'longitude': lon, 'hourly': HAUPT_VARIABLEN,
              'forecast_days': min(tage, cfg['max_tage']), 'timezone': 'Europe/Vienna'}
    if 'url_param' in cfg:
        params['models'] = cfg['url_param']
    try:
        r = requests.get(cfg['url'], params=params, timeout=15)
        data = r.json()
    except Exception:
        return None
    if 'hourly' not in data:
        return None
    h = data['hourly']; n = len(h['time'])
    def hol(key):
        v = h.get(key)
        return v if v else [None]*n

    df = pd.DataFrame({'time': pd.to_datetime(h['time']),
                       'temp': hol('temperature_2m'),
                       'feuchte': hol('relative_humidity_2m'),
                       'niederschlag': hol('precipitation'),
                       'wind': hol('wind_speed_10m'),
                       'boeen': hol('wind_gusts_10m'),
                       'windrichtung': hol('wind_direction_10m'),
                       'wolken': hol('cloud_cover'),
                       'wolken_tief': hol('cloud_cover_low'),
                       'wolken_mittel': hol('cloud_cover_mid'),
                       'wolken_hoch': hol('cloud_cover_high'),
                       'taupunkt': hol('dew_point_2m')})
    return df.dropna(subset=['temp'])


@st.cache_data(ttl=1800, show_spinner=False)
def hole_ensemble(lat, lon, modell_name, tage):
    cfg = ENSEMBLE_MODELLE[modell_name]
    params = {'latitude': lat, 'longitude': lon, 'hourly': ENS_VARIABLEN,
              'models': cfg['model_key'], 'forecast_days': min(tage, 16),
              'timezone': 'Europe/Vienna'}
    try:
        r = requests.get(ENSEMBLE_API, params=params, timeout=30)
        data = r.json()
    except Exception:
        return None
    if 'hourly' not in data:
        return None
    h = data['hourly']
    zeiten = pd.to_datetime(h['time']); n = len(zeiten)

    def sammle(prefix, n_max=60):
        m = []
        for i in range(n_max):
            key = f'{prefix}_member{i:02d}'
            if key in h and h[key] is not None:
                vals = h[key]
                if any(v is not None for v in vals):
                    m.append([v if v is not None else np.nan for v in vals])
        return np.array(m) if m else None

    temp_m = sammle('temperature_2m')
    if temp_m is None:
        return None
    feuchte_m = sammle('relative_humidity_2m')
    wind_m = sammle('wind_speed_10m')
    ns_m = sammle('precipitation')
    n_mitgl = temp_m.shape[0]

    def med(arr):
        return np.nanmedian(arr, axis=0) if arr is not None else [None]*n

    t_med = np.nanmedian(temp_m, axis=0)
    t_p10 = np.nanpercentile(temp_m, 10, axis=0)
    t_p90 = np.nanpercentile(temp_m, 90, axis=0)

    if ns_m is not None:
        ns_med = np.nanmedian(ns_m, axis=0)
        ns_wahr = (ns_m > 0.1).mean(axis=0) * 100
    else:
        ns_med = np.zeros(n); ns_wahr = np.zeros(n)

    df = pd.DataFrame({'time': zeiten, 'temp_median': t_med,
                       'temp_p10': t_p10, 'temp_p90': t_p90,
                       'temp_spread': t_p90 - t_p10,
                       'feuchte_median': med(feuchte_m),
                       'wind_median': med(wind_m),
                       'ns_median': ns_med, 'ns_wahrscheinlichkeit': ns_wahr,
                       'n_mitglieder': n_mitgl})
    return df.dropna(subset=['temp_median'])


SOIL_LAYERS = [
    ('soil_moisture_0_to_1cm',   '0–1 cm',   'Oberfläche'),
    ('soil_moisture_1_to_3cm',   '1–3 cm',   'Saatbett'),
    ('soil_moisture_3_to_9cm',   '3–9 cm',   'Wurzelraum oben'),
    ('soil_moisture_9_to_27cm',  '9–27 cm',  'Wurzelraum tief'),
]


@st.cache_data(ttl=1800, show_spinner=False)
def hole_bodendaten(lat, lon, tage):
    """Vergangenheit (10 Tage) + Zukunft: Regen, Verdunstung, Bodenfeuchte (4 Schichten)"""
    stundenvars = ','.join([k for k, _, _ in SOIL_LAYERS]) + ',precipitation'
    params = {'latitude': lat, 'longitude': lon,
              'daily': 'precipitation_sum,et0_fao_evapotranspiration',
              'hourly': stundenvars,
              'past_days': BODEN_VORGESCHICHTE_TAGE,
              'forecast_days': min(tage, 16), 'timezone': 'Europe/Vienna'}
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=20)
        data = r.json()
    except Exception:
        return None
    if 'daily' not in data:
        return None
    d = data['daily']
    df_daily = pd.DataFrame({
        'datum': pd.to_datetime(d['time']).date,
        'precip': [x if x is not None else 0 for x in d.get('precipitation_sum', [])],
        'et0': [x if x is not None else 3.5 for x in d.get('et0_fao_evapotranspiration', [])],
    })

    # --- stündliche Bodenfeuchte je Schicht + Niederschlag ---
    df_hourly = None
    soil_now = None
    if 'hourly' in data:
        h = data['hourly']
        spalten = {'time': pd.to_datetime(h['time'])}
        n = len(h['time'])
        for key, label, _ in SOIL_LAYERS:
            werte = h.get(key)
            if werte:
                spalten[label] = [v if v is not None else np.nan for v in werte]
        ns = h.get('precipitation')
        spalten['precip_h'] = ([v if v is not None else 0.0 for v in ns]
                               if ns else [0.0]*n)
        df_hourly = pd.DataFrame(spalten)

        # aktueller Wert der tiefsten verfügbaren Schicht
        jetzt = datetime.now()
        for _, label, _ in reversed(SOIL_LAYERS):
            if label in df_hourly.columns:
                gueltig = df_hourly[(df_hourly['time'] <= jetzt)
                                    & df_hourly[label].notna()]
                if not gueltig.empty:
                    soil_now = float(gueltig.iloc[-1][label])
                    break

    return {'daily': df_daily, 'hourly': df_hourly, 'soil_moisture_now': soil_now}


def berechne_infiltration(df_hourly):
    """
    Schätzt je Stunde, wieviel Regen tatsächlich in den Boden geht
    (effektiver Niederschlag) und wieviel oberflächlich abläuft.

    Grundgedanke (vereinfachte Infiltrationsphysik):
      • Trockener Boden nimmt grundsätzlich mehr auf ALS feuchter …
      • … ABER nur bis zu einer maximalen Aufnahmerate. Sehr intensiver
        Regen (Gewitter) übersteigt diese Rate: das Wasser läuft ab,
        obwohl der Boden trocken ist.
      • Bei bereits nassem Boden sinkt die Aufnahmerate stark.
    """
    if df_hourly is None or 'precip_h' not in df_hourly.columns:
        return None

    # Referenz-Schicht: oberste verfügbare
    ref_spalte = None
    for _, label, _ in SOIL_LAYERS:
        if label in df_hourly.columns:
            ref_spalte = label
            break
    if ref_spalte is None:
        return None

    df = df_hourly.copy()
    feuchte_start = df[ref_spalte].fillna(0.25).values
    regen = df['precip_h'].values

    # Dynamische Simulation: die oberste Schicht füllt sich beim Regen,
    # gibt aber laufend Wasser an tiefere Schichten ab (Perkolation).
    # Dadurch kann langsamer Landregen fast vollständig einsickern,
    # während kurzer Starkregen die Aufnahmerate übersteigt und abläuft.
    SAETTIGUNG = 0.45        # Feuchte, ab der praktisch nichts mehr reingeht
    PERKOLATION = 0.9        # mm/h, die aus der obersten Schicht nach unten abfließen

    max_rate_arr, eff_arr, ab_arr = [], [], []
    feuchte = float(feuchte_start[0]) if len(feuchte_start) else 0.25

    for i, r in enumerate(regen):
        # Modellwert als Anker nutzen, damit wir nicht wegdriften
        if not np.isnan(feuchte_start[i]):
            feuchte = 0.5 * feuchte + 0.5 * float(feuchte_start[i])

        # Maximale Aufnahmerate (mm/h) je nach Feuchte:
        #   sehr trocken (0.10) → ~11 mm/h
        #   normal       (0.25) → ~ 8 mm/h
        #   nass         (0.40) → ~ 3 mm/h
        mr = float(np.clip(14.0 - 27.0 * feuchte, 0.8, 14.0))

        rein = min(float(r), mr)
        ab = max(0.0, float(r) - mr)

        # Speicheränderung: Zufluss minus Perkolation nach unten
        feuchte = feuchte + rein / 120.0 - PERKOLATION / 120.0
        feuchte = float(np.clip(feuchte, 0.05, SAETTIGUNG))

        max_rate_arr.append(mr)
        eff_arr.append(rein)
        ab_arr.append(ab)

    df['max_aufnahme'] = max_rate_arr
    df['ns_effektiv'] = eff_arr
    df['ns_abfluss'] = ab_arr
    return df


# ============================================================
# BODEN-INDEX (Wasserbilanz + Kennzahlen)
# ============================================================

def berechne_bodenindex(boden):
    """
    Liefert:
      per_datum_level: {datum: (level 0-3, grund)}
      per_datum_dry:   {datum: (drylevel 0-2, grund)}
      metriken: dict für die Übersicht
    """
    if boden is None or boden['daily'].empty:
        return None

    df = boden['daily'].sort_values('datum').reset_index(drop=True)
    heute = datetime.now().date()

    # --- Wasserbilanz über alle Tage (Vergangenheit + Zukunft) ---
    # Reihenfolge je Tag: zuerst Regen aufnehmen, dann Verlust abziehen.
    # Drainage wirkt anteilig auf die Bilanz (viel Wasser drainiert absolut
    # mehr, trocknet aber relativ langsamer aus als eine feste Menge) plus
    # Verdunstung. So bleibt ein großes Regenereignis realistisch länger
    # spürbar, während nach ~10 Trockentagen wieder abgetrocknet ist.
    balance = 0.0
    per_datum_balance = {}
    for _, row in df.iterrows():
        balance += row['precip']                      # Regen aufnehmen
        drainage = BODEN_DRAINAGE_MM * (0.5 + 0.5 * min(balance, 60) / 60)
        verlust = row['et0'] + drainage
        balance = max(0.0, balance - verlust)         # Verluste abziehen
        per_datum_balance[row['datum']] = balance

    # --- Kennzahlen aus der Vergangenheit ---
    vergangen = df[df['datum'] <= heute]
    regen_3t = vergangen.tail(3)['precip'].sum()
    regen_7t = vergangen.tail(7)['precip'].sum()
    regen_10t = vergangen.tail(10)['precip'].sum()

    # Tage seit letztem nennenswerten Regen (>3 mm)
    tage_seit_regen = 0
    for _, row in vergangen.iloc[::-1].iterrows():
        if row['precip'] >= 3:
            break
        tage_seit_regen += 1

    # --- Level je Datum aus der Bilanz ---
    def bilanz_level(b):
        if b >= BILANZ_EXTREM: return 3
        if b >= BILANZ_NASS: return 2
        if b >= BILANZ_FEUCHT: return 1
        return 0
    level_texte = {0: 'Boden abgetrocknet', 1: 'Böden feucht',
                   2: 'Böden nass', 3: 'Böden sehr nass – nicht befahrbar'}

    per_datum_level = {}
    for datum, b in per_datum_balance.items():
        lv = bilanz_level(b)
        per_datum_level[datum] = (lv, level_texte[lv])

    # --- Dürre je Datum (für Saat) ---
    # dry_level steigt mit Trockenheit; in der Zukunft weiter, solange kein Regen
    per_datum_dry = {}
    trockentage = tage_seit_regen
    for _, row in df.iterrows():
        if row['datum'] < heute:
            continue
        if row['datum'] > heute:
            if row['precip'] >= 3:
                trockentage = 0
            else:
                trockentage += 1
        b = per_datum_balance[row['datum']]
        if trockentage >= 12 and b < BILANZ_FEUCHT:
            per_datum_dry[row['datum']] = (2, 'Dürre – Aussaat ungünstig')
        elif trockentage >= 7 and b < BILANZ_FEUCHT:
            per_datum_dry[row['datum']] = (1, 'sehr trocken')
        else:
            per_datum_dry[row['datum']] = (0, '')

    # --- aktueller Zustand (heute) für die Übersicht ---
    # Falls "heute" nicht exakt in den Tagesdaten liegt, nimm den letzten
    # verfügbaren Vergangenheitstag als aktuellen Zustand.
    if heute in per_datum_level:
        heute_level = per_datum_level[heute]
        balance_heute = per_datum_balance[heute]
    else:
        vergangene_daten = [d for d in per_datum_level if d <= heute]
        ref = max(vergangene_daten) if vergangene_daten else min(per_datum_level)
        heute_level = per_datum_level[ref]
        balance_heute = per_datum_balance[ref]

    # Bodenfeuchte-Status aus Modell (falls vorhanden)
    sm = boden.get('soil_moisture_now')
    if sm is None:
        sm_status = None
    elif sm < 0.15:
        sm_status = 'trocken'
    elif sm < 0.30:
        sm_status = 'normal'
    elif sm < 0.40:
        sm_status = 'feucht'
    else:
        sm_status = 'nass'

    metriken = {
        'regen_3t': regen_3t, 'regen_7t': regen_7t, 'regen_10t': regen_10t,
        'tage_seit_regen': tage_seit_regen,
        'balance_heute': balance_heute,
        'zustand_level': heute_level[0], 'zustand_text': heute_level[1],
        'soil_moisture': sm, 'soil_status': sm_status,
    }

    return {'per_datum_level': per_datum_level, 'per_datum_dry': per_datum_dry,
            'metriken': metriken}


def boden_modifikator(ampel_int, af, datum, bodenindex):
    """Wendet den Bodeneinfluss auf einen Ampelwert an. Gibt (neuer_int, grund) zurück."""
    if bodenindex is None:
        return ampel_int, None
    grund = None
    ai = ampel_int

    if af.get('boden_relevant'):
        lv, lv_text = bodenindex['per_datum_level'].get(datum, (0, ''))
        if lv >= 3:
            ai = 0; grund = lv_text
        elif lv == 2:
            ai = max(0, ai - 2); grund = lv_text
        elif lv == 1:
            ai = max(0, ai - 1); grund = lv_text

    if af.get('duerre_relevant'):
        dry, dry_text = bodenindex['per_datum_dry'].get(datum, (0, ''))
        if dry >= 2:
            ai = max(0, ai - 2)
            grund = (grund + '; ' if grund else '') + dry_text
        elif dry == 1:
            ai = max(0, ai - 1)
            grund = (grund + '; ' if grund else '') + dry_text

    return ai, grund


# ============================================================
# BEWERTUNG
# ============================================================

def bewerte_werte(temp, feuchte, wind, ns, af):
    """
    Bewertet Einzelwerte gegen ein Parameter-Dict af.
    Rückgabe: (stufe 0–4, begruendungen)

    Grundgedanke: Für jeden Parameter wird die Überschreitung des Grenzwerts
    auf einen Prozentwert normiert. Aus der Zahl und der Höhe der
    Überschreitungen ergibt sich die Eignungsstufe. Dadurch wird ein einzelner
    kleiner Ausreißer anders gewichtet als mehrere deutliche Verstöße.
    """
    if af.get('modus') == 'frost':
        if temp is None:
            return 0, ['keine Daten']
        if temp <= af.get('temp_frost', 0):
            return 0, [f'Frost ({temp:.1f} °C)']
        if temp <= af.get('temp_warn', 3):
            return 2, [f'Frostgefahr ({temp:.1f} °C)']
        if temp <= af.get('temp_warn', 3) + 2:
            return 3, [f'grenzwertig ({temp:.1f} °C)']
        return 4, [f'frostfrei ({temp:.1f} °C)']

    ueber = []      # Liste von (anteil, begruendung)

    def pruefe(wert, grenze, bezug, richtung, text_fmt):
        """richtung 'max' = Wert darf Grenze nicht überschreiten, 'min' umgekehrt."""
        if wert is None or grenze is None:
            return
        try:
            wert = float(wert); grenze = float(grenze)
        except (TypeError, ValueError):
            return
        if np.isnan(wert):
            return
        diff = (wert - grenze) if richtung == 'max' else (grenze - wert)
        if diff <= 0:
            return
        anteil = diff / max(bezug, 1e-6)
        ueber.append((anteil, text_fmt(wert, anteil)))

    # Temperatur
    if 'temp_max' in af:
        pruefe(temp, af['temp_max'], BEZUG_TEMP, 'max',
               lambda w, a: f'zu warm ({w:.0f} °C)')
    if 'temp_min' in af:
        pruefe(temp, af['temp_min'], BEZUG_TEMP, 'min',
               lambda w, a: f'zu kalt ({w:.0f} °C)')

    # Luftfeuchte
    if 'humidity_max' in af:
        pruefe(feuchte, af['humidity_max'], BEZUG_FEUCHTE, 'max',
               lambda w, a: f'zu feucht ({w:.0f} %)')
    if 'humidity_min' in af:
        pruefe(feuchte, af['humidity_min'], BEZUG_FEUCHTE, 'min',
               lambda w, a: f'zu trocken ({w:.0f} %)')

    # Wind — Bezug ist der Grenzwert selbst (Verhältnisskala)
    if 'wind_max' in af:
        bezug_w = max(float(af['wind_max']), BEZUG_WIND_MIN)
        pruefe(wind, af['wind_max'], bezug_w, 'max',
               lambda w, a: f'zu windig ({w:.0f} km/h)')

    # Niederschlag — bei Grenzwert 0 zählt jeder messbare Regen
    if 'precip_max' in af:
        bezug_n = max(float(af['precip_max']), BEZUG_NS_MIN)

        def ns_text(w, a):
            if w >= 2.5:
                return f'Starkregen ({w:.1f} mm/h)'
            if w >= 0.5:
                return f'Regen ({w:.1f} mm/h)'
            return f'leichter Regen ({w:.1f} mm/h)'

        pruefe(ns, af['precip_max'], bezug_n, 'max', ns_text)

    if not ueber:
        return 4, ['alle Kriterien erfüllt']

    anteile = sorted((a for a, _ in ueber), reverse=True)
    gruende = [t for _, t in sorted(ueber, key=lambda x: -x[0])]
    groesste = anteile[0]
    n_ab_20 = sum(1 for a in anteile if a >= 0.20)
    n_ab_30 = sum(1 for a in anteile if a >= 0.30)

    # Stufenzuordnung: von der schwersten Ausprägung abwärts prüfen
    if groesste >= 0.50 or n_ab_30 >= 2:
        return 0, gruende
    if groesste >= 0.35 or n_ab_20 >= 2:
        return 1, gruende
    if groesste >= 0.15 or len(anteile) >= 2:
        return 2, gruende
    return 3, gruende


def bewerte_ensemble_df(ens_df, af):
    erg = ens_df.apply(lambda r: bewerte_werte(
        r['temp_median'], r['feuchte_median'], r['wind_median'],
        r['ns_median'], af), axis=1)
    ens_df = ens_df.copy()
    ens_df['ampel_int'] = [e[0] for e in erg]
    return ens_df


# ============================================================
# LADEN
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def hole_sonnenzeiten(lat, lon, tage):
    """Sonnenauf- und -untergang je Tag."""
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast',
                         params={'latitude': lat, 'longitude': lon,
                                 'daily': 'sunrise,sunset',
                                 'forecast_days': min(tage, 16),
                                 'timezone': 'Europe/Vienna'}, timeout=10)
        d = r.json().get('daily', {})
    except Exception:
        return None
    if not d or 'sunrise' not in d:
        return None
    return pd.DataFrame({
        'datum': pd.to_datetime(d['time']).date,
        'aufgang': pd.to_datetime(d['sunrise']),
        'untergang': pd.to_datetime(d['sunset'])})


def lade_alle_daten(lat, lon, af, tage, fortschritt=None, standorthoehe=None):
    schritte = len(HAUPTLAUFE) + len(ENSEMBLE_MODELLE) + 2
    schritt = 0
    haupt_daten, ensemble_daten = {}, {}

    # Höhendifferenz Standort ↔ Modellgitter bestimmen
    modellhoehe = hole_modellhoehe(lat, lon)
    delta_h = None
    if standorthoehe is not None and modellhoehe is not None:
        delta_h = float(standorthoehe) - float(modellhoehe)

    for name in HAUPTLAUFE:
        schritt += 1
        if fortschritt: fortschritt.progress(schritt/schritte, text=f"Lade {name} …")
        df = hole_hauptlauf(lat, lon, name, tage)
        if df is not None and not df.empty:
            haupt_daten[name] = hoehenkorrektur(df, delta_h)

    for name, cfg in ENSEMBLE_MODELLE.items():
        schritt += 1
        if fortschritt:
            fortschritt.progress(schritt/schritte,
                                 text=f"Lade {name} ({cfg['mitglieder']} Mitgl.) …")
        df = hole_ensemble(lat, lon, name, tage)
        if df is not None and not df.empty:
            df = hoehenkorrektur(df, delta_h)
            ensemble_daten[name] = bewerte_ensemble_df(df, af)

    schritt += 1
    if fortschritt: fortschritt.progress(schritt/schritte, text="Lade Bodendaten …")
    boden = hole_bodendaten(lat, lon, tage)
    bodenindex = berechne_bodenindex(boden)
    infil = berechne_infiltration(boden['hourly']) if boden else None

    schritt += 1
    if fortschritt: fortschritt.progress(schritt/schritte, text="Lade Sonnenstände …")
    sonne = hole_sonnenzeiten(lat, lon, tage)

    return {'haupt': haupt_daten, 'ensemble': ensemble_daten,
            'bodenindex': bodenindex, 'boden': boden, 'infiltration': infil,
            'sonne': sonne, 'modellhoehe': modellhoehe,
            'standorthoehe': standorthoehe, 'delta_h': delta_h}


# ============================================================
# KONSENS
# ============================================================

def berechne_konsens(alle_daten, af, tage):
    haupt = alle_daten['haupt']
    ensemble = alle_daten['ensemble']
    bodenindex = alle_daten.get('bodenindex')

    alle_zeiten = set()
    for df in haupt.values(): alle_zeiten |= set(df['time'])
    for df in ensemble.values(): alle_zeiten |= set(df['time'])
    alle_zeiten = sorted(alle_zeiten)

    zeilen = []
    for zeit in alle_zeiten:
        stunden_voraus = (zeit - datetime.now()).total_seconds() / 3600
        votes = []
        temps, feuchten, winde, ns_vals, ns_wahrs = [], [], [], [], []
        boeen_l, wrichtung_l, wolken_l = [], [], []
        wolken_t_l, wolken_m_l, wolken_h_l, taupunkt_l = [], [], [], []
        n_ens_mitglieder = 0
        n_haupt_laeufe = 0
        temp_spread = 0

        for name, cfg in ENSEMBLE_MODELLE.items():
            if name not in ensemble: continue
            z = ensemble[name][ensemble[name]['time'] == zeit]
            if z.empty: continue
            z = z.iloc[0]
            votes.append((z['ampel_int'], cfg['mitglieder']/10))
            n_ens_mitglieder += cfg['mitglieder']
            if z['temp_median'] is not None: temps.append(z['temp_median'])
            if z['feuchte_median'] is not None: feuchten.append(z['feuchte_median'])
            if z['wind_median'] is not None: winde.append(z['wind_median'])
            ns_vals.append(z['ns_median']); ns_wahrs.append(z['ns_wahrscheinlichkeit'])
            temp_spread = max(temp_spread, z['temp_spread'])

        for name, df in haupt.items():
            z = df[df['time'] == zeit]
            if z.empty: continue
            z = z.iloc[0]
            ai, _ = bewerte_werte(z['temp'], z['feuchte'], z['wind'],
                                  z['niederschlag'], af)
            votes.append((ai, 1.0))
            n_haupt_laeufe += 1
            if z['temp'] is not None: temps.append(z['temp'])
            if z['feuchte'] is not None: feuchten.append(z['feuchte'])
            if z['wind'] is not None: winde.append(z['wind'])
            if z['niederschlag'] is not None:
                ns_vals.append(z['niederschlag'])
                ns_wahrs.append(100 if z['niederschlag'] > 0.1 else 0)
            for feld, ziel in (('boeen', boeen_l), ('windrichtung', wrichtung_l),
                               ('wolken', wolken_l), ('wolken_tief', wolken_t_l),
                               ('wolken_mittel', wolken_m_l),
                               ('wolken_hoch', wolken_h_l),
                               ('taupunkt', taupunkt_l)):
                if feld in z.index and z[feld] is not None and not pd.isna(z[feld]):
                    ziel.append(float(z[feld]))

        if not votes: continue

        gesamt_g = sum(g for _, g in votes)
        konsens_wert = sum(a*g for a, g in votes) / gesamt_g

        # Abrunden statt kaufmännisch runden: Die Bewertung soll im Zweifel
        # die vorsichtigere Stufe wählen.
        if konsens_wert >= 3.60: k_int = 4
        elif konsens_wert >= 2.60: k_int = 3
        elif konsens_wert >= 1.60: k_int = 2
        elif konsens_wert >= 0.70: k_int = 1
        else: k_int = 0

        # Bodeneinfluss
        boden_grund = None
        if bodenindex is not None:
            k_int, boden_grund = boden_modifikator(
                k_int, af, zeit.date(), bodenindex)

        k_ampel = INT_AMPEL[k_int]

        # Sicherheit
        ints = [a for a, _ in votes]
        std = np.std(ints)
        if stunden_voraus <= 48:
            sicherheit = ('sehr hoch' if std < 0.45 else 'hoch' if std < 0.95
                          else 'mittel' if std < 1.45 else 'gering')
        elif stunden_voraus <= 120:
            sicherheit = ('hoch' if std < 0.45 else 'mittel' if std < 1.05
                          else 'gering')
        else:
            sicherheit = 'mittel' if std < 0.6 else 'gering'

        # Begründung aus Medianwerten
        temp_med = float(np.median(temps)) if temps else None
        feuchte_med = float(np.median(feuchten)) if feuchten else None
        wind_med = float(np.median(winde)) if winde else None
        ns_med = float(np.median(ns_vals)) if ns_vals else 0
        ns_wahr = float(np.mean(ns_wahrs)) if ns_wahrs else 0
        _, wetter_gruende = bewerte_werte(temp_med, feuchte_med, wind_med, ns_med, af)
        if boden_grund:
            wetter_gruende = [boden_grund] + wetter_gruende

        zeilen.append({
            'time': zeit, 'ampel': k_ampel, 'ampel_int': k_int,
            'sicherheit': sicherheit, 'stunden_voraus': stunden_voraus,
            'n_laeufe': n_ens_mitglieder + n_haupt_laeufe,
            'n_ens_mitglieder': n_ens_mitglieder, 'n_haupt': n_haupt_laeufe,
            'temp_median': temp_med, 'temp_spread': float(temp_spread),
            'feuchte_median': feuchte_med, 'wind_median': wind_med,
            'ns_median': ns_med, 'ns_wahrscheinlichkeit': ns_wahr,
            'boeen': float(np.median(boeen_l)) if boeen_l else None,
            'windrichtung': (float(np.median(wrichtung_l))
                             if wrichtung_l else None),
            'wolken': float(np.median(wolken_l)) if wolken_l else None,
            'wolken_tief': float(np.median(wolken_t_l)) if wolken_t_l else None,
            'wolken_mittel': float(np.median(wolken_m_l)) if wolken_m_l else None,
            'wolken_hoch': float(np.median(wolken_h_l)) if wolken_h_l else None,
            'taupunkt': float(np.median(taupunkt_l)) if taupunkt_l else None,
            'gruende': '; '.join(wetter_gruende), 'boden_grund': boden_grund or '',
        })

    return pd.DataFrame(zeilen)


def wende_trockenfenster_an(konsens_df, af):
    """
    Für Anwendungen mit trockenfenster_h > 0 (z. B. Heuernte): Eine Stunde ist
    nur dann geeignet zum BEGINNEN, wenn die folgenden X Stunden trocken bleiben.
    Dämpft Stunden, nach denen Regen im Fenster liegt, und markiert, wenn das
    Fenster über die Prognose hinausreicht (nicht bestätigbar).
    """
    tf = int(af.get('trockenfenster_h', 0) or 0)
    if tf <= 0 or konsens_df.empty:
        return konsens_df

    df = konsens_df.sort_values('time').reset_index(drop=True)
    regen = df['ns_median'].fillna(0).values
    n = len(df)
    schwelle = max(af.get('precip_max', 0.0), 0.2)

    neue_int = df['ampel_int'].tolist()
    neue_gruende = df['gruende'].tolist()

    for i in range(n):
        if neue_int[i] == 0:
            continue
        ende = min(n, i + tf)
        fenster = regen[i:ende]
        reicht = (ende - i) >= tf
        summe = float(np.nansum(fenster))
        max_h = float(np.nanmax(fenster)) if len(fenster) else 0.0

        if max_h > 2.0 or summe > 5.0:
            neue_int[i] = 0
            neue_gruende[i] = f'erheblicher Regen im {tf}-h-Fenster' + (
                '; ' + neue_gruende[i] if neue_gruende[i] else '')
        elif max_h > 0.8 or summe > 2.5:
            neue_int[i] = min(neue_int[i], 1)
            neue_gruende[i] = f'Regen im {tf}-h-Fenster' + (
                '; ' + neue_gruende[i] if neue_gruende[i] else '')
        elif max_h > schwelle or summe > 1.0:
            neue_int[i] = min(neue_int[i], 2)
            neue_gruende[i] = f'etwas Regen im {tf}-h-Fenster' + (
                '; ' + neue_gruende[i] if neue_gruende[i] else '')
        elif not reicht:
            neue_int[i] = min(neue_int[i], 3)
            neue_gruende[i] = (f'{tf}-h-Trockenfenster reicht über den '
                               f'Prognosezeitraum hinaus')

    df['ampel_int'] = neue_int
    df['ampel'] = [INT_AMPEL[a] for a in neue_int]
    df['gruende'] = neue_gruende
    return df


def finde_bestes_fenster(konsens_df):
    """Längster zusammenhängender Block der besten erreichbaren Stufe."""
    for ziel in ['sehr gut', 'gut', 'bedingt']:
        bloecke, akt = [], None
        for _, row in konsens_df.iterrows():
            if row['ampel'] == ziel:
                if akt and (row['time'] - akt['ende']) <= timedelta(hours=1):
                    akt['ende'] = row['time']; akt['stunden'] += 1
                else:
                    if akt: bloecke.append(akt)
                    akt = {'start': row['time'], 'ende': row['time'], 'stunden': 1,
                           'sicherheit': row['sicherheit']}
            else:
                if akt: bloecke.append(akt); akt = None
        if akt: bloecke.append(akt)
        if bloecke:
            best = max(bloecke, key=lambda x: x['stunden'])
            return ziel, best, len(bloecke)
    return None, None, 0


# ============================================================
# PLOTLY-HEATMAP (24h, interaktiv)
# ============================================================

# ============================================================
# WETTERÜBERSICHT MIT SYMBOLEN
# ============================================================

TAGESABSCHNITTE = [
    ('Vormittag', 6, 11),
    ('Mittag', 12, 14),
    ('Nachmittag', 15, 18),
    ('Abend/Nacht', 19, 5),   # über Mitternacht
]


def _wettersymbol(wolken, ns_mm, ns_wahr, temp, ist_tag):
    """
    Leitet ein Wettersymbol aus Bewölkung, Niederschlag und Tageszeit ab.
    Rückgabe: (symbol, kurzbeschreibung)
    """
    wolken = 0.0 if wolken is None or pd.isna(wolken) else float(wolken)
    ns_mm = 0.0 if ns_mm is None or pd.isna(ns_mm) else float(ns_mm)
    ns_wahr = 0.0 if ns_wahr is None or pd.isna(ns_wahr) else float(ns_wahr)
    kalt = temp is not None and not pd.isna(temp) and float(temp) <= 1.0

    # Niederschlag dominiert die Darstellung
    if ns_mm >= 1.5 or (ns_wahr >= 70 and ns_mm >= 0.5):
        return ('❄️', 'Schneefall') if kalt else ('🌧️', 'Regen')
    if ns_mm >= 0.4 or ns_wahr >= 55:
        return ('🌨️', 'Schneeschauer') if kalt else ('🌦️', 'Schauer')
    if ns_mm >= 0.1 or ns_wahr >= 35:
        return ('🌦️', 'einzelne Schauer')

    # Sonst nach Bewölkung
    if wolken >= 85:
        return ('☁️', 'bedeckt')
    if wolken >= 60:
        return ('🌥️', 'stark bewölkt')
    if wolken >= 30:
        return ('⛅', 'wechselnd bewölkt') if ist_tag else ('☁️', 'bewölkt')
    if wolken >= 12:
        return ('🌤️', 'heiter') if ist_tag else ('🌙', 'gering bewölkt')
    return ('☀️', 'sonnig') if ist_tag else ('🌙', 'klar')


def baue_wetteruebersicht(konsens_df, max_tage=7):
    """
    Kompakte Übersicht: Zeilen = Tagesabschnitte, Spalten = Tage.
    Rückgabe: (kopfzeilen, zeilen) für die Darstellung als HTML-Tabelle.
    """
    if konsens_df is None or konsens_df.empty:
        return None, None

    df = konsens_df.copy()
    df['datum'] = df['time'].dt.date
    df['stunde'] = df['time'].dt.hour

    tage = sorted(df['datum'].unique())[:max_tage]
    kopf = [f"{tag_kurz(pd.Timestamp(t))}<br><span style='opacity:.65'>"
            f"{pd.Timestamp(t).strftime('%d.%m.')}</span>" for t in tage]

    zeilen = []
    for label, von, bis in TAGESABSCHNITTE:
        zellen = []
        for tag in tage:
            if von <= bis:
                teil = df[(df['datum'] == tag)
                          & (df['stunde'] >= von) & (df['stunde'] <= bis)]
            else:
                # Abschnitt über Mitternacht: Abend des Tages + frühe Stunden
                teil = df[((df['datum'] == tag) & (df['stunde'] >= von))
                          | ((df['datum'] == tag) & (df['stunde'] <= bis))]
            if teil.empty:
                zellen.append(None)
                continue

            wolken = teil['wolken'].astype(float).mean() if 'wolken' in teil else np.nan
            ns_mm = teil['ns_median'].astype(float).max()
            ns_wahr = teil['ns_wahrscheinlichkeit'].astype(float).mean()
            temp_mit = teil['temp_median'].astype(float).mean()
            temp_max = teil['temp_median'].astype(float).max()
            temp_min = teil['temp_median'].astype(float).min()
            wind = (teil['wind_median'].astype(float).mean()
                    if 'wind_median' in teil else np.nan)
            ist_tag = von <= 18
            sym, beschr = _wettersymbol(wolken, ns_mm, ns_wahr, temp_mit, ist_tag)

            # Eignungsstufe des Abschnitts (schlechteste Stunde zählt)
            stufe = int(teil['ampel_int'].min()) if 'ampel_int' in teil else None

            zellen.append({
                'symbol': sym, 'beschreibung': beschr,
                'temp': temp_mit, 'temp_min': temp_min, 'temp_max': temp_max,
                'ns_mm': ns_mm, 'ns_wahr': ns_wahr, 'wind': wind,
                'stufe': stufe})
        zeilen.append((label, zellen))

    return kopf, zeilen


def wetteruebersicht_html(kopf, zeilen):
    """Rendert die Übersicht als schlanke HTML-Tabelle."""
    if not kopf or not zeilen:
        return ''

    css = """
    <style>
      .ew-wt {width:100%; border-collapse:separate; border-spacing:3px;
              font-size:0.82rem; margin-top:.2rem;}
      .ew-wt th {font-weight:600; padding:6px 4px; text-align:center;
                 font-size:0.8rem; opacity:.85; line-height:1.25;}
      .ew-wt td.lbl {text-align:left; font-weight:600; padding:6px 8px;
                     white-space:nowrap; opacity:.8; font-size:0.78rem;}
      .ew-wt td.cell {text-align:center; padding:7px 3px; border-radius:7px;
                      line-height:1.3;}
      .ew-sym {font-size:1.25rem; display:block; line-height:1.5;}
      .ew-t {font-weight:600;}
      .ew-r {font-size:0.7rem; opacity:.75;}
      @media (max-width: 640px) {
        .ew-wt {font-size:0.72rem; border-spacing:2px;}
        .ew-sym {font-size:1.05rem;}
        .ew-wt td.lbl {padding:4px 4px; font-size:0.68rem;}
        .ew-wt td.cell {padding:5px 2px;}
        .ew-r {font-size:0.62rem;}
      }
    </style>
    """

    html = [css, '<table class="ew-wt"><thead><tr><th></th>']
    for k in kopf:
        html.append(f'<th>{k}</th>')
    html.append('</tr></thead><tbody>')

    for label, zellen in zeilen:
        html.append(f'<tr><td class="lbl">{label}</td>')
        for z in zellen:
            if z is None:
                html.append('<td class="cell" style="opacity:.25">–</td>')
                continue
            if z['stufe'] is not None:
                farbe = STUFEN[z['stufe']]['farbe']
                hg = f'background: {farbe}1f; box-shadow: inset 0 0 0 1px {farbe}55;'
            else:
                hg = ''
            regen = ''
            if z['ns_mm'] >= 0.1:
                regen = f"<span class='ew-r'>{z['ns_mm']:.1f} mm</span>"
            elif z['ns_wahr'] >= 25:
                regen = f"<span class='ew-r'>{z['ns_wahr']:.0f} %</span>"
            html.append(
                f'<td class="cell" style="{hg}" title="{z["beschreibung"]}">'
                f'<span class="ew-sym">{z["symbol"]}</span>'
                f'<span class="ew-t">{z["temp"]:.0f}°</span>'
                f'{"<br>" + regen if regen else ""}</td>')
        html.append('</tr>')
    html.append('</tbody></table>')
    return ''.join(html)


def baue_heatmap(konsens_df, sonne=None):
    df = konsens_df.copy()
    df['datum'] = df['time'].dt.date
    df['stunde'] = df['time'].dt.hour

    tage_list = sorted(df['datum'].unique())
    stunden = list(range(24))

    z = np.full((len(tage_list), 24), np.nan)
    custom = [[None]*24 for _ in range(len(tage_list))]
    unsicher_x, unsicher_y = [], []
    lookup = {}

    y_labels = [f'{tag_kurz(pd.Timestamp(t))} {pd.Timestamp(t).strftime("%d.%m.")}'
                for t in tage_list]

    for i, tag in enumerate(tage_list):
        for j in stunden:
            zelle = df[(df['datum'] == tag) & (df['stunde'] == j)]
            if zelle.empty: continue
            r = zelle.iloc[0]
            z[i][j] = r['ampel_int']
            lookup[(y_labels[i], j)] = r.to_dict()

            def zahl(feld, nk=0):
                """Formatiert einen Wert robust; fehlende Werte werden zu –."""
                v = r.get(feld) if hasattr(r, 'get') else None
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return '–'
                try:
                    return f"{float(v):.{nk}f}"
                except (TypeError, ValueError):
                    return '–'

            custom[i][j] = [
                r.get('ampel', '–'), r.get('sicherheit', '–'),
                zahl('temp_median'), zahl('temp_spread'),
                zahl('wind_median'), zahl('feuchte_median'),
                zahl('ns_median', 1), zahl('ns_wahrscheinlichkeit'),
                int(r.get('n_laeufe') or 0), r.get('gruende', ''),
                zahl('taupunkt'), zahl('boeen'),
                (himmelsrichtung(r.get('windrichtung'))
                 if r.get('windrichtung') is not None else '–'),
            ]
            if r.get('sicherheit') == 'gering':
                unsicher_x.append(j); unsicher_y.append(y_labels[i])

    # Diskrete Farbskala über fünf Stufen (0–4)
    grenzen = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    colorscale = []
    for stufe, (u, o) in enumerate(grenzen):
        farbe = STUFEN[stufe]['farbe']
        colorscale.append([u, farbe])
        colorscale.append([o, farbe])

    fig = go.Figure(data=go.Heatmap(
        z=z, x=stunden, y=y_labels, customdata=custom,
        colorscale=colorscale, zmin=0, zmax=4, showscale=False,
        xgap=2, ygap=2, hoverongaps=False,
        hovertemplate=(
            '<b>%{y}  %{x}:00</b><br>'
            'Eignung: %{customdata[0]} · %{customdata[1]}<br>'
            'Temperatur: %{customdata[2]} °C (±%{customdata[3]})<br>'
            'Taupunkt: %{customdata[10]} °C · Feuchte: %{customdata[5]} %<br>'
            'Wind: %{customdata[4]} km/h · Böen: %{customdata[11]} km/h '
            '(%{customdata[12]})<br>'
            'Niederschlag: %{customdata[6]} mm/h (%{customdata[7]} %)<br>'
            'Läufe: %{customdata[8]}<br>'
            '<i>%{customdata[9]}</i><extra></extra>')))

    if unsicher_x:
        fig.add_trace(go.Scatter(
            x=unsicher_x, y=unsicher_y, mode='markers',
            marker=dict(symbol='x-thin', size=7,
                        line=dict(color='rgba(255,255,255,0.7)', width=1.2)),
            hoverinfo='skip', showlegend=False))

    # Sonnenauf- und -untergang je Tag markieren
    if sonne is not None and not sonne.empty:
        for i, tag in enumerate(tage_list):
            zeile = sonne[sonne['datum'] == tag]
            if zeile.empty:
                continue
            auf = zeile.iloc[0]['aufgang']
            unter = zeile.iloc[0]['untergang']
            for zeitpunkt, farbe in ((auf, '#d9a441'), (unter, '#7b6ca8')):
                if pd.isna(zeitpunkt):
                    continue
                xpos = zeitpunkt.hour + zeitpunkt.minute / 60.0 - 0.5
                fig.add_shape(type='line', x0=xpos, x1=xpos,
                              y0=i - 0.5, y1=i + 0.5,
                              line=dict(color=farbe, width=2))

    fig.update_layout(
        height=max(280, len(tage_list)*46 + 120),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title='Uhrzeit', tickmode='array',
                   tickvals=list(range(0, 24, 2)), side='top', fixedrange=True),
        yaxis=dict(autorange='reversed', fixedrange=True),
        plot_bgcolor='rgba(0,0,0,0)')
    return fig, lookup


def _zeitlabel(t):
    return f"{tag_kurz(t)} {t.strftime('%d.%m.')} {t.hour:02d}:00"


def _basis_layout(fig, hoehe, legende=True):
    # Achsen fixieren: verhindert ungewolltes Zoomen per Fingertipp am Handy
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        height=hoehe, margin=dict(l=10, r=60, t=10, b=10),
        legend=(dict(orientation='h', y=-0.22, x=0) if legende else None),
        showlegend=legende, hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.15)', zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def zeichne_temperatur(konsens_df):
    """Temperatur mit Ensemble-Streubreite und Taupunkt."""
    df = konsens_df.sort_values('time').reset_index(drop=True)
    if df.empty:
        return None
    zeiten = df['time']
    temps = pd.to_numeric(df['temp_median'], errors='coerce').values
    spread = pd.to_numeric(df['temp_spread'], errors='coerce').fillna(0).values
    p10, p90 = temps - spread / 2, temps + spread / 2
    taup = (pd.to_numeric(df['taupunkt'], errors='coerce').values
            if 'taupunkt' in df.columns else np.full(len(df), np.nan))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(zeiten) + list(zeiten[::-1]), y=list(p90) + list(p10[::-1]),
        fill='toself', fillcolor='rgba(192,57,43,0.13)',
        line=dict(color='rgba(0,0,0,0)'), hoverinfo='skip',
        name='Streubereich P10–P90'))
    fig.add_trace(go.Scatter(
        x=zeiten, y=temps, mode='lines', name='Temperatur (Median)',
        line=dict(color='#c0392b', width=2.2),
        text=[f"<b>{_zeitlabel(t)}</b><br>Median: {m:.1f} °C<br>"
              f"Streubereich: {lo:.1f} – {hi:.1f} °C"
              for t, m, lo, hi in zip(zeiten, temps, p10, p90)],
        hovertemplate='%{text}<extra></extra>'))
    if not np.all(np.isnan(taup)):
        fig.add_trace(go.Scatter(
            x=zeiten, y=taup, mode='lines', name='Taupunkt',
            line=dict(color='#2e86c1', width=1.6, dash='dash'),
            text=[f"<b>{_zeitlabel(t)}</b><br>Taupunkt: {v:.1f} °C"
                  for t, v in zip(zeiten, taup)],
            hovertemplate='%{text}<extra></extra>'))
    fig.update_yaxes(title_text='°C')
    return _basis_layout(fig, 300)


def zeichne_niederschlag(konsens_df):
    """Stundensumme und Eintrittswahrscheinlichkeit."""
    df = konsens_df.sort_values('time').reset_index(drop=True)
    if df.empty:
        return None
    zeiten = df['time']
    precip = pd.to_numeric(df['ns_median'], errors='coerce').fillna(0).values
    wahr = pd.to_numeric(df['ns_wahrscheinlichkeit'], errors='coerce').fillna(0).values

    hov = []
    for t, p, w in zip(zeiten, precip, wahr):
        txt = f"<b>{_zeitlabel(t)}</b><br>Wahrscheinlichkeit: {w:.0f} %"
        if p > 0.05:
            txt += f"<br>Stundensumme: {p:.1f} mm"
        hov.append(txt)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=zeiten, y=precip, name='Stundensumme (mm)',
        marker_color='#2e86c1', opacity=0.8,
        text=hov, hovertemplate='%{text}<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=zeiten, y=wahr, mode='lines', name='Eintrittswahrscheinlichkeit (%)',
        yaxis='y2', line=dict(color='#1a5276', width=1.5, dash='dot'),
        text=hov, hovertemplate='%{text}<extra></extra>'))
    fig.update_yaxes(title_text='mm/h')
    fig.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 105],
                                  showgrid=False,
                                  title=dict(text='%', font=dict(size=11))))
    return _basis_layout(fig, 280)


def zeichne_laeufe(konsens_df):
    """Anzahl der einfließenden Modell-Läufe je Stunde."""
    df = konsens_df.sort_values('time').reset_index(drop=True)
    if df.empty:
        return None
    zeiten = df['time']
    laeufe = pd.to_numeric(df['n_laeufe'], errors='coerce').fillna(0).values
    ens = (pd.to_numeric(df['n_ens_mitglieder'], errors='coerce').fillna(0).values
           if 'n_ens_mitglieder' in df.columns else laeufe)
    haupt = (pd.to_numeric(df['n_haupt'], errors='coerce').fillna(0).values
             if 'n_haupt' in df.columns else np.zeros(len(df)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=zeiten, y=laeufe, mode='lines', name='Läufe gesamt',
        fill='tozeroy', fillcolor='rgba(108,52,131,0.22)',
        line=dict(color='#6c3483', width=1.6),
        text=[f"<b>{_zeitlabel(t)}</b><br>Läufe gesamt: {int(l)}<br>"
              f"davon Ensemble: {int(e)} · Hauptläufe: {int(h)}"
              for t, l, e, h in zip(zeiten, laeufe, ens, haupt)],
        hovertemplate='%{text}<extra></extra>'))
    fig.update_yaxes(title_text='Anzahl', rangemode='tozero')
    return _basis_layout(fig, 220, legende=False)


def zeichne_wind(konsens_df):
    """Windgeschwindigkeit, Böen und Windrichtung."""
    from plotly.subplots import make_subplots

    df = konsens_df.sort_values('time').reset_index(drop=True)
    if df.empty or 'wind_median' not in df.columns:
        return None

    zeiten = df['time']
    wind = pd.to_numeric(df['wind_median'], errors='coerce').values
    boeen = (pd.to_numeric(df['boeen'], errors='coerce').values
             if 'boeen' in df.columns else np.full(len(df), np.nan))
    richtung = (pd.to_numeric(df['windrichtung'], errors='coerce').values
                if 'windrichtung' in df.columns else np.full(len(df), np.nan))

    def zeitlabel(t):
        return f"{tag_kurz(t)} {t.strftime('%d.%m.')} {t.hour:02d}:00"

    hover_wind = [f"<b>{zeitlabel(t)}</b><br>Wind: {w:.0f} km/h"
                  + (f"<br>Böen: {b:.0f} km/h" if not np.isnan(b) else "")
                  + (f"<br>Richtung: {himmelsrichtung(r)} ({r:.0f}°)"
                     if not np.isnan(r) else "")
                  for t, w, b, r in zip(zeiten, wind, boeen, richtung)]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=('', 'Anströmrichtung'))

    if not np.all(np.isnan(boeen)):
        fig.add_trace(go.Scatter(
            x=zeiten, y=boeen, mode='lines', name='Böen',
            line=dict(color='#8e5a9e', width=1.4, dash='dot'),
            text=hover_wind, hovertemplate='%{text}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=zeiten, y=wind, mode='lines', name='mittlere Windgeschwindigkeit',
        line=dict(color='#2e7d8f', width=2), fill='tozeroy',
        fillcolor='rgba(46,125,143,0.16)',
        text=hover_wind, hovertemplate='%{text}<extra></extra>'), row=1, col=1)

    # Windrichtung als Pfeile alle 3 Stunden
    schritt = max(1, len(df) // 60)
    idx = list(range(0, len(df), schritt))
    gueltig = [i for i in idx if not np.isnan(richtung[i])]
    if gueltig:
        fig.add_trace(go.Scatter(
            x=[zeiten[i] for i in gueltig],
            y=[0] * len(gueltig), mode='markers',
            marker=dict(symbol='arrow', size=13, color='#2e7d8f',
                        angle=[float(richtung[i]) + 180 for i in gueltig],
                        line=dict(width=0)),
            name='Windrichtung',
            text=[hover_wind[i] for i in gueltig],
            hovertemplate='%{text}<extra></extra>'), row=2, col=1)

    fig.update_yaxes(title_text='km/h', row=1, col=1,
                     gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(visible=False, range=[-1, 1], row=2, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=380, margin=dict(l=10, r=20, t=42, b=10),
                      legend=dict(orientation='h', y=-0.14, x=0),
                      hovermode='closest',
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig


def zeichne_bewoelkung(konsens_df):
    """Bewölkung nach Schichten als Heatmap plus Gesamtbedeckung."""
    from plotly.subplots import make_subplots

    df = konsens_df.sort_values('time').reset_index(drop=True)
    schichten = [('wolken_tief', 'tief (< 2 km)'),
                 ('wolken_mittel', 'mittel (2–6 km)'),
                 ('wolken_hoch', 'hoch (> 6 km)')]
    vorhanden = [(k, lbl) for k, lbl in schichten if k in df.columns
                 and df[k].notna().any()]
    if not vorhanden:
        return None

    zeiten = df['time']
    z, custom = [], []
    for key, lbl in vorhanden:
        werte = pd.to_numeric(df[key], errors='coerce').values
        z.append(werte)
        custom.append([
            f"<b>Bewölkung {lbl}</b><br>{tag_kurz(t)} {t.strftime('%d.%m.')} "
            f"{t.hour:02d}:00<br>Bedeckung: {v:.0f} %"
            if not np.isnan(v) else '' for t, v in zip(zeiten, werte)])

    colorscale = [[0.0, '#eaf2f8'], [0.35, '#c3ced6'],
                  [0.7, '#8d979f'], [1.0, '#4c545c']]

    hat_gesamt = 'wolken' in df.columns and df['wolken'].notna().any()
    fig = make_subplots(
        rows=2 if hat_gesamt else 1, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38] if hat_gesamt else [1.0],
        vertical_spacing=0.1,
        subplot_titles=(['', 'Gesamtbedeckung (%)'] if hat_gesamt
                        else ['']))

    fig.add_trace(go.Heatmap(
        z=z, x=zeiten, y=[lbl for _, lbl in vorhanden], customdata=custom,
        colorscale=colorscale, zmin=0, zmax=100,
        colorbar=dict(title=dict(text='%', font=dict(size=10)),
                      thickness=12, len=0.5, y=0.78),
        hovertemplate='%{customdata}<extra></extra>'), row=1, col=1)

    if hat_gesamt:
        gesamt = pd.to_numeric(df['wolken'], errors='coerce').values
        hov = [f"<b>{tag_kurz(t)} {t.strftime('%d.%m.')} {t.hour:02d}:00</b>"
               f"<br>Gesamtbedeckung: {v:.0f} %"
               for t, v in zip(zeiten, gesamt)]
        fig.add_trace(go.Scatter(
            x=zeiten, y=gesamt, mode='lines', name='Gesamtbedeckung',
            line=dict(color='#5b6770', width=1.8), fill='tozeroy',
            fillcolor='rgba(91,103,112,0.18)',
            text=hov, hovertemplate='%{text}<extra></extra>'), row=2, col=1)
        fig.update_yaxes(range=[0, 105], title_text='%', row=2, col=1,
                         gridcolor='rgba(128,128,128,0.15)')

    fig.update_xaxes(showgrid=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=380 if hat_gesamt else 240,
                      margin=dict(l=10, r=10, t=42, b=10),
                      showlegend=False,
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig


def zeichne_bodenprofil(boden, infil_df):
    """
    Bodenfeuchte-Profil als Heatmap:
      Y = Bodenschichten (oben → unten), X = Zeit (Vergangenheit + Prognose)
      Farbe = Feuchte (trocken → gesättigt)
    Darunter: Niederschlag aufgeteilt in eingesickert vs. oberflächlich abgelaufen.
    """
    from plotly.subplots import make_subplots

    if boden is None or boden.get('hourly') is None:
        return None
    dfh = boden['hourly']
    vorhandene = [(lbl, beschr) for _, lbl, beschr in SOIL_LAYERS
                  if lbl in dfh.columns]
    if not vorhandene:
        return None

    zeiten = dfh['time']
    y_labels = [f'{lbl}' for lbl, _ in vorhandene]
    z = [dfh[lbl].values.astype(float) for lbl, _ in vorhandene]

    # Hover je Zelle
    custom = []
    for (lbl, beschr) in vorhandene:
        werte = dfh[lbl].values.astype(float)
        zeile = []
        for t, v in zip(zeiten, werte):
            if np.isnan(v):
                zeile.append('')
                continue
            if v < 0.12: zustand = 'sehr trocken'
            elif v < 0.20: zustand = 'trocken'
            elif v < 0.30: zustand = 'normal'
            elif v < 0.38: zustand = 'feucht'
            else: zustand = 'nass / gesättigt'
            zeile.append(
                f"<b>{lbl} — {beschr}</b><br>"
                f"{tag_kurz(t)} {t.strftime('%d.%m.')} {t.hour:02d}:00<br>"
                f"Feuchte: {v:.2f} m³/m³ ({zustand})")
        custom.append(zeile)

    # Farbskala: sandbraun (trocken) → grün → blau (nass)
    colorscale = [
        [0.00, '#c9a227'],   # staubtrocken
        [0.25, '#d9c68a'],
        [0.45, '#a8c090'],   # normal
        [0.65, '#5aa9c9'],
        [1.00, '#1a4f7a'],   # gesättigt
    ]

    hat_infil = infil_df is not None and 'ns_effektiv' in infil_df.columns
    rows = 2 if hat_infil else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4] if hat_infil else [1.0],
        vertical_spacing=0.09,
        subplot_titles=(['', 'Infiltration und Oberflächenabfluss (mm/h)']
                        if hat_infil else ['']))

    fig.add_trace(go.Heatmap(
        z=z, x=zeiten, y=y_labels, customdata=custom,
        colorscale=colorscale, zmin=0.05, zmax=0.45,
        colorbar=dict(title=dict(text='m³/m³', font=dict(size=10)),
                      thickness=12, len=0.55, y=0.75),
        hovertemplate='%{customdata}<extra></extra>'), row=1, col=1)

    if hat_infil:
        fig.add_trace(go.Bar(
            x=infil_df['time'], y=infil_df['ns_effektiv'],
            name='eingesickert', marker_color='#2e86c1',
            hovertemplate='eingesickert: %{y:.1f} mm<extra></extra>'),
            row=2, col=1)
        fig.add_trace(go.Bar(
            x=infil_df['time'], y=infil_df['ns_abfluss'],
            name='abgelaufen', marker_color='#e67e22',
            hovertemplate='oberflächlich abgelaufen: %{y:.1f} mm<extra></extra>'),
            row=2, col=1)
        fig.update_layout(barmode='stack')
        fig.update_yaxes(title=dict(text='mm/h', font=dict(size=10)), row=2, col=1)

    # "Jetzt"-Linie
    jetzt = datetime.now()
    fig.add_vline(x=jetzt, line=dict(color='rgba(200,60,60,0.8)', width=2, dash='dash'))
    fig.add_annotation(x=jetzt, y=1.06, yref='paper', text='jetzt',
                       showarrow=False, font=dict(size=10, color='#c0392b'))

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(
        height=430 if hat_infil else 260,
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation='h', y=-0.12, x=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(autorange='reversed', row=1, col=1)
    fig.update_xaxes(showgrid=False)
    return fig


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="EnsembleWetter", page_icon="◭",
                   layout="wide", initial_sidebar_state="collapsed")

# Plotly-Konfiguration: Zoom per Fingertipp aus, Werkzeugleiste nur auf Wunsch
PLOT_CONFIG = {
    'scrollZoom': False,
    'doubleClick': False,
    'displaylogo': False,
    'displayModeBar': 'hover',
    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d'],
    'responsive': True,
}
PLOT_CONFIG_HEATMAP = dict(PLOT_CONFIG)
PLOT_CONFIG_HEATMAP['displayModeBar'] = False

st.markdown("""
<style>
  /* Streamlit-eigene Bedienelemente ausblenden */
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
  header {visibility: hidden;}
  .stDeployButton {display: none !important;}
  [data-testid="stToolbar"] {display: none !important;}
  [data-testid="stDecoration"] {display: none !important;}
  [data-testid="stStatusWidget"] {display: none !important;}
  [data-testid="collapsedControl"] {display: none !important;}
  section[data-testid="stSidebar"] {display: none !important;}
  a[href*="github.com"] {display: none !important;}
  .viewerBadge_container__1QSob, .viewerBadge_link__1S137 {display: none !important;}

  .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem;
                    max-width: 1240px;}
  h1 {font-weight: 600; letter-spacing: -0.6px; margin-bottom: 0.1rem;
      font-size: 2.0rem;}
  h2, h3 {font-weight: 600; letter-spacing: -0.2px;}
  div[data-testid="stMetricValue"] {font-size: 1.25rem; font-weight: 600;}
  div[data-testid="stMetricLabel"] {font-size: 0.78rem; opacity: 0.72;}

  .ew-section {font-size: 1.06rem; font-weight: 600; letter-spacing: -0.2px;
               margin: 1.7rem 0 0.15rem 0;}
  .ew-sub {font-size: 0.84rem; opacity: 0.68; margin-bottom: 0.5rem;
           line-height: 1.45;}
  .ew-note {font-size: 0.79rem; opacity: 0.62; line-height: 1.5;
            margin-top: 0.35rem;}
  .ew-legend {display: flex; flex-wrap: wrap; gap: 0.45rem 1.0rem;
              align-items: center; font-size: 0.78rem; opacity: 0.85;
              margin: 0.45rem 0 0.2rem 0;}
  .ew-chip {display: inline-flex; align-items: center; gap: 0.35rem;}
  .ew-dot {width: 12px; height: 12px; border-radius: 3px;
           display: inline-block;}
  .ew-headline {font-size: 1.0rem; font-weight: 600; margin-bottom: 0.15rem;}

  /* Eingabebereich */
  .ew-formcard {border-radius: 12px; padding: 0.2rem 0 0.4rem 0;
                margin-bottom: 0.3rem;}

  /* Mobile Anpassungen */
  @media (max-width: 640px) {
    .block-container {padding-left: 0.9rem; padding-right: 0.9rem;
                      padding-top: 1.0rem;}
    h1 {font-size: 1.55rem;}
    .ew-section {font-size: 0.98rem; margin-top: 1.4rem;}
    .ew-sub {font-size: 0.78rem;}
    div[data-testid="stMetricValue"] {font-size: 1.05rem;}
    div[data-testid="stMetricLabel"] {font-size: 0.7rem;}
    .ew-legend {font-size: 0.72rem; gap: 0.35rem 0.7rem;}
  }
</style>
""", unsafe_allow_html=True)


def abschnitt(titel, untertitel=None):
    st.markdown(f'<div class="ew-section">{titel}</div>', unsafe_allow_html=True)
    if untertitel:
        st.markdown(f'<div class="ew-sub">{untertitel}</div>',
                    unsafe_allow_html=True)


def notiz(text):
    st.markdown(f'<div class="ew-note">{text}</div>', unsafe_allow_html=True)


def stufen_legende():
    chips = ''.join(
        f'<span class="ew-chip"><span class="ew-dot" '
        f'style="background:{STUFEN[s]["farbe"]}"></span>{STUFEN[s]["name"]}</span>'
        for s in [4, 3, 2, 1, 0])
    chips += ('<span class="ew-chip"><span style="font-weight:700">×</span>'
              'geringe Vorhersagesicherheit</span>')
    st.markdown(f'<div class="ew-legend">{chips}</div>', unsafe_allow_html=True)


PARAM_FELDER = [
    ('temp_min', 'Temperatur min', '°C', -40.0, 50.0, 0.5),
    ('temp_max', 'Temperatur max', '°C', -40.0, 50.0, 0.5),
    ('humidity_min', 'Luftfeuchte min', '%', 0.0, 100.0, 1.0),
    ('humidity_max', 'Luftfeuchte max', '%', 0.0, 100.0, 1.0),
    ('wind_max', 'Wind max', 'km/h', 0.0, 150.0, 1.0),
    ('precip_max', 'Niederschlag max', 'mm/h', 0.0, 10.0, 0.1),
    ('trockenfenster_h', 'Trockenfenster', 'h', 0.0, 72.0, 1.0),
]
FROST_FELDER = [
    ('temp_warn', 'Warnschwelle', '°C', -20.0, 10.0, 0.5),
    ('temp_frost', 'Frostschwelle', '°C', -20.0, 10.0, 0.5),
]


def parameter_editor(fall_name, basis):
    af = {'name': fall_name,
          'beschreibung': basis.get('beschreibung', ''),
          'modus': basis.get('modus', 'standard'),
          'boden_relevant': basis.get('boden_relevant', False),
          'duerre_relevant': basis.get('duerre_relevant', False)}
    felder = FROST_FELDER if af['modus'] == 'frost' else PARAM_FELDER
    with st.expander("Richtwerte anpassen (optional)", expanded=False):
        st.caption("Fachliche Vorgabewerte. Sie lassen sich an Standort, "
                   "Bodenart und eigene Erfahrung anpassen.")
        spalten = st.columns(2)
        i = 0
        for key, label, einheit, mn, mx, step in felder:
            if key in basis or (key == 'trockenfenster_h'
                                and af['modus'] != 'frost'):
                with spalten[i % 2]:
                    af[key] = st.number_input(
                        f"{label} ({einheit})", min_value=mn, max_value=mx,
                        value=float(basis.get(key, 0.0)), step=step,
                        key=f"{fall_name}__{key}")
                i += 1
        if af['modus'] != 'frost':
            af['boden_relevant'] = st.checkbox(
                "Bodenzustand einbeziehen", value=af['boden_relevant'],
                key=f"{fall_name}__boden")
    if 'trockenfenster_h' in af:
        af['trockenfenster_h'] = int(af['trockenfenster_h'])
    return af


# ============================================================
# KOPFBEREICH UND EINGABE
# ============================================================

st.title("EnsembleWetter")
st.caption("Präzisionsprognose für Landwirtschaft, Bau und Alpinraum — "
           "Multi-Modell-Ensemble mit Boden- und Höhenanalyse")

st.markdown('<div class="ew-formcard">', unsafe_allow_html=True)

e1, e2 = st.columns([1, 1])
with e1:
    ort_eingabe = st.text_input("Standort", value="Wieselburg",
                                placeholder="Ort eingeben, z. B. Leoben")
with e2:
    kategorie = st.selectbox("Bereich", list(ANWENDUNGSFAELLE.keys()), index=0)

e3, e4 = st.columns([1, 1])
with e3:
    faelle = list(ANWENDUNGSFAELLE[kategorie].keys())
    fall_name = st.selectbox("Tätigkeit", faelle, index=0)
with e4:
    tage = st.slider("Prognosezeitraum (Tage)", 1, 16, 5)

basis = ANWENDUNGSFAELLE[kategorie][fall_name]
st.caption(basis.get('beschreibung', ''))

af_aktiv = parameter_editor(fall_name, basis)

gewaehlter_ort = None
if ort_eingabe:
    treffer = geocode_ort(ort_eingabe)
    if not treffer:
        st.error(f"Standort „{ort_eingabe}“ wurde nicht gefunden. "
                 f"Bitte Schreibweise prüfen.")
    elif len(treffer) == 1:
        gewaehlter_ort = treffer[0]
    else:
        labels = [t['label'] for t in treffer]
        wahl = st.selectbox("Mehrere Orte gefunden — bitte auswählen", labels)
        gewaehlter_ort = treffer[labels.index(wahl)]

los = st.button("Analyse starten", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if los and gewaehlter_ort:
    bar = st.progress(0.0, text="Wetterdaten werden geladen …")
    alle_daten = lade_alle_daten(gewaehlter_ort['lat'], gewaehlter_ort['lon'],
                                 af_aktiv, tage, fortschritt=bar,
                                 standorthoehe=gewaehlter_ort.get('hoehe'))
    bar.progress(1.0, text="Auswertung wird berechnet …")
    konsens = berechne_konsens(alle_daten, af_aktiv, tage)
    konsens = wende_trockenfenster_an(konsens, af_aktiv)
    bar.empty()
    st.session_state['konsens'] = konsens
    st.session_state['alle_daten'] = alle_daten
    st.session_state['kontext'] = {
        'ort': gewaehlter_ort['label'], 'anwendung': fall_name,
        'tage': tage, 'af': af_aktiv, 'kategorie': kategorie,
        'lat': gewaehlter_ort['lat'], 'lon': gewaehlter_ort['lon'],
        'hoehe': gewaehlter_ort.get('hoehe')}
elif los and not gewaehlter_ort:
    st.warning("Bitte zuerst einen gültigen Standort eingeben.")

# ============================================================
# ERGEBNISDARSTELLUNG
# ============================================================

if 'konsens' in st.session_state:
    konsens = st.session_state['konsens']
    alle_daten = st.session_state['alle_daten']
    kontext = st.session_state['kontext']
    af_kontext = kontext.get('af', {})
    bodenindex = alle_daten.get('bodenindex')
    sonne = alle_daten.get('sonne')
    hoehe = kontext.get('hoehe')
    delta_h = alle_daten.get('delta_h')

    st.divider()
    st.subheader(f"{kontext['ort']} · {kontext['anwendung']} · "
                 f"{kontext['tage']} Tage")

    if hoehe is not None and delta_h is not None and abs(delta_h) >= 100:
        dt = delta_h / 100.0 * TEMP_GRADIENT_PRO_100M
        richtung = "kälter" if delta_h > 0 else "milder"
        st.info(f"Höhenkorrektur aktiv. Standort {hoehe:.0f} m, Modellgitter "
                f"{alle_daten.get('modellhoehe') or 0:.0f} m ({delta_h:+.0f} m). "
                f"Temperatur und Taupunkt um {abs(dt):.1f} °C {richtung} "
                f"gerechnet, Wind entsprechend angepasst.")

    ziel, best, n_bloecke = finde_bestes_fenster(konsens)
    if best is not None:
        s = best['start'].strftime(f'{tag_kurz(best["start"])} %d.%m. %H:%M')
        e = (best['ende'] + timedelta(hours=1)).strftime('%H:%M')
        extra = f" · {n_bloecke - 1} weitere Fenster" if n_bloecke > 1 else ""
        text = (f"**Empfohlenes Zeitfenster:** {s} – {e} "
                f"({best['stunden']} h) · Eignung {ziel} · "
                f"Vorhersagesicherheit {best['sicherheit']}{extra}")
        if ziel == 'sehr gut':
            st.success(text)
        elif ziel == 'gut':
            st.info(text)
        else:
            st.warning(text + " — kein durchgehend gutes Fenster verfügbar.")
    else:
        st.error("Im gewählten Prognosezeitraum ergibt sich kein Zeitfenster, "
                 "das die gesetzten Anforderungen hinreichend erfüllt.")

    # --- Wetterüberblick mit Symbolen ---
    kopf, zeilen = baue_wetteruebersicht(konsens, max_tage=min(kontext['tage'], 7))
    if kopf:
        abschnitt("Wetterüberblick",
                  "Allgemeine Wetterlage nach Tagesabschnitten. "
                  "Die Hinterlegung zeigt die Eignung für die gewählte Tätigkeit.")
        st.markdown(wetteruebersicht_html(kopf, zeilen), unsafe_allow_html=True)

    zeige_boden = bool(af_kontext.get('boden_relevant')) and bodenindex is not None
    if zeige_boden:
        m = bodenindex['metriken']
        abschnitt("Bodenzustand",
                  "Niederschlagsbilanz der vergangenen zehn Tage")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Niederschlag 3 Tage", f"{m['regen_3t']:.0f} mm")
        c2.metric("Niederschlag 7 Tage", f"{m['regen_7t']:.0f} mm")
        c3.metric("Letzter Niederschlag",
                  "heute oder gestern" if m['tage_seit_regen'] <= 1
                  else f"vor {m['tage_seit_regen']} Tagen")
        c4.metric("Wasserbilanz", f"{m['balance_heute']:.0f} mm")
        if m['zustand_level'] >= 1:
            notiz("Der Bodenzustand fließt in die Bewertung ein; betroffene "
                  "Zeitfenster sind entsprechend herabgestuft.")

    abschnitt("Eignungsmatrix",
              "Stündliche Bewertung im gewichteten Konsens aller Modelle. "
              "Ein Feld antippen, um die Detailwerte zu sehen.")
    stufen_legende()
    fig, lookup = baue_heatmap(konsens, sonne=sonne)
    event = st.plotly_chart(fig, use_container_width=True, key="hm",
                            on_select="rerun", config=PLOT_CONFIG_HEATMAP)

    gewaehlt = None
    try:
        pts = event.selection.points if event and event.selection else []
        if pts:
            p = pts[0]
            gewaehlt = lookup.get((p['y'], int(p['x'])))
    except Exception:
        gewaehlt = None

    if gewaehlt:
        t = gewaehlt['time']
        st.markdown(f'<div class="ew-headline">Detailwerte — {tag_kurz(t)} '
                    f'{t.strftime("%d.%m.")}, {t.hour:02d}:00 Uhr</div>',
                    unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Eignung", gewaehlt['ampel'])
        d1.caption(f"Vorhersagesicherheit {gewaehlt['sicherheit']}")
        if gewaehlt['temp_median'] is not None:
            d2.metric("Temperatur", f"{gewaehlt['temp_median']:.0f} °C",
                      f"± {gewaehlt['temp_spread']:.0f} °C Streubreite",
                      delta_color="off")
        if gewaehlt.get('taupunkt') is not None and not pd.isna(gewaehlt['taupunkt']):
            d2.metric("Taupunkt", f"{gewaehlt['taupunkt']:.0f} °C")
        if gewaehlt['wind_median'] is not None:
            boe = gewaehlt.get('boeen')
            d3.metric("Wind", f"{gewaehlt['wind_median']:.0f} km/h",
                      (f"Böen bis {boe:.0f} km/h"
                       if boe is not None and not pd.isna(boe) else None),
                      delta_color="off")
        d3.metric("Niederschlag", f"{gewaehlt['ns_median']:.1f} mm/h",
                  f"{gewaehlt['ns_wahrscheinlichkeit']:.0f} % Wahrscheinlichkeit",
                  delta_color="off")
        d4.metric("Modell-Läufe", int(gewaehlt['n_laeufe']))
        d4.caption(f"{gewaehlt['n_ens_mitglieder']} Ensemble-Mitglieder, "
                   f"{gewaehlt['n_haupt']} Hauptläufe")
        if gewaehlt['gruende']:
            notiz(f"Bewertungsgrundlage: {gewaehlt['gruende']}")
    else:
        notiz("Senkrechte Striche kennzeichnen Sonnenaufgang und "
              "Sonnenuntergang des jeweiligen Tages.")

    abschnitt("Temperatur und Taupunkt",
              "Ensemble-Median mit Streubereich zwischen 10. und 90. Perzentil")
    ft = zeichne_temperatur(konsens)
    if ft is not None:
        st.plotly_chart(ft, use_container_width=True, key="temp",
                        config=PLOT_CONFIG)
        notiz("Nähert sich die Temperatur dem Taupunkt, steigen Nebel-, Tau- "
              "und Reifwahrscheinlichkeit; die Abtrocknung kommt zum Erliegen.")

    abschnitt("Niederschlag",
              "Stundensumme im Median und Eintrittswahrscheinlichkeit "
              "über alle Ensemble-Mitglieder")
    fn = zeichne_niederschlag(konsens)
    if fn is not None:
        st.plotly_chart(fn, use_container_width=True, key="ns",
                        config=PLOT_CONFIG)

    fw = zeichne_wind(konsens)
    if fw is not None:
        abschnitt("Wind",
                  "Mittlere Geschwindigkeit, Böenspitzen und Anströmrichtung")
        st.plotly_chart(fw, use_container_width=True, key="wind",
                        config=PLOT_CONFIG)

    abschnitt("Datentiefe",
              "Anzahl der je Stunde einfließenden Modell-Läufe")
    fl = zeichne_laeufe(konsens)
    if fl is not None:
        st.plotly_chart(fl, use_container_width=True, key="laeufe",
                        config=PLOT_CONFIG)
        notiz("Mit zunehmendem Vorhersagehorizont fallen die hochauflösenden "
              "Kurzfristmodelle weg; die Datenbasis wird schmaler und die "
              "Aussage entsprechend unschärfer.")

    fc = zeichne_bewoelkung(konsens)
    if fc is not None:
        abschnitt("Bewölkung",
                  "Bedeckungsgrad nach Schichthöhe und in Summe")
        st.plotly_chart(fc, use_container_width=True, key="wolken",
                        config=PLOT_CONFIG)
        notiz("Tiefe Bewölkung bremst die Abtrocknung am Boden. Hohe "
              "Bewölkung wirkt vor allem auf die nächtliche Ausstrahlung "
              "und damit auf das Frostrisiko.")

    if zeige_boden:
        fig_boden = zeichne_bodenprofil(alle_daten.get('boden'),
                                        alle_daten.get('infiltration'))
        if fig_boden is not None:
            abschnitt("Bodenfeuchte und Wasseraufnahme",
                      "Modellierte Feuchte nach Tiefe sowie der Anteil des "
                      "Niederschlags, der tatsächlich infiltriert")
            st.plotly_chart(fig_boden, use_container_width=True,
                            key="bodenprofil", config=PLOT_CONFIG)
            infil = alle_daten.get('infiltration')
            if infil is not None:
                jetzt = datetime.now()
                verg = infil[infil['time'] <= jetzt]
                zuk = infil[infil['time'] > jetzt]
                if not verg.empty:
                    ges = verg['precip_h'].sum()
                    rein = verg['ns_effektiv'].sum()
                    ab = verg['ns_abfluss'].sum()
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Niederschlag zehn Tage", f"{ges:.0f} mm")
                    b2.metric("davon infiltriert", f"{rein:.0f} mm",
                              f"{rein / ges * 100:.0f} %" if ges > 0 else None,
                              delta_color="off")
                    b3.metric("Oberflächenabfluss", f"{ab:.0f} mm",
                              f"{ab / ges * 100:.0f} %" if ges > 0 else None,
                              delta_color="off")
                if not zuk.empty and zuk['ns_abfluss'].sum() > 3:
                    notiz(f"Prognostiziert laufen rund "
                          f"{zuk['ns_abfluss'].sum():.0f} mm oberflächlich ab "
                          f"und erreichen den Wurzelraum nicht.")
            notiz("Dunkle Färbung kennzeichnet hohe Bodenfeuchte. Bei hoher "
                  "Niederschlagsintensität übersteigt die Menge die "
                  "Aufnahmefähigkeit des Bodens. Modellwerte — die örtliche "
                  "Bodenart kann abweichen.")

    lat = kontext.get('lat')
    lon = kontext.get('lon')

    def windy_karte(overlay, hinweis):
        if lat is None or lon is None:
            return
        components.iframe(
            f"https://embed.windy.com/embed2.html?lat={lat}&lon={lon}"
            f"&detailLat={lat}&detailLon={lon}&width=650&height=450&zoom=8"
            f"&level=surface&overlay={overlay}&menu=&message=&marker=true"
            f"&calendar=&pressure=&type=map&location=coordinates&detail="
            f"&metricWind=km%2Fh&metricTemp=%C2%B0C&radarRange=-1",
            height=460)
        st.caption(hinweis)

    abschnitt("Kartenansichten",
              "Aktuelle Felddaten als Ergänzung zur Punktprognose")
    with st.expander("Niederschlagsradar"):
        windy_karte("radar", "Radarbild der vergangenen und kommenden Stunden "
                             "(Quelle: Windy.com).")
    with st.expander("Windkarte"):
        windy_karte("wind", "Windfeld und Anströmrichtung im Umfeld des "
                            "Standorts (Quelle: Windy.com).")
    with st.expander("Temperaturkarte"):
        windy_karte("temp", "Flächenhafte Temperaturverteilung "
                            "(Quelle: Windy.com).")

    with st.expander("Datengrundlage und verwendete Modelle"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Deterministische Hauptläufe**")
            for name in alle_daten['haupt']:
                st.caption(f"{name} — Gitterweite "
                           f"{HAUPTLAUFE[name]['aufloesung']}")
        with cols[1]:
            st.markdown("**Ensemble-Systeme**")
            for name in alle_daten['ensemble']:
                cfg = ENSEMBLE_MODELLE[name]
                st.caption(f"{name} — {cfg['mitglieder']} Mitglieder, "
                           f"Gitterweite {cfg['aufloesung']}")
        st.markdown("**Bewertungsmaßstab**")
        st.caption("Für jeden Parameter wird die Überschreitung des Grenzwerts "
                   "auf einen Prozentwert normiert. Aus Anzahl und Ausmaß der "
                   "Überschreitungen ergibt sich die Eignungsstufe: "
                   "sehr gut ohne Überschreitung, gut bei einer geringen, "
                   "bedingt ab 15 Prozent oder zwei Überschreitungen, "
                   "ungünstig ab 35 Prozent, ungeeignet ab 50 Prozent "
                   "beziehungsweise zwei deutlichen Überschreitungen.")
        if hoehe is not None:
            st.caption(f"Standorthöhe {hoehe:.0f} m über Adria, "
                       f"Modellgitterhöhe "
                       f"{alle_daten.get('modellhoehe') or 0:.0f} m, "
                       f"Temperaturgradient {TEMP_GRADIENT_PRO_100M} °C je "
                       f"100 Höhenmeter.")

    st.divider()
    notiz("EnsembleWetter befindet sich in aktiver Entwicklung. "
          "Die Angaben ersetzen keine amtliche Wetterwarnung. Im Alpinraum "
          "ersetzt die App weder Lawinenlagebericht noch Tourenplanung.")

else:
    st.info("Standort und Tätigkeit oben auswählen, anschließend auf "
            "„Analyse starten“ tippen.")
