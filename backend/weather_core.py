"""
EnsembleWetter Austria — Kern-Wetterlogik
==========================================
Dieses Modul ist bewusst ohne Streamlit-Abhängigkeiten geschrieben.
Es kann direkt in FastAPI, Tests oder Jupyter verwendet werden.
"""
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

APP_VERSION = "v23"

import json
import re
from urllib.parse import quote
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go



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
    # --- Hochauflösende Regionalmodelle für den Alpenraum ---
    # Sie decken nicht das gesamte Bundesgebiet ab. Liefert ein Modell
    # für den gewählten Punkt keine Daten, wird es automatisch
    # übersprungen; die übrigen Modelle tragen die Auswertung.
    'GeoSphere AROME 2.5km': {
        'url': 'https://api.open-meteo.com/v1/forecast',
        'url_param': 'arome_seamless',
        'url_param_alt': ['geosphere_seamless', 'arome_austria'],
        'max_tage': 3, 'aufloesung': '2,5 km',
        'gebiet': 'Österreich und weiterer Alpenraum',
        'quelle': 'GeoSphere Austria'},
    'MeteoSwiss ICON-CH1 1km': {
        'url': 'https://api.open-meteo.com/v1/meteoswiss',
        'url_param': 'meteoswiss_icon_ch1',
        'max_tage': 2, 'aufloesung': '1 km',
        'gebiet': 'Alpenraum, Schwerpunkt West',
        'quelle': 'MeteoSchweiz'},
    'MeteoSwiss ICON-CH2 2km': {
        'url': 'https://api.open-meteo.com/v1/meteoswiss',
        'url_param': 'meteoswiss_icon_ch2',
        'max_tage': 5, 'aufloesung': '2,1 km',
        'gebiet': 'Alpenraum, Schwerpunkt West',
        'quelle': 'MeteoSchweiz'},
}

ENSEMBLE_MODELLE = {
    'ICON-D2-EPS 2km': {'model_key': 'icon_d2', 'mitglieder': 20,
                        'max_stunden': 48, 'aufloesung': '2 km', 'fokus': 'kurzfrist'},
    'ICON-EU-EPS 13km': {'model_key': 'icon_eu', 'mitglieder': 40,
                         'max_stunden': 120, 'aufloesung': '13 km', 'fokus': 'mittelfrist'},
    'ICON-EPS Global 26km': {'model_key': 'icon_seamless', 'mitglieder': 40,
                             'max_stunden': 180, 'aufloesung': '26 km', 'fokus': 'mittelfrist'},
    'ECMWF IFS ENS 9km': {'model_key': 'ecmwf_ifs025', 'model_key_alt': ['ecmwf_ifs04'], 'mitglieder': 51,
                          'max_stunden': 360, 'aufloesung': '9 km', 'fokus': 'langfrist'},
    'GFS ENS 25km': {'model_key': 'gfs025', 'model_key_alt': ['gfs05', 'gfs_seamless'], 'mitglieder': 31,
                     'max_stunden': 240, 'aufloesung': '25 km', 'fokus': 'langfrist'},
    'GEM ENS 25km': {'model_key': 'gem_global', 'mitglieder': 21,
                     'max_stunden': 384, 'aufloesung': '25 km', 'fokus': 'langfrist'},
}
# ============================================================
# MODELLFAMILIEN — Grundlage einer ehrlichen Übereinstimmungsangabe
# ============================================================
# Mehrere der abgerufenen Quellen teilen sich denselben dynamischen Kern.
# ICON-D2, ICON-EU, ICON global und die MeteoSchweiz-Läufe stammen alle
# vom ICON-Kern; ECMWF IFS deterministisch und IFS-ENS sind derselbe
# Modelllauf in unterschiedlicher Aufbereitung. Zählte man sie einzeln,
# entstünde eine Scheinübereinstimmung: Neun von elf Quellen sind sich
# einig, weil ein einziges Modell sich entschieden hat.
#
# Deshalb bekommt jede Familie ein festes Gesamtgewicht, das intern auf
# ihre vorhandenen Mitglieder aufgeteilt wird. Die angezeigte
# Übereinstimmung bezieht sich auf die Zahl unabhängiger Familien.
MODELL_FAMILIE = {
    # --- ICON-Kern (DWD, MeteoSchweiz) ---
    'ICON EU 7km': 'icon',
    'MeteoSwiss ICON-CH1 1km': 'icon',
    'MeteoSwiss ICON-CH2 2km': 'icon',
    'ICON-D2-EPS 2km': 'icon',
    'ICON-EU-EPS 13km': 'icon',
    'ICON-EPS Global 26km': 'icon',
    # --- ECMWF IFS ---
    'ECMWF IFS 9km': 'ifs',
    'ECMWF IFS ENS 9km': 'ifs',
    # --- NOAA GFS ---
    'GFS 25km': 'gfs',
    'GFS ENS 25km': 'gfs',
    # --- Environment Canada ---
    'GEM ENS 25km': 'gem',
    # --- Météo-France: ARPEGE liefert AROME die Randwerte,
    #     die beiden sind daher nicht unabhängig voneinander ---
    'ARPEGE 13km': 'mf',
    'GeoSphere AROME 2.5km': 'mf',
}

# Grundgewicht je Familie. ECMWF und ICON schneiden über Mitteleuropa
# in unabhängigen Verifikationen am besten ab, GFS und GEM fallen ab.
FAMILIEN_GEWICHT = {
    'ifs':  1.00,
    'icon': 1.00,
    'mf':   0.85,
    'gfs':  0.65,
    'gem':  0.55,
}
FAMILIEN_NAME = {
    'ifs': 'ECMWF IFS', 'icon': 'ICON (DWD/MeteoSchweiz)',
    'mf': 'Météo-France ARPEGE/AROME', 'gfs': 'NOAA GFS',
    'gem': 'Environment Canada GEM',
}


def _aufloesung_km(text):
    """Maschenweite als Zahl aus Angaben wie '2,5 km' herauslösen."""
    try:
        return float(str(text).split()[0].replace(',', '.'))
    except Exception:
        return 25.0


def skill_gewicht(aufloesung_km, stunden_voraus):
    """
    Gütegewicht in Abhängigkeit von Maschenweite und Vorlaufzeit.

    Ein 2,5-km-Modell bildet Täler, Hangneigung und lokale Konvektion
    ab, ein 25-km-Modell sieht davon nichts. Kurzfristig ist dieser
    Unterschied entscheidend. Ab etwa fünf Tagen verliert die feine
    Auflösung ihren Vorteil, weil dann die großräumige Entwicklung
    dominiert — dort sind die globalen Modelle gleichwertig.
    """
    r = float(aufloesung_km)
    h = float(stunden_voraus)
    if h <= 24:
        if r <= 3:    return 2.30
        if r <= 8:    return 1.70
        if r <= 15:   return 1.15
        return 0.60
    if h <= 72:
        if r <= 3:    return 1.85
        if r <= 8:    return 1.55
        if r <= 15:   return 1.20
        return 0.80
    if h <= 120:
        if r <= 8:    return 1.30
        if r <= 15:   return 1.15
        return 0.95
    return 1.00


ENSEMBLE_API = 'https://ensemble-api.open-meteo.com/v1/ensemble'
ENS_VARIABLEN = ('temperature_2m,relative_humidity_2m,precipitation,'
                 'wind_speed_10m')
HAUPT_VARIABLEN = ('temperature_2m,relative_humidity_2m,precipitation,'
                   'wind_speed_10m,wind_gusts_10m,wind_direction_10m,'
                   'cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,'
                   'dew_point_2m,apparent_temperature,cape,uv_index,'
                   'precipitation_probability,is_day,'
                   'shortwave_radiation,temperature_850hPa,'
                   'convective_inhibition,lifted_index,'
                   'freezing_level_height,visibility,snowfall,'
                   'wind_speed_80m')

# Höhengradienten für alpine Korrektur
# Nur noch Rückfallwert. Der tatsächlich verwendete Gradient wird in
# bestimme_gradient() aus den 850-hPa-Daten abgeleitet, weil ein fester
# Wert bei Inversionen das Vorzeichen verfehlt.
TEMP_GRADIENT_PRO_100M = 0.65      # °C Abnahme je 100 m
WIND_FAKTOR_PRO_1000M = 0.35       # relative Windzunahme je 1000 m
ALPIN_SCHWELLE_M = 1200            # ab hier gilt ein Standort als alpin

# ============================================================
# BODEN-KONFIGURATION (kalibrierbar!)
# ============================================================
# Diese Werte sind Startwerte für Grünland/Normalboden und
# sollten mit Rückmeldung der Landwirte angepasst werden.

# Bodenarten unterscheiden sich erheblich in Wasserleitfähigkeit und
# Tragfähigkeit. Die Kennwerte steuern, wie schnell ein Boden abtrocknet
# und ab welcher Wasserbilanz er als befahrbar gilt.
# Bodenkennwerte. "nutzbare_fk" ist die nutzbare Feldkapazität im
# Oberboden (0–30 cm) in mm — die Wassermenge, die der Boden gegen die
# Schwerkraft halten kann. Oberhalb dieser Grenze steht Sickerwasser an,
# der Boden ist gesättigt und verdichtungsempfindlich. Erst mit dieser
# Größe wird aus der bloßen Bilanzzahl ein Sättigungsgrad, der sich
# zwischen den Bodenarten überhaupt vergleichen lässt.
BODENARTEN = {
    'Sand / leichter Boden': {
        'drainage': 11.0, 'bilanz_faktor': 1.55, 'infiltration': 14.0,
        'nutzbare_fk': 45.0, 'abtrocknung': 1.45,
        'hinweis': 'trocknet rasch ab, früh wieder befahrbar'},
    'Lehm / mittlerer Boden': {
        'drainage': 6.0, 'bilanz_faktor': 1.0, 'infiltration': 8.0,
        'nutzbare_fk': 60.0, 'abtrocknung': 1.0,
        'hinweis': 'ausgeglichenes Verhalten, Standardannahme'},
    'Ton / schwerer Boden': {
        'drainage': 3.2, 'bilanz_faktor': 0.62, 'infiltration': 4.0,
        'nutzbare_fk': 75.0, 'abtrocknung': 0.62,
        'hinweis': 'hält Wasser lange, Verdichtungsgefahr'},
    'Moor / Anmoor': {
        'drainage': 2.4, 'bilanz_faktor': 0.48, 'infiltration': 3.0,
        'nutzbare_fk': 95.0, 'abtrocknung': 0.45,
        'hinweis': 'sehr empfindlich, lange Abtrocknungszeit'},
}
BODENART_STANDARD = 'Lehm / mittlerer Boden'

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
            'trocknung_relevant': True, 'trocknung_min': 0.60,
            'beschreibung': 'Mahd + 2–3 Tage Abtrocknung am Boden. Bewertet '
                            'wird die Trocknungsleistung aus Sättigungs'
                            'defizit, Wind und Einstrahlung.'},
        'Heuernte (Belüftung)': {
            'temp_min': 13, 'temp_max': 35, 'humidity_max': 75, 'wind_max': 25,
            'precip_max': 0.0, 'trockenfenster_h': 12, 'boden_relevant': True,
            'trocknung_relevant': True, 'trocknung_min': 0.40,
            'beschreibung': 'Anwelken, Trocknung dann per Belüftung im Stock'},
        'Silieren / Anwelksilage': {
            'temp_min': 10, 'temp_max': 35, 'humidity_max': 78, 'wind_max': 35,
            'precip_max': 0.0, 'trockenfenster_h': 8, 'boden_relevant': True,
            'trocknung_relevant': True, 'trocknung_min': 0.32,
            'beschreibung': 'Anwelken auf 30–40 % Trockenmasse; Regen auf dem '
                            'Schwad führt zu Wiederbefeuchtung und '
                            'Fehlgärung'},
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
            'temp_min': 16, 'temp_max': 38, 'humidity_max': 60, 'wind_max': 35,
            'precip_max': 0.0, 'trockenfenster_h': 8, 'boden_relevant': True,
            'trocknung_relevant': True, 'trocknung_min': 0.55,
            'beschreibung': 'Die Luftfeuchte dient als Anhaltspunkt für die '
                            'Kornfeuchte; maßgeblich bleibt die Messung am '
                            'Bestand'},
        'Pflügen / Bodenbearbeitung': {
            'temp_min': 3, 'temp_max': 35, 'wind_max': 40, 'precip_max': 1.0,
            'boden_relevant': True,
            'beschreibung': 'Boden abgetrocknet genug zum Befahren'},
        'Spritzen Fungizide': {
            'temp_min': 10, 'temp_max': 25, 'humidity_min': 55, 'humidity_max': 85,
            'wind_max': 15, 'precip_max': 0.0, 'trockenfenster_h': 3,
            'boden_relevant': False,
            'beschreibung': 'Mäßige Feuchte für gute Benetzung, danach '
                            'mindestens drei Stunden regenfrei zum Antrocknen'},
        'Spritzen Herbizide': {
            'temp_min': 8, 'temp_max': 25, 'humidity_min': 50, 'humidity_max': 90,
            'wind_max': 12, 'precip_max': 0.0, 'trockenfenster_h': 4,
            'boden_relevant': False,
            'beschreibung': 'Sehr windstill wegen Abdrift, danach mindestens '
                            'vier Stunden regenfrei für die Aufnahme'},
        'Spritzen Insektizide': {
            'temp_min': 8, 'temp_max': 25, 'humidity_min': 45, 'humidity_max': 90,
            'wind_max': 12, 'precip_max': 0.0, 'trockenfenster_h': 3,
            'boden_relevant': False,
            'beschreibung': 'Windstill, kühl und außerhalb des Bienenflugs; '
                            'bevorzugt Abend oder früher Morgen'},
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
            'temp_min': 10, 'temp_max': 26, 'humidity_min': 50, 'humidity_max': 85,
            'wind_max': 12, 'precip_max': 0.0, 'trockenfenster_h': 4,
            'boden_relevant': False,
            'beschreibung': 'Laubwand erfordert sehr geringe Abdrift; '
                            'nach dem Auftrag vier Stunden regenfrei'},
        'Obstbau Spritzung': {
            'temp_min': 8, 'temp_max': 27, 'humidity_min': 50, 'humidity_max': 88,
            'wind_max': 14, 'precip_max': 0.0, 'trockenfenster_h': 4,
            'boden_relevant': False,
            'beschreibung': 'Baumkrone gleichmäßig benetzen; danach vier '
                            'Stunden regenfrei, sonst Abwaschung'},
        'Obsternte': {
            'outdoor': True,
            'temp_min': 5, 'temp_max': 32, 'humidity_max': 85, 'wind_max': 30,
            'precip_max': 0.0, 'trockenfenster_h': 4, 'boden_relevant': False,
            'beschreibung': 'Trocken, Früchte nicht nass ernten'},
        'Gemüsebau Pflanzung': {
            'temp_min': 8, 'temp_max': 28, 'humidity_min': 40, 'wind_max': 20,
            'precip_max': 0.5, 'boden_relevant': True, 'duerre_relevant': True,
            'beschreibung': 'Mild, gleichmäßige Bodenfeuchte'},
    },
    'Bergsport & Outdoor': {
        'Hochtour / Gletscher': {
            'outdoor': True,
            'temp_min': -25, 'temp_max': 30, 'wind_max': 40, 'precip_max': 0.0,
            'trockenfenster_h': 0, 'boden_relevant': False, 'alpin_bewölkung': True, 'sicht_relevant': True,
            'nullgrad_relevant': True,
            'beschreibung': 'Niederschlagsfrei, wenig Wind, früher Aufbruch. '
                            'Höhenkorrektur für Temperatur, Wind und Böen ist aktiv.'},
        'Klettern (Fels)': {
            'outdoor': True,
            'temp_min': 8, 'temp_max': 32, 'humidity_max': 90, 'wind_max': 30,
            'precip_max': 0.0, 'trockenfenster_h': 12, 'boden_relevant': False,
            'alpin_bewölkung': True, 'sicht_relevant': True,
            'nullgrad_relevant': True,
            'beschreibung': 'Fels muss abgetrocknet sein (12h trocken)'},
        'Wandern / Hüttentour': {
            'outdoor': True,
            'temp_min': 3, 'temp_max': 32, 'wind_max': 45, 'precip_max': 0.0,
            'boden_relevant': False,
            'beschreibung': 'Kein Dauerregen, vertretbarer Wind'},
        'Mountainbike / Trail': {
            'outdoor': True,
            'temp_min': 5, 'temp_max': 33, 'wind_max': 40, 'precip_max': 0.0,
            'trockenfenster_h': 6, 'boden_relevant': False,
            'beschreibung': 'Trails abgetrocknet, kein Regen'},
    },
    'Straßenverkehr': {
        'Autofahren': {
            'temp_min': -30, 'temp_max': 45, 'wind_max': 90, 'precip_max': 10.0,
            'boden_relevant': False,
            'beschreibung': 'Nur bei extremen Bedingungen eingeschränkt: '
                            'Schneeglätte, Eisregen, Sturm, Starkregen',
            '_hinweis': 'autofahren'},
        'Motorradfahren': {
            'temp_min': 5, 'temp_max': 40, 'wind_max': 55, 'precip_max': 0.3,
            'boden_relevant': False,
            'beschreibung': 'Nasse Fahrbahn, Wind und Kälte senken '
                            'Haftung und Sicherheit deutlich',
            '_hinweis': 'motorrad'},
        'Rennradfahren': {
            'temp_min': 8, 'temp_max': 38, 'wind_max': 35, 'precip_max': 0.1,
            'boden_relevant': False,
            'beschreibung': 'Sehr windempfindlich; nasse Straße und '
                            'Kälte kritisch für Reifen und Bremsen',
            '_hinweis': 'rennrad'},
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
            'outdoor': True,
            'temp_min': 3, 'temp_max': 34, 'wind_max': 25, 'precip_max': 0.0,
            'boden_relevant': False,
            'beschreibung': 'Trocken, wenig Wind (Absturzgefahr)'},
        'Veranstaltung im Freien': {
            'outdoor': True,
            'temp_min': 12, 'temp_max': 33, 'wind_max': 30, 'precip_max': 0.0,
            'boden_relevant': False,
            'beschreibung': 'Trocken und angenehm'},
        'Rasen mähen (privat)': {
            'outdoor': True,
            'temp_min': 8, 'temp_max': 33, 'humidity_max': 90, 'wind_max': 35,
            'precip_max': 0.0, 'trockenfenster_h': 4, 'boden_relevant': False,
            'beschreibung': 'Gras trocken, kein Regen'},
    },
    'Freizeit & Erholung': {
        'Angeln': {
            'outdoor': True,
            'temp_min': 4, 'temp_max': 32, 'wind_max': 25, 'precip_max': 0.3,
            'boden_relevant': False,
            'beschreibung': 'Wenig Wind für die Schnurführung; leichter Regen '
                            'stört kaum, Gewitter beendet den Ansitz'},
        'Baden & Freibad': {
            'outdoor': True,
            'temp_min': 22, 'temp_max': 38, 'wind_max': 25, 'precip_max': 0.0,
            'boden_relevant': False,
            'beschreibung': 'Warm, sonnig und trocken'},
        'Camping & Zelten': {
            'outdoor': True,
            'temp_min': 8, 'temp_max': 32, 'wind_max': 30, 'precip_max': 0.0,
            'trockenfenster_h': 8, 'boden_relevant': True,
            'beschreibung': 'Trockener Auf- und Abbau, tragfähiger Untergrund'},
        'Grillen & Gartenfest': {
            'outdoor': True,
            'temp_min': 14, 'temp_max': 34, 'wind_max': 25, 'precip_max': 0.0,
            'boden_relevant': False,
            'beschreibung': 'Mild, trocken und windarm'},
        'Landschaftsfotografie': {
            'outdoor': True, 'sicht_kritisch': True,
            'temp_min': -12, 'temp_max': 35, 'wind_max': 35, 'precip_max': 0.2,
            'boden_relevant': False,
            'beschreibung': 'Klare Sicht und interessantes Licht; leichte '
                            'Bewölkung ist oft von Vorteil'},
        'Laufen & Joggen': {
            'outdoor': True,
            'temp_min': 2, 'temp_max': 26, 'humidity_max': 88, 'wind_max': 30,
            'precip_max': 0.2, 'boden_relevant': False,
            'beschreibung': 'Kühl bis mild; Hitze und Schwüle belasten '
                            'den Kreislauf'},
        'Sternbeobachtung': {
            'outdoor': True, 'sicht_kritisch': True,
            'temp_min': -15, 'temp_max': 30, 'wind_max': 25, 'precip_max': 0.0,
            'boden_relevant': False, 'alpin_bewölkung': True,
            'sicht_relevant': True,
            'beschreibung': 'Klarer Himmel ist entscheidend; jede Bewölkung '
                            'schränkt die Sicht stark ein'},
    },
    'Haus & Garten': {
        'Fenster putzen': {
            'temp_min': 4, 'temp_max': 28, 'wind_max': 25, 'precip_max': 0.0,
            'trockenfenster_h': 3, 'boden_relevant': False,
            'beschreibung': 'Bedeckt ist besser als pralle Sonne, sonst '
                            'trocknet das Wasser zu rasch und schliert'},
        'Heckenschnitt & Gartenarbeit': {
            'outdoor': True,
            'temp_min': 5, 'temp_max': 30, 'wind_max': 35, 'precip_max': 0.1,
            'boden_relevant': False,
            'beschreibung': 'Trocken und nicht zu heiß'},
        'Holz & Terrasse ölen': {
            'temp_min': 12, 'temp_max': 30, 'humidity_max': 70, 'wind_max': 20,
            'precip_max': 0.0, 'trockenfenster_h': 24, 'boden_relevant': False,
            'beschreibung': 'Untergrund muss trocken sein und einen Tag '
                            'lang trocken bleiben'},
        'Umzug & Transport': {
            'temp_min': -8, 'temp_max': 32, 'wind_max': 40, 'precip_max': 0.1,
            'boden_relevant': False,
            'beschreibung': 'Trocken, damit Möbel und Kartons keinen '
                            'Schaden nehmen'},
        'Wäsche im Freien trocknen': {
            'temp_min': 8, 'temp_max': 40, 'humidity_max': 70, 'wind_max': 35,
            'precip_max': 0.0, 'trockenfenster_h': 6, 'boden_relevant': False,
            'beschreibung': 'Trockene Luft und etwas Wind beschleunigen '
                            'das Trocknen deutlich'},
    },
    'Frost': {
        'Frostüberwachung': {
            'modus': 'frost', 'temp_warn': 3, 'temp_frost': 0,
            'boden_relevant': False,
            'beschreibung': 'Überwachung Tiefsttemperaturen und Frostrisiko'},
    },
}


def _alphabetisch(katalog):
    """
    Ordnet Kategorien und Tätigkeiten alphabetisch.
    Deutsche Umlaute werden dabei wie ihre Grundvokale einsortiert,
    damit „Düngung“ nicht hinter „Z“ landet.
    """
    def schluessel(text):
        ersetzungen = (('ä', 'a'), ('ö', 'o'), ('ü', 'u'), ('ß', 'ss'))
        t = text.lower()
        for a, b in ersetzungen:
            t = t.replace(a, b)
        return t

    return {kat: dict(sorted(faelle.items(), key=lambda p: schluessel(p[0])))
            for kat, faelle in sorted(katalog.items(),
                                      key=lambda p: schluessel(p[0]))}


ANWENDUNGSFAELLE = _alphabetisch(ANWENDUNGSFAELLE)


def finde_fall(name):
    """Sucht einen Anwendungsfall über alle Kategorien und gibt (kategorie, params)."""
    for kat, faelle in ANWENDUNGSFAELLE.items():
        if name in faelle:
            return kat, faelle[name]
    return None, None


# Fünfstufige Eignungsskala
# Farbwerte aus der Markenpalette abgeleitet (Grün → Gelb → Rot)
# Die fünf Farben sind so gewählt, dass sie sich auch in der Helligkeit
# deutlich unterscheiden: von dunklem Grün über Gelb bis zu kräftigem Rot.
# Damit bleibt die Abstufung bei Rot-Grün-Schwäche als Hell-Dunkel-Verlauf
# erkennbar, ohne dass Symbole die Matrix überladen. Wer es genau wissen
# will, sieht beim Antippen einer Zelle die Stufe im Klartext.
STUFEN = {
    4: {'name': 'sehr gut',   'farbe': '#2E8B57', 'kurz': 'sehr gut'},
    3: {'name': 'gut',        'farbe': '#66BB6A', 'kurz': 'gut'},
    2: {'name': 'bedingt',    'farbe': '#FFC107', 'kurz': 'bedingt'},
    1: {'name': 'ungünstig',  'farbe': '#F57C00', 'kurz': 'ungünstig'},
    0: {'name': 'ungeeignet', 'farbe': '#D93B30', 'kurz': 'ungeeignet'},
}
FARBEN = {v['name']: v['farbe'] for v in STUFEN.values()}
AMPEL_INT = {v['name']: k for k, v in STUFEN.items()}
INT_AMPEL = {k: v['name'] for k, v in STUFEN.items()}

# Bezugsgrößen für die Normierung der Grenzwertüberschreitung.
# Alpine Bewölkungs-Logik
ALPIN_BEWÖLKUNG_FAELLE = {'Hochtour / Gletscher', 'Klettern (Fels)'}
ALPIN_HOEHE_SCHWELLE = 1500   # m ü.A. ab der Bewölkung in Bewertung einfließt

# Sie übersetzen eine absolute Überschreitung in einen Prozentwert,
# damit Temperatur, Feuchte, Wind und Niederschlag vergleichbar werden.
BEZUG_TEMP = 10.0        # °C entsprechen 100 % Überschreitung
BEZUG_FEUCHTE = 20.0     # Prozentpunkte entsprechen 100 %
BEZUG_WIND_MIN = 10.0    # km/h Untergrenze des Bezugs
BEZUG_NS_MIN = 1.5       # mm/h Untergrenze des Bezugs
WOCHENTAGE_LANG = {'Monday': 'Montag', 'Tuesday': 'Dienstag',
                   'Wednesday': 'Mittwoch', 'Thursday': 'Donnerstag',
                   'Friday': 'Freitag', 'Saturday': 'Samstag',
                   'Sunday': 'Sonntag'}
WOCHENTAGE = {'Monday':'Mo','Tuesday':'Di','Wednesday':'Mi','Thursday':'Do',
              'Friday':'Fr','Saturday':'Sa','Sunday':'So'}

def tag_kurz(dt):
    return WOCHENTAGE.get(dt.strftime('%A'), dt.strftime('%a'))


# ============================================================
# DATEN ABRUFEN
# ============================================================

def _bezirk_kurz(bezeichnung):
    """
    Kürzt amtliche Verwaltungsbezeichnungen auf den reinen Namen.
    Aus "Politischer Bezirk Scheibbs" wird "Scheibbs".
    """
    if not bezeichnung:
        return None
    text = str(bezeichnung).strip()
    praefixe = ['Politischer Bezirk ', 'Bezirk ', 'Landkreis ',
                'Verwaltungsgemeinschaft ', 'Kreis ', 'Bezirksteil ',
                'Statutarstadt ']
    for p in praefixe:
        if text.startswith(p):
            text = text[len(p):].strip()
            break
    # Nachgestellte Zusätze wie "(Stadt)" entfernen
    if text.endswith(' (Stadt)'):
        text = text[:-8].strip()
    return text or None


def geocode_ort(name):
    """
    Sucht Orte und liefert immer eine Postleitzahl, sofern vorhanden.

    Primärquelle ist Nominatim (OpenStreetMap), da diese zuverlässig
    Postleitzahlen für den deutschsprachigen Raum liefert. Fällt sie aus,
    wird auf die Open-Meteo Geocoding-API zurückgegriffen.
    Die Seehöhe wird anschließend über Open-Meteo ergänzt.
    """
    treffer = _geocode_nominatim(name)
    if not treffer:
        treffer = _geocode_openmeteo(name)
    return treffer or None


def _geocode_nominatim(name):
    """OpenStreetMap-Suche: liefert Postleitzahlen zuverlässig."""
    try:
        r = requests.get(
            'https://nominatim.openstreetmap.org/search',
            params={'q': name, 'format': 'jsonv2', 'addressdetails': 1,
                    'limit': 8, 'accept-language': 'de',
                    'countrycodes': 'at,de,ch,it,li'},
            headers={'User-Agent': NOMINATIM_KENNUNG},
            timeout=8)
        daten = r.json()
    except Exception:
        return []
    if not isinstance(daten, list) or not daten:
        return []

    treffer = []
    for x in daten:
        adr = x.get('address', {}) or {}
        ortsname = (adr.get('city') or adr.get('town') or adr.get('village')
                    or adr.get('municipality') or adr.get('hamlet')
                    or x.get('name') or name)
        plz = adr.get('postcode')
        bezirk = _bezirk_kurz(adr.get('county') or adr.get('district'))
        land = adr.get('state')
        staat = adr.get('country')

        teile = [f"{plz} {ortsname}" if plz else str(ortsname)]
        if bezirk and bezirk != ortsname:
            teile.append(bezirk)
        if land and land != bezirk:
            teile.append(land)
        if staat:
            teile.append(staat)

        try:
            lat = float(x['lat']); lon = float(x['lon'])
        except (KeyError, TypeError, ValueError):
            continue

        eintrag = {'lat': lat, 'lon': lon, 'hoehe': None,
                   'plz': plz, 'name': ortsname,
                   'label': ', '.join(teile)}
        # Duplikate vermeiden
        if not any(abs(t['lat'] - lat) < 0.01 and abs(t['lon'] - lon) < 0.01
                   for t in treffer):
            treffer.append(eintrag)

    # Seehöhen ergänzen (ein Aufruf je Treffer, maximal 5)
    for t in treffer[:5]:
        t['hoehe'] = hole_modellhoehe(t['lat'], t['lon'])
        if t['hoehe'] is not None:
            t['label'] += f", {t['hoehe']:.0f} m"
    return treffer


def _geocode_openmeteo(name):
    """Rückfallebene: Open-Meteo Geocoding."""
    try:
        r = requests.get('https://geocoding-api.open-meteo.com/v1/search',
                         params={'name': name, 'count': 8, 'language': 'de'},
                         timeout=8)
        data = r.json()
    except Exception:
        return []
    if 'results' not in data or not data['results']:
        return []

    treffer = []
    for x in data['results']:
        hoehe = x.get('elevation')
        pc = x.get('postcodes')
        plz = None
        if isinstance(pc, list) and pc:
            plz = str(pc[0]).strip()
        elif isinstance(pc, str) and pc.strip():
            plz = pc.strip()

        teile = [f"{plz} {x['name']}" if plz else x['name']]
        def _b(s):
            if not s: return s
            for p in ['Politischer Bezirk ','Bezirk ','Landkreis ','Stadtbezirk ']:
                if s.startswith(p): return s[len(p):]
            return s
        bezirk = _b(x.get('admin2') or x.get('admin3'))
        land = x.get('admin1')
        if bezirk and bezirk != x['name']:
            teile.append(bezirk)
        if land and land != bezirk:
            teile.append(land)
        if x.get('country'):
            teile.append(x['country'])
        if hoehe is not None:
            teile.append(f"{hoehe:.0f} m")

        treffer.append({'lat': x['latitude'], 'lon': x['longitude'],
                        'hoehe': hoehe, 'plz': plz, 'name': x['name'],
                        'label': ', '.join(teile)})
    return treffer


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


H_850HPA_M = 1500.0     # Näherung für die Höhe der 850-hPa-Fläche
# Ein negativer Gradient bedeutet: oben wärmer als unten — eine Inversion.
# Der kleine Abstand zu null fängt Rechenrauschen ab.
GRADIENT_UNTERGRENZE = -0.05   # K/100 m


def bestimme_gradient(haupt_daten, modellhoehe):
    """
    Ermittelt den tatsächlichen vertikalen Temperaturgradienten aus dem
    Abstand zwischen Bodentemperatur und der 850-hPa-Fläche.

    Der bisher verwendete feste Wert von 0,65 K je 100 m gilt nur bei
    durchmischter Atmosphäre. In der kalten Jahreszeit bildet sich in
    Becken und Tälern regelmäßig eine Inversion: unten kalt, oben mild.
    Wer dann nach Standardgradient rechnet, macht in einem Tal die
    Temperatur zu warm und übersieht Frost — genau in der Lage, in der
    Frostwarnungen den größten Wert hätten.

    Rückgabe: DataFrame mit Zeit, Gradient (K/100 m) und Inversionsflag.
    """
    reihen = []
    for df in haupt_daten.values():
        if 't850' not in df.columns:
            continue
        t850 = pd.to_numeric(df['t850'], errors='coerce')
        tsfc = pd.to_numeric(df['temp'], errors='coerce')
        if t850.isna().all():
            continue
        dz = max(H_850HPA_M - float(modellhoehe or 0), 300.0) / 100.0
        reihen.append(pd.DataFrame({'time': df['time'],
                                    'grad': (tsfc - t850) / dz}))
    if not reihen:
        return None

    g = pd.concat(reihen).groupby('time', as_index=False)['grad'].median()
    # Plausibilitätsgrenzen: stärker als trockenadiabatisch ist unphysikalisch,
    # extreme Inversionen werden gedeckelt statt verworfen.
    g['grad'] = g['grad'].clip(-1.2, 0.98)
    g['inversion'] = g['grad'] < GRADIENT_UNTERGRENZE
    return g


def saettigungsdampfdruck(t_c):
    """Sättigungsdampfdruck in kPa nach Magnus-Formel."""
    t = pd.to_numeric(t_c, errors='coerce')
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def dampfdruckdefizit(t_c, rel_feuchte):
    """
    Sättigungsdefizit (VPD) in kPa.

    Das VPD ist die physikalisch maßgebliche Größe für Verdunstung und
    damit für die Heutrocknung. Die relative Feuchte allein taugt dafür
    nicht: 70 % bei 12 °C und 70 % bei 28 °C bedeuten völlig
    unterschiedliche Trocknungsleistungen.
    """
    es = saettigungsdampfdruck(t_c)
    rf = pd.to_numeric(rel_feuchte, errors='coerce').clip(0, 100)
    return (es * (1.0 - rf / 100.0)).clip(lower=0)


def trocknungsrate(temp, feuchte, wind, strahlung=None):
    """
    Relative Trocknungsleistung für Halmfutter, Skala 0 bis rund 1,5.

    Grundlage ist das Dampfdruckdefizit, verstärkt durch Wind (erneuert
    die gesättigte Grenzschicht über dem Schwad) und durch Einstrahlung
    (erwärmt das Erntegut über die Lufttemperatur). Der Wert ist keine
    Absolutgröße in kg Wasser je Stunde, sondern ein Index: ab etwa 0,45
    trocknet das Futter brauchbar, unter 0,2 praktisch nicht.
    """
    vpd = dampfdruckdefizit(temp, feuchte)
    w = pd.to_numeric(wind, errors='coerce').fillna(0).clip(0, 40)
    windfaktor = 1.0 + 0.45 * (w / 20.0) ** 0.7
    if strahlung is not None:
        st_w = pd.to_numeric(strahlung, errors='coerce').fillna(0).clip(0, 1000)
        strahlfaktor = 0.55 + 0.85 * (st_w / 600.0).clip(0, 1.3)
    else:
        strahlfaktor = 1.0
    return (vpd * 0.85 * windfaktor * strahlfaktor)


def hoehenkorrektur(df, delta_h, spalten=None, gradient_df=None):
    """
    Rechnet Modellwerte auf die tatsächliche Standorthöhe um.
    delta_h = Standorthöhe − Modellhöhe (positiv = Standort liegt höher).

    Berücksichtigt werden:
      • Temperatur und Taupunkt über den vertikalen Gradienten. Dabei wird
        zwischen feuchter und trockener Schichtung unterschieden: Bei hoher
        relativer Feuchte ist die Abnahme geringer (feuchtadiabatisch, rund
        0,5 °C je 100 m), bei trockener Luft stärker (bis 0,85 °C je 100 m).
      • Wind: Zunahme mit der Höhe durch geringere Bodenreibung.
      • Böen: verstärken sich in exponierter Lage überproportional.
      • Relative Feuchte: steigt mit sinkender Temperatur an.
    """
    if not delta_h or abs(delta_h) < 100:
        return df
    df = df.copy()

    # --- Temperaturgradient nach Feuchte differenzieren ---
    if 'feuchte' in df.columns:
        feuchte = pd.to_numeric(df['feuchte'], errors='coerce').fillna(70.0)
    elif 'feuchte_median' in df.columns:
        feuchte = pd.to_numeric(df['feuchte_median'], errors='coerce').fillna(70.0)
    else:
        feuchte = pd.Series(np.full(len(df), 70.0), index=df.index)

    # Bevorzugt der aus 850 hPa abgeleitete tatsächliche Gradient. Nur wenn
    # dieser fehlt, wird auf die Feuchteschätzung zurückgegriffen
    # (0,50 K/100 m gesättigt bis 0,85 K/100 m trocken).
    gradient = None
    if gradient_df is not None and not gradient_df.empty:
        _g = df[['time']].merge(gradient_df[['time', 'grad']],
                                on='time', how='left')
        gradient = pd.to_numeric(_g['grad'], errors='coerce')
        gradient.index = df.index
    if gradient is None or gradient.isna().all():
        gradient = 0.85 - 0.35 * (feuchte.clip(0, 100) / 100.0)
    else:
        gradient = gradient.fillna(0.85 - 0.35 * (feuchte.clip(0, 100) / 100.0))
    dt = (delta_h / 100.0) * gradient

    for sp in (spalten or ['temp', 'temp_median', 'temp_p10', 'temp_p90',
                           'taupunkt']):
        if sp in df.columns:
            df[sp] = pd.to_numeric(df[sp], errors='coerce') - dt

    # --- Wind: nimmt mit der Höhe zu (geringere Reibung, Exposition) ---
    windfaktor = 1.0 + (delta_h / 1000.0) * WIND_FAKTOR_PRO_1000M
    windfaktor = float(np.clip(windfaktor, 0.6, 2.5))
    # Böen verstärken sich in exponierter Lage stärker als der Mittelwind
    boeenfaktor = 1.0 + (delta_h / 1000.0) * (WIND_FAKTOR_PRO_1000M * 1.35)
    boeenfaktor = float(np.clip(boeenfaktor, 0.6, 3.0))

    for sp in ['wind', 'wind_median']:
        if sp in df.columns:
            df[sp] = pd.to_numeric(df[sp], errors='coerce') * windfaktor
    if 'boeen' in df.columns:
        df['boeen'] = pd.to_numeric(df['boeen'], errors='coerce') * boeenfaktor

    # --- Relative Feuchte steigt bei sinkender Temperatur ---
    for sp in ['feuchte', 'feuchte_median']:
        if sp in df.columns and delta_h > 0:
            f_alt = pd.to_numeric(df[sp], errors='coerce')
            # Bei Inversion (dt negativ) würde die Formel die Feuchte
            # unsinnig absenken — dort bleibt der Modellwert stehen.
            zuschlag = (dt * 4.0).clip(lower=0)
            df[sp] = (f_alt + zuschlag).clip(0, 100)

    return df


def schneefallgrenze(temp_standort, hoehe_standort, feuchte=70.0):
    """
    Schätzt die Höhe der Schneefallgrenze aus Temperatur und Standorthöhe.

    Faustregel der Alpinmeteorologie: Der Übergang von Regen zu Schnee
    liegt rund 100 bis 300 Meter unterhalb der Nullgradgrenze, weil
    fallende Flocken beim Schmelzen Energie entziehen. Bei trockener Luft
    liegt sie tiefer als bei feuchter.
    """
    if temp_standort is None or hoehe_standort is None:
        return None
    try:
        t = float(temp_standort)
        h = float(hoehe_standort)
    except (TypeError, ValueError):
        return None

    gradient = 0.85 - 0.35 * (float(feuchte) / 100.0)
    nullgradgrenze = h + (t / gradient) * 100.0
    # Abschlag für den Schmelzprozess: trockene Luft = größerer Abschlag
    abschlag = 100.0 + 200.0 * (1.0 - float(feuchte) / 100.0)
    return nullgradgrenze - abschlag


def himmelsrichtung(grad):
    """Wandelt Gradzahl in Himmelsrichtungs-Kürzel."""
    if grad is None or (isinstance(grad, float) and np.isnan(grad)):
        return '–'
    richtungen = ['N', 'NNO', 'NO', 'ONO', 'O', 'OSO', 'SO', 'SSO',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    return richtungen[int((float(grad) + 11.25) % 360 // 22.5)]


def hole_hauptlauf(lat, lon, modell_name, tage):
    cfg = HAUPTLAUFE[modell_name]
    params = {'latitude': lat, 'longitude': lon, 'hourly': HAUPT_VARIABLEN,
              'forecast_days': min(tage, cfg['max_tage']), 'timezone': 'Europe/Vienna'}
    # Modellnamen und mögliche Ausweichnamen nacheinander probieren
    if 'url_param' in cfg:
        kandidaten = [cfg['url_param']] + list(cfg.get('url_param_alt', []))
    else:
        kandidaten = [None]

    data = None
    for schluessel in kandidaten:
        versuch_params = dict(params)
        if schluessel:
            versuch_params['models'] = schluessel
        try:
            r = requests.get(cfg['url'], params=versuch_params, timeout=15)
            versuch = r.json()
        except Exception:
            continue
        if isinstance(versuch, dict) and 'hourly' in versuch:
            h_test = versuch['hourly']
            # Nur akzeptieren wenn tatsächlich Temperaturwerte enthalten sind
            temps = h_test.get('temperature_2m') or []
            if any(v is not None for v in temps):
                data = versuch
                break
    if data is None:
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
                       'taupunkt': hol('dew_point_2m'),
                       'gefuehlt': hol('apparent_temperature'),
                       'cape': hol('cape'),
                       'uv': hol('uv_index'),
                       'ns_wahr_modell': hol('precipitation_probability'),
                       'strahlung': hol('shortwave_radiation'),
                       't850': hol('temperature_850hPa'),
                       'cin': hol('convective_inhibition'),
                       'lifted_index': hol('lifted_index'),
                       'nullgradgrenze': hol('freezing_level_height'),
                       'sicht': hol('visibility'),
                       'schneefall': hol('snowfall'),
                       'wind80': hol('wind_speed_80m'),
                       'ist_tag': hol('is_day')})
    return df.dropna(subset=['temp'])


def hole_ensemble(lat, lon, modell_name, tage):
    cfg = ENSEMBLE_MODELLE[modell_name]

    # Primären Modellnamen und mögliche Ausweichnamen nacheinander probieren
    kandidaten = [cfg['model_key']] + list(cfg.get('model_key_alt', []))
    data = None
    for schluessel in kandidaten:
        params = {'latitude': lat, 'longitude': lon, 'hourly': ENS_VARIABLEN,
                  'models': schluessel, 'forecast_days': min(tage, 16),
                  'timezone': 'Europe/Vienna'}
        try:
            r = requests.get(ENSEMBLE_API, params=params, timeout=30)
            versuch = r.json()
        except Exception:
            continue
        if isinstance(versuch, dict) and 'hourly' in versuch:
            data = versuch
            break
    if data is None:
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

    # --- Niederschlag: der Median ist hier die falsche Kennzahl ---
    # Niederschlagsverteilungen sind stark rechtsschief und enthalten sehr
    # viele Nullen. Zeigen 20 von 51 Mitgliedern einen Schauer, ist der
    # Median null — die App meldete "trocken", obwohl vier von zehn Läufen
    # Regen bringen. Für eine Arbeitsentscheidung zählt die
    # Überschreitungswahrscheinlichkeit, nicht der mittlere Wert.
    if ns_m is not None:
        ns_med = np.nanmedian(ns_m, axis=0)
        ns_mit = np.nanmean(ns_m, axis=0)
        ns_p75 = np.nanpercentile(ns_m, 75, axis=0)
        ns_p90 = np.nanpercentile(ns_m, 90, axis=0)
        with np.errstate(invalid='ignore'):
            ns_wahr = np.nanmean(ns_m > 0.2, axis=0) * 100   # messbarer Regen
            ns_wahr_1 = np.nanmean(ns_m > 1.0, axis=0) * 100  # spürbar
            ns_wahr_5 = np.nanmean(ns_m > 5.0, axis=0) * 100  # kräftig
    else:
        ns_med = np.zeros(n); ns_mit = np.zeros(n)
        ns_p75 = np.zeros(n); ns_p90 = np.zeros(n)
        ns_wahr = np.zeros(n); ns_wahr_1 = np.zeros(n); ns_wahr_5 = np.zeros(n)

    df = pd.DataFrame({'time': zeiten, 'temp_median': t_med,
                       'temp_p10': t_p10, 'temp_p90': t_p90,
                       'temp_spread': t_p90 - t_p10,
                       'feuchte_median': med(feuchte_m),
                       'wind_median': med(wind_m),
                       'ns_median': ns_med, 'ns_mittel': ns_mit,
                       'ns_p75': ns_p75, 'ns_p90': ns_p90,
                       'ns_wahrscheinlichkeit': ns_wahr,
                       'ns_wahr_1mm': ns_wahr_1, 'ns_wahr_5mm': ns_wahr_5,
                       'n_mitglieder': n_mitgl})
    return df.dropna(subset=['temp_median'])


# ============================================================
# PROGNOSEGÜTE — Abgleich früherer Vorhersagen mit dem Verlauf
# ============================================================
# Bisher gab es keinerlei Rückkopplung an die Wirklichkeit: Nichts prüfte,
# ob eine als „sehr gut“ bewertete Stunde tatsächlich trocken blieb.
# Damit waren alle Schwellenwerte begründete Vermutungen.
#
# Open-Meteo liefert über die Previous-Runs-Schnittstelle, was die Modelle
# vor einem, drei und fünf Tagen für einen Zeitpunkt vorhergesagt haben.
# Verglichen wird gegen den aktuellen Lauf über denselben Zeitraum — das
# ist die beste verfügbare Näherung an den tatsächlichen Verlauf, aber
# ausdrücklich keine Stationsmessung. Systematische Modellfehler, die
# auch die Analyse betreffen, bleiben dadurch unsichtbar. Der Wert liegt
# darin, die zeitliche Entwicklung der Treffsicherheit sichtbar zu machen.
# ------------------------------------------------------------
# INCA — die Referenz, gegen die geprüft wird
# ------------------------------------------------------------
# Bisher wurden alte Vorhersagen gegen den aktuellen Modelllauf verglichen.
# Das ist schwach: Fehler, die auch die Analyse betreffen, bleiben unsichtbar.
#
# GeoSphere Austria betreibt mit INCA ein Analysesystem, das Stations-
# messungen, Radar, Satellit und ein Geländemodell zu einem 1-km-Raster
# verrechnet — stündlich, für ganz Österreich, offen und ohne Schlüssel
# abrufbar (Creative Commons Attribution 4.0).
#
# Das ist der Verifikation gegen eine einzelne Wetterstation überlegen:
# Eine Station steht selten dort, wo der Schlag liegt, und ihre Werte
# gelten streng genommen nur für ihren Standort. INCA liefert für jeden
# Punkt in Österreich eine beobachtungsgestützte Zeitreihe — also auch
# für die Wiese im Waldviertel, drei Kilometer von der nächsten TAWES.
# ============================================================
# LIZENZLAGE DER VERWENDETEN DATENQUELLEN  —  VOR MONETARISIERUNG LESEN
# ============================================================
# Alle eingebundenen Daten sind rechtmäßig nutzbar und in der Fußzeile
# genannt. Eine Einschränkung ist für die geplante Freemium-Variante
# jedoch entscheidend:
#
#   OPEN-METEO
#   Die Daten selbst stehen unter CC BY 4.0 und dürfen auch kommerziell
#   verwendet werden. Die KOSTENLOSE SCHNITTSTELLE (api.open-meteo.com)
#   ist laut Nutzungsbedingungen jedoch ausdrücklich auf nicht-kommerzielle
#   Nutzung beschränkt; als kommerziell gilt dort unter anderem der
#   Betrieb von Websites oder Apps mit Abonnements oder Werbung.
#   Sobald es ein zahlendes Angebot gibt, ist ein Abo mit Zugriff über
#   customer-api.open-meteo.com erforderlich. Die Abfragesyntax bleibt
#   gleich, es ändern sich nur Domain und ein Schlüsselparameter.
#   Zusätzlich gilt für die freie Nutzung: höchstens 10.000 Abrufe je Tag.
#   Ein Aufruf mit vielen Variablen zählt dabei mehrfach — bei rund
#   einem Dutzend Modellabfragen je Auswertung ist die Grenze schneller
#   erreicht, als die Zahl vermuten lässt.
#
#   NOMINATIM (OpenStreetMap)
#   Läuft auf gespendeten Servern. Zulässig ist Nutzung, die direkt durch
#   Eingaben der Anwender ausgelöst wird, bei überschaubarer Nutzerzahl;
#   höchstens eine Anfrage je Sekunde. Anwendungen, deren Kernfunktion
#   Geokodierung ist, müssen einen eigenen Dienst betreiben — das trifft
#   hier nicht zu, die Ortssuche ist Beiwerk. Bei wachsender Nutzerzahl
#   sollte dennoch auf einen kommerziellen Anbieter oder eine eigene
#   Instanz gewechselt werden.
#
#   VENTUSKY
#   Die Einbettungsfunktion wird vom Anbieter als kostenlos und
#   ausdrücklich für Medien und Websites beworben; sie wird von
#   kommerziellen Nachrichtenportalen eingesetzt. Vor dem Start eines
#   zahlenden Angebots empfiehlt sich trotzdem eine kurze schriftliche
#   Rückfrage bei InMeteo.
#
#   GEOSPHERE INCA — CC BY 4.0, Namensnennung erfolgt in der Fußzeile.
#   MONTSERRAT / RALEWAY — SIL Open Font License, kommerziell frei.
#
# Hinweis: Die Schriften werden derzeit von fonts.googleapis.com geladen.
# Dabei wird die IP-Adresse der Anwender an Google übertragen. Deutsche
# Gerichte haben das ohne Einwilligung als Verstoß gegen die DSGVO
# gewertet. Für den produktiven Betrieb sollten die Schriftdateien
# selbst ausgeliefert werden.
#
# Dies ist eine technische Einschätzung, keine Rechtsberatung.
# ============================================================

# Nominatim verlangt eine Kennung, die die Anwendung eindeutig
# identifiziert; voreingestellte Kennungen von HTTP-Bibliotheken werden
# ausdrücklich abgelehnt. Ein Kontaktweg ist nicht Pflicht, aber dringend
# zu empfehlen — fällt eine Anwendung negativ auf, wird sie sonst ohne
# Rückfrage gesperrt.
# TODO: Vor dem öffentlichen Start eigene Kontaktadresse ergänzen.
NOMINATIM_KENNUNG = ('EnsembleWetterAustria/1.0 '
                     '(+https://ensemblewetter-austria.streamlit.app)')

INCA_API = 'https://dataset.api.hub.geosphere.at/v1/timeseries/historical/inca-v1-1h-1km'
INCA_META = INCA_API + '/metadata'
# Abdeckung laut Datensatzbeschreibung
INCA_BBOX = {'lat_min': 45.77, 'lat_max': 49.48,
             'lon_min': 7.10, 'lon_max': 17.74}
# Rückfallnamen, falls die Metadaten nicht erreichbar sind
INCA_STANDARD = {'temp': 'T2M', 'ns': 'RR'}


def im_inca_gebiet(lat, lon):
    """Liegt der Punkt im INCA-Raster?"""
    try:
        return (INCA_BBOX['lat_min'] <= float(lat) <= INCA_BBOX['lat_max']
                and INCA_BBOX['lon_min'] <= float(lon) <= INCA_BBOX['lon_max'])
    except (TypeError, ValueError):
        return False


def inca_parameternamen():
    """
    Liest die Parameterkürzel aus den INCA-Metadaten.

    Die Kürzel werden nicht fest verdrahtet, sondern zur Laufzeit gesucht:
    Ändert GeoSphere die Benennung, findet die Zuordnung sie über Name und
    Einheit trotzdem — statt still falsche Daten zu liefern.
    """
    namen = dict(INCA_STANDARD)
    try:
        r = requests.get(INCA_META, timeout=15)
        meta = r.json()
    except Exception:
        return namen
    params = meta.get('parameters') if isinstance(meta, dict) else None
    if not isinstance(params, list):
        return namen

    def suche(stichworte, einheiten):
        for p in params:
            if not isinstance(p, dict):
                continue
            text = f"{p.get('name', '')} {p.get('long_name', '')} " \
                   f"{p.get('desc', '')}".lower()
            einheit = str(p.get('unit', '')).lower()
            if any(w in text for w in stichworte) and \
                    any(e in einheit for e in einheiten):
                return p.get('name')
        return None

    t = suche(('temperatur', 'temperature', '2m'), ('°c', 'celsius', 'c', 'k'))
    n = suche(('niederschlag', 'precipitation', 'rain'), ('mm', 'kg m-2'))
    if t: namen['temp'] = t
    if n: namen['ns'] = n
    return namen


def hole_inca_zeitreihe(lat, lon, tage_zurueck):
    """
    Stündliche INCA-Analyse für einen Punkt.

    Rückgabe: DataFrame mit time / ist_temp / ist_ns, oder None.
    Ein Fehler hier darf die Auswertung niemals aufhalten.
    """
    if not im_inca_gebiet(lat, lon):
        return None
    namen = inca_parameternamen()
    ende = datetime.now()
    start = ende - timedelta(days=int(tage_zurueck) + 1)
    params = {
        'parameters': f"{namen['temp']},{namen['ns']}",
        'start': start.strftime('%Y-%m-%dT%H:00'),
        'end': ende.strftime('%Y-%m-%dT%H:00'),
        'lat_lon': f'{float(lat):.5f},{float(lon):.5f}',
        'output_format': 'geojson',
    }
    try:
        r = requests.get(INCA_API, params=params, timeout=25)
        if r.status_code != 200:
            return None
        d = r.json()
    except Exception:
        return None

    # Die Antwort ist GeoJSON. Die Zeitachse liegt je nach Endpunkt entweder
    # oben oder innerhalb des Features; beide Varianten werden akzeptiert,
    # damit eine Formatänderung nicht sofort den ganzen Abschnitt lahmlegt.
    def _finde_zeiten(obj):
        if isinstance(obj, dict):
            for schluessel in ('timestamps', 'time', 'times'):
                if isinstance(obj.get(schluessel), list) and obj[schluessel]:
                    return obj[schluessel]
            for wert in obj.values():
                gefunden = _finde_zeiten(wert)
                if gefunden:
                    return gefunden
        elif isinstance(obj, list):
            for wert in obj[:3]:
                gefunden = _finde_zeiten(wert)
                if gefunden:
                    return gefunden
        return None

    try:
        roh_zeiten = _finde_zeiten(d)
        if not roh_zeiten:
            return None
        zeiten = pd.to_datetime(pd.Series(roh_zeiten), errors='coerce',
                                utc=True)
        werte = d['features'][0]['properties']['parameters']
        if not isinstance(werte, dict):
            return None
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    def hol(schluessel):
        eintrag = werte.get(schluessel)
        if eintrag is None:
            # Groß-/Kleinschreibung der Kürzel kann abweichen
            for k, v in werte.items():
                if str(k).lower() == str(schluessel).lower():
                    eintrag = v
                    break
        if isinstance(eintrag, dict):
            eintrag = eintrag.get('data')
        if not isinstance(eintrag, list):
            return None
        return pd.to_numeric(pd.Series(eintrag), errors='coerce')

    t = hol(namen['temp'])
    n = hol(namen['ns'])
    if t is None or len(t) != len(zeiten):
        return None

    # Zeitstempel kommen in UTC, die Vorhersagedaten in Ortszeit —
    # ohne Umrechnung wäre der Vergleich um ein bis zwei Stunden versetzt
    # und die Fehler fielen künstlich zu groß aus.
    try:
        zeiten = zeiten.dt.tz_convert('Europe/Vienna').dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass

    df = pd.DataFrame({
        'time': zeiten.values, 'ist_temp': t.values,
        'ist_ns': (n.values if n is not None and len(n) == len(zeiten)
                   else np.nan)})
    return df.dropna(subset=['ist_temp'])


PREVIOUS_RUNS_API = 'https://previous-runs-api.open-meteo.com/v1/forecast'
VERIFIKATION_TAGE = 7


def hole_prognoseguete(lat, lon, tage_zurueck=VERIFIKATION_TAGE):
    """
    Holt frühere Modellläufe und vergleicht sie mit dem aktuellen Lauf.

    Rückgabe: dict mit Kennzahlen je Vorlaufzeit oder None, wenn die
    Schnittstelle nicht erreichbar ist. Ein Fehler hier darf die App
    niemals blockieren — die Auswertung ist eine Zusatzinformation.
    """
    vorlauf = (1, 3, 5)
    felder = ['temperature_2m', 'precipitation']
    stunden = list(felder)
    for f in felder:
        for v in vorlauf:
            stunden.append(f'{f}_previous_day{v}')

    params = {'latitude': lat, 'longitude': lon,
              'hourly': ','.join(stunden),
              'past_days': int(tage_zurueck), 'forecast_days': 1,
              'timezone': 'Europe/Vienna'}
    try:
        r = requests.get(PREVIOUS_RUNS_API, params=params, timeout=20)
        data = r.json()
    except Exception:
        return None
    if not isinstance(data, dict) or 'hourly' not in data:
        return None

    h = data['hourly']
    if 'time' not in h:
        return None
    try:
        zeiten = pd.to_datetime(h['time'])
    except Exception:
        return None

    def spalte(key):
        v = h.get(key)
        if not v:
            return None
        return pd.to_numeric(pd.Series(v), errors='coerce')

    ist_temp = spalte('temperature_2m')
    ist_ns = spalte('precipitation')
    if ist_temp is None:
        return None

    jetzt = pd.Timestamp(datetime.now())
    zeit_serie = pd.Series(zeiten)
    vergangen = zeit_serie <= jetzt
    if vergangen.sum() < 24:
        return None

    # --- Referenz: möglichst die INCA-Analyse, sonst der eigene Modelllauf ---
    quelle = 'modell'
    inca = hole_inca_zeitreihe(lat, lon, tage_zurueck)
    if inca is not None and len(inca) >= 24:
        _abgleich = pd.DataFrame({'time': zeit_serie}).merge(
            inca, on='time', how='left')
        _treffer = _abgleich['ist_temp'].notna() & vergangen
        # Nur übernehmen, wenn INCA den Zeitraum wirklich abdeckt
        if _treffer.sum() >= 24:
            ist_temp = _abgleich['ist_temp']
            if _abgleich['ist_ns'].notna().sum() >= 24:
                ist_ns = _abgleich['ist_ns']
            vergangen = _treffer
            quelle = 'inca'

    ergebnis = {'zeitraum_tage': int(tage_zurueck), 'vorlauf': {},
                'quelle': quelle}
    for v in vorlauf:
        eintrag = {}
        # --- Temperatur: mittlerer absoluter Fehler ---
        pt = spalte(f'temperature_2m_previous_day{v}')
        if pt is not None:
            diff = (pt - ist_temp)[vergangen].dropna()
            if len(diff) >= 24:
                eintrag['temp_mae'] = float(diff.abs().mean())
                eintrag['temp_bias'] = float(diff.mean())
                eintrag['temp_n'] = int(len(diff))

        # --- Niederschlag: Trefferquote und Fehlalarme ---
        # Bewertet wird die Ja/Nein-Aussage „messbarer Regen“,
        # weil genau diese in die Eignungsbewertung eingeht.
        pn = spalte(f'precipitation_previous_day{v}')
        if pn is not None and ist_ns is not None:
            paar = pd.DataFrame({'prog': pn, 'ist': ist_ns})[vergangen].dropna()
            if len(paar) >= 24:
                p_ja = paar['prog'] > 0.2
                i_ja = paar['ist'] > 0.2
                treffer = int((p_ja & i_ja).sum())
                verpasst = int((~p_ja & i_ja).sum())
                fehlalarm = int((p_ja & ~i_ja).sum())
                korrekt_trocken = int((~p_ja & ~i_ja).sum())
                eintrag['ns_n'] = int(len(paar))
                eintrag['ns_treffer'] = treffer
                eintrag['ns_verpasst'] = verpasst
                eintrag['ns_fehlalarm'] = fehlalarm
                eintrag['ns_genauigkeit'] = (
                    (treffer + korrekt_trocken) / len(paar) * 100)
                if (treffer + verpasst) > 0:
                    eintrag['ns_entdeckung'] = treffer / (treffer + verpasst) * 100
                if (treffer + fehlalarm) > 0:
                    eintrag['ns_praezision'] = treffer / (treffer + fehlalarm) * 100
        if eintrag:
            ergebnis['vorlauf'][v] = eintrag

    return ergebnis if ergebnis['vorlauf'] else None


SOIL_LAYERS = [
    ('soil_moisture_0_to_1cm',   '0–1 cm',   'Oberfläche'),
    ('soil_moisture_1_to_3cm',   '1–3 cm',   'Saatbett'),
    ('soil_moisture_3_to_9cm',   '3–9 cm',   'Wurzelraum oben'),
    ('soil_moisture_9_to_27cm',  '9–27 cm',  'Wurzelraum tief'),
]


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

def berechne_bodenindex(boden, bodenart=BODENART_STANDARD):
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

    kenn = BODENARTEN.get(bodenart, BODENARTEN[BODENART_STANDARD])
    drainage_mm = kenn['drainage']
    # Schwere Böden gelten schon bei geringerer Bilanz als kritisch
    f = kenn['bilanz_faktor']
    s_feucht = BILANZ_FEUCHT * f
    s_nass = BILANZ_NASS * f
    s_extrem = BILANZ_EXTREM * f

    # --- Wasserbilanz über alle Tage (Vergangenheit + Zukunft) ---
    # Reihenfolge je Tag: zuerst Regen aufnehmen, dann Verlust abziehen.
    # Drainage wirkt anteilig auf die Bilanz (viel Wasser drainiert absolut
    # mehr, trocknet aber relativ langsamer aus als eine feste Menge) plus
    # Verdunstung. So bleibt ein großes Regenereignis realistisch länger
    # spürbar, während nach ~10 Trockentagen wieder abgetrocknet ist.
    # Zusätzlich zur reinen mm-Bilanz wird ein Sättigungsgrad geführt:
    # dieselbe Überschussmenge bedeutet auf Sand und auf Moor völlig
    # Verschiedenes. Die Drainage hängt nun vom Füllstand ab — ein
    # gesättigter Boden gibt schnell ab, ein halb gefüllter kaum noch.
    nutzbare_fk = kenn.get('nutzbare_fk', 60.0)
    abtrocknung = kenn.get('abtrocknung', 1.0)

    balance = 0.0
    per_datum_balance = {}
    per_datum_saettigung = {}
    for _, row in df.iterrows():
        balance += row['precip']                      # Regen aufnehmen
        fuellgrad = min(balance / nutzbare_fk, 1.6) if nutzbare_fk else 0.0
        # Sickerung setzt erst oberhalb der Feldkapazität kräftig ein
        drainage = drainage_mm * (0.25 + 0.95 * min(fuellgrad, 1.2))
        # Verdunstung wird bei trockenem Boden gebremst (Wasserstress)
        et_wirksam = row['et0'] * abtrocknung * min(1.0, 0.35 + fuellgrad)
        balance = max(0.0, balance - (et_wirksam + drainage))
        per_datum_balance[row['datum']] = balance
        per_datum_saettigung[row['datum']] = (
            balance / nutzbare_fk if nutzbare_fk else 0.0)

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
        if b >= s_extrem: return 3
        if b >= s_nass: return 2
        if b >= s_feucht: return 1
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
    # "ref" wurde bisher nur im else-Zweig gesetzt und danach weiter
    # verwendet — lag "heute" exakt in den Daten, stürzte die Funktion mit
    # UnboundLocalError ab. Der Bezugstag wird jetzt in beiden Fällen gesetzt.
    if heute in per_datum_level:
        ref = heute
    else:
        vergangene_daten = [d for d in per_datum_level if d <= heute]
        ref = (max(vergangene_daten) if vergangene_daten
               else min(per_datum_level))
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

    saettigung_heute = per_datum_saettigung.get(ref, 0.0)

    metriken = {
        'regen_3t': regen_3t, 'regen_7t': regen_7t, 'regen_10t': regen_10t,
        'tage_seit_regen': tage_seit_regen,
        'balance_heute': balance_heute,
        'saettigung': saettigung_heute,
        'nutzbare_fk': nutzbare_fk,
        'zustand_level': heute_level[0], 'zustand_text': heute_level[1],
        'soil_moisture': sm, 'soil_status': sm_status,
    }

    return {'per_datum_level': per_datum_level, 'per_datum_dry': per_datum_dry,
            'per_datum_saettigung': per_datum_saettigung,
            'metriken': metriken, 'bodenart': bodenart,
            'bodenart_hinweis': kenn['hinweis']}


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

def gewitterrisiko(cape, cin, lifted_index, stunde=None, ns_wahr=None):
    """
    Gewitterrisiko aus dem Zusammenspiel mehrerer Größen.

    CAPE allein taugt nicht als Maß. 2000 J/kg unter einer kräftigen
    Sperrschicht (CIN) bedeuten meist einen ruhigen Tag; 800 J/kg ohne
    Deckel und mit labiler Schichtung reichen dagegen für kräftige
    Zellen. Zusätzlich wird die Tageszeit berücksichtigt: Für Bergsport
    ist nicht die Tagesmenge an Energie entscheidend, sondern wann die
    Auslösung wahrscheinlich wird.

    Rückgabe: {'stufe': 0–3, 'text': Kurzbegründung}
    """
    def _z(v, standard=0.0):
        try:
            f = float(v)
            return standard if np.isnan(f) else f
        except (TypeError, ValueError):
            return standard

    c = _z(cape)
    hemmung = abs(_z(cin))          # CIN wird negativ geliefert
    li = _z(lifted_index, 6.0)      # positiv = stabil

    if c < 300:
        return {'stufe': 0, 'text': 'keine nennenswerte Labilität'}

    # Grundpunktzahl aus der Energie
    punkte = 0.0
    if c >= 2500:   punkte += 3.0
    elif c >= 1500: punkte += 2.2
    elif c >= 800:  punkte += 1.4
    else:           punkte += 0.6

    # Labilität verstärkt, Stabilität dämpft
    if li <= -6:    punkte += 1.0
    elif li <= -3:  punkte += 0.6
    elif li <= -1:  punkte += 0.25
    elif li >= 2:   punkte -= 0.8

    # Sperrschicht: ab etwa 100 J/kg wird die Auslösung unwahrscheinlich
    if hemmung >= 200:   punkte -= 1.8
    elif hemmung >= 100: punkte -= 1.0
    elif hemmung >= 50:  punkte -= 0.4

    # Tagesgang: Auslösung überwiegend am Nachmittag und frühen Abend
    if stunde is not None:
        try:
            hh = int(stunde)
            if 12 <= hh <= 20:   punkte += 0.5
            elif 21 <= hh <= 23: punkte += 0.15
            elif 0 <= hh <= 8:   punkte -= 0.7
        except (TypeError, ValueError):
            pass

    # Wenn die Modelle ohnehin Niederschlag zeigen, ist die Auslösung
    # bereits eingepreist
    if ns_wahr is not None and _z(ns_wahr) >= 50:
        punkte += 0.4

    teile = [f'CAPE {c:.0f}']
    if hemmung >= 50:
        teile.append(f'CIN {hemmung:.0f}')
    if li <= 0:
        teile.append(f'LI {li:.0f}')
    text = ', '.join(teile)

    if punkte >= 3.4:  stufe = 3
    elif punkte >= 2.3: stufe = 2
    elif punkte >= 1.4: stufe = 1
    else:               stufe = 0

    if stufe == 0 and hemmung >= 100 and c >= 800:
        text += ' — Sperrschicht verhindert Auslösung'
    return {'stufe': stufe, 'text': text}


def bewerte_werte(temp, feuchte, wind, ns, af):
    """
    Bewertet Einzelwerte gegen ein Parameter-Dict af.
    Rückgabe: (stufe 0–4, begruendungen)

    Grundgedanke: Für jeden Parameter wird die Überschreitung des Grenzwerts
    auf einen Prozentwert normiert. Aus der Zahl und der Höhe der
    Überschreitungen ergibt sich die Eignungsstufe. Dadurch wird ein einzelner
    kleiner Ausreißer anders gewichtet als mehrere deutliche Verstöße.
    """
    hinweis = af.get('_hinweis', '')

    # --- Sonderfall: Autofahren ---
    if hinweis == 'autofahren':
        t = float(temp) if temp is not None else 10.0
        w = float(wind) if wind is not None else 0.0
        n = float(ns) if ns is not None else 0.0
        gruende = []
        stufe = 4

        if n >= 8.0 or w >= 90:
            stufe = 0
            if n >= 8.0: gruende.append(f'Starkregen ({n:.1f} mm/h)')
            if w >= 90: gruende.append(f'Sturm ({w:.0f} km/h)')
        elif t <= 0 and n > 0.3:
            stufe = 0; gruende.append(f'Glatteisgefahr ({t:.0f}°C, Niederschlag)')
        elif t <= 2 and n > 0.1:
            stufe = 1; gruende.append(f'Schneeglätte möglich ({t:.0f}°C)')
        elif n >= 4.0 or w >= 70:
            stufe = 1
            if n >= 4.0: gruende.append(f'starker Regen ({n:.1f} mm/h)')
            if w >= 70: gruende.append(f'starker Sturm ({w:.0f} km/h)')
        elif n >= 1.5 or w >= 55:
            stufe = 2
            if n >= 1.5: gruende.append(f'Regen ({n:.1f} mm/h)')
            if w >= 55: gruende.append(f'Sturmböen ({w:.0f} km/h)')
        elif n >= 0.3 or w >= 40:
            stufe = 3
            if n >= 0.3: gruende.append(f'leichter Regen ({n:.1f} mm/h)')
            if w >= 40: gruende.append(f'Wind ({w:.0f} km/h)')

        return stufe, gruende if gruende else ['Straßenverhältnisse gut']

    # --- Sonderfall: Motorradfahren ---
    if hinweis == 'motorrad':
        t = float(temp) if temp is not None else 15.0
        w = float(wind) if wind is not None else 0.0
        n = float(ns) if ns is not None else 0.0
        gruende = []; score = 0

        if n > 0:
            anteil = n / 1.5
            if anteil >= 0.5: score += 3; gruende.append(f'nasse Fahrbahn ({n:.1f} mm/h)')
            elif anteil >= 0.2: score += 2; gruende.append(f'leichter Regen ({n:.1f} mm/h)')
            else: score += 1; gruende.append(f'Niesel ({n:.1f} mm/h)')
        if t < 3:
            score += 3; gruende.append(f'zu kalt ({t:.0f}°C, Reifenhaftung)')
        elif t < 5:
            score += 2; gruende.append(f'kalt ({t:.0f}°C)')
        elif t < 8:
            score += 1; gruende.append(f'kühl ({t:.0f}°C)')
        if w > af.get('wind_max', 55):
            anteil = (w - af['wind_max']) / af['wind_max']
            if anteil >= 0.5: score += 3; gruende.append(f'Sturm ({w:.0f} km/h)')
            elif anteil >= 0.25: score += 2; gruende.append(f'starker Wind ({w:.0f} km/h)')
            else: score += 1; gruende.append(f'Wind ({w:.0f} km/h)')

        if score == 0: return 4, ['optimale Bedingungen']
        if score <= 1: return 3, gruende
        if score <= 3: return 2, gruende
        if score <= 5: return 1, gruende
        return 0, gruende

    # --- Sonderfall: Rennradfahren ---
    if hinweis == 'rennrad':
        t = float(temp) if temp is not None else 18.0
        w = float(wind) if wind is not None else 0.0
        n = float(ns) if ns is not None else 0.0
        gruende = []; score = 0

        if n > 0:
            anteil = n / 1.0
            if anteil >= 0.5: score += 4; gruende.append(f'Regen ({n:.1f} mm/h)')
            elif anteil >= 0.1: score += 2; gruende.append(f'leichter Regen ({n:.1f} mm/h)')
            else: score += 1; gruende.append(f'Niesel ({n:.1f} mm/h)')
        if t < 3:
            score += 4; gruende.append(f'zu kalt ({t:.0f}°C)')
        elif t < 8:
            score += 2; gruende.append(f'kalt ({t:.0f}°C)')
        elif t < 12:
            score += 1; gruende.append(f'kühl ({t:.0f}°C)')
        if w > af.get('wind_max', 35):
            anteil = (w - af['wind_max']) / af['wind_max']
            if anteil >= 0.7: score += 4; gruende.append(f'Sturm ({w:.0f} km/h)')
            elif anteil >= 0.4: score += 3; gruende.append(f'starker Wind ({w:.0f} km/h)')
            elif anteil >= 0.15: score += 2; gruende.append(f'Wind ({w:.0f} km/h)')
            else: score += 1; gruende.append(f'leichter Wind ({w:.0f} km/h)')

        if score == 0: return 4, ['ideale Bedingungen']
        if score <= 1: return 3, gruende
        if score <= 3: return 2, gruende
        if score <= 5: return 1, gruende
        return 0, gruende

    if af.get('modus') == 'frost':
        if temp is None:
            return 0, ['keine Daten']
        t = float(temp)
        frost_s = af.get('temp_frost', 0)
        warn_s = af.get('temp_warn', 3)

        # Bei klarem Himmel und Windstille kühlt die bodennahe Luft durch
        # nächtliche Ausstrahlung deutlich stärker aus als die Messhöhe
        # in zwei Metern. Der Abschlag bildet diesen Effekt ab.
        wolken = float(af.get('_wolken_gesamt', 50) or 50)
        w = float(wind) if wind is not None else 10.0
        ausstrahlung = 0.0
        if wolken < 40 and w < 8:
            ausstrahlung = 3.5
        elif wolken < 60 and w < 12:
            ausstrahlung = 2.0
        elif wolken < 80:
            ausstrahlung = 1.0
        t_boden = t - ausstrahlung

        zusatz = (f', bodennah etwa {t_boden:.0f} °C'
                  if ausstrahlung >= 1.0 else '')

        if t_boden <= frost_s:
            return 0, [f'Frost ({t:.1f} °C{zusatz})']
        if t_boden <= warn_s:
            return 1, [f'hohe Frostgefahr ({t:.1f} °C{zusatz})']
        if t <= warn_s + 2:
            return 2, [f'Frostgefahr nicht ausgeschlossen ({t:.1f} °C)']
        if t <= warn_s + 5:
            return 3, [f'gering erhöhtes Risiko ({t:.1f} °C)']
        return 4, [f'frostfrei ({t:.1f} °C)']

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

    # ============================================================
    # NIEDERSCHLAG — bewertet über Wahrscheinlichkeit, nicht über Median
    # ============================================================
    # Der Median einer Niederschlagsverteilung ist als Entscheidungsgröße
    # untauglich: Zeigen 20 von 51 Ensemble-Mitgliedern einen Schauer,
    # liegt der Median bei null. Maßgeblich sind daher
    #   • P(> 0,2 mm)  — Regenrisiko überhaupt
    #   • P(> 1 mm)    — Menge, die ein Schwad wiederbefeuchtet
    #   • das 75. Perzentil als realistische Mengenannahme
    ns_wert = float(ns) if ns is not None and not np.isnan(float(ns)) else 0.0
    ns_p75 = float(af.get('_ns_p75') or ns_wert)
    ns_wahr = float(af.get('_ns_wahr') or 0.0)
    ns_wahr_1 = float(af.get('_ns_wahr_1mm') or 0.0)
    ns_wahr_5 = float(af.get('_ns_wahr_5mm') or 0.0)
    # Menge, mit der gerechnet wird: das obere Ende der Verteilung
    ns_ansatz = max(ns_wert, ns_p75)

    if 'precip_max' in af:
        grenze_n = float(af['precip_max'])
        bezug_n = max(grenze_n, BEZUG_NS_MIN)

        def ns_text(w, a):
            if w >= 2.5:
                return f'Starkregen (bis {w:.1f} mm/h)'
            if w >= 0.5:
                return f'Regen (bis {w:.1f} mm/h)'
            return f'leichter Regen (bis {w:.1f} mm/h)'

        pruefe(ns_ansatz, grenze_n, bezug_n, 'max', ns_text)

        if grenze_n == 0.0:
            # --- Null-Toleranz: die Wahrscheinlichkeit ist die Kernzahl ---
            # Bei Heu und Silage entscheidet nicht, wie viel es regnet,
            # sondern ob. Schon ein Drittel nasser Läufe macht das
            # Zeitfenster unbrauchbar für eine Arbeitsplanung.
            if ns_wahr >= 60 or ns_wahr_1 >= 35:
                ueber.append((0.75, f'Regen sehr wahrscheinlich '
                                    f'({ns_wahr:.0f} % Risiko)'))
            elif ns_wahr >= 40 or ns_wahr_1 >= 20:
                ueber.append((0.48, f'hohes Regenrisiko ({ns_wahr:.0f} %)'))
            elif ns_wahr >= 25:
                ueber.append((0.30, f'Regenrisiko {ns_wahr:.0f} %'))
            elif ns_wahr >= 12:
                ueber.append((0.15, f'Schauerrisiko {ns_wahr:.0f} %'))
        else:
            # --- Toleranter Fall: erst deutliches Risiko zählt ---
            if ns_wahr_1 >= 55:
                ueber.append((0.42, f'anhaltender Regen wahrscheinlich '
                                    f'({ns_wahr_1:.0f} %)'))
            elif ns_wahr >= 65:
                ueber.append((0.26, f'hohes Regenrisiko ({ns_wahr:.0f} %)'))
            elif ns_wahr >= 45:
                ueber.append((0.14, f'Regenrisiko {ns_wahr:.0f} %'))

        # Starkregen ist unabhängig vom Grenzwert ein Ausschlusskriterium
        if ns_wahr_5 >= 25:
            ueber.append((0.55, f'Starkregenrisiko ({ns_wahr_5:.0f} %)'))

    # ============================================================
    # TROCKNUNGSLEISTUNG — für Halmfutter die eigentliche Zielgröße
    # ============================================================
    # Die relative Feuchte allein sagt wenig aus: 70 % bei 12 °C und
    # 70 % bei 28 °C bedeuten völlig verschiedene Trocknungsraten.
    # Maßgeblich ist das Sättigungsdefizit, verstärkt durch Wind und
    # Einstrahlung.
    # Nur bei Tageslicht bewerten: Dass nachts nichts trocknet, ist keine
    # Eigenschaft des Wetters, sondern der Tageszeit. Würde man das als
    # Mangel werten, wäre jedes mehrtägige Heufenster durch die Nächte
    # zerschnitten — und die Fenstersuche fände nie einen brauchbaren Block.
    _strahl = af.get('_strahlung')
    _ist_hell = (_strahl is not None and not pd.isna(_strahl)
                 and float(_strahl) > 15.0)
    if (af.get('trocknung_relevant') and _ist_hell
            and temp is not None and feuchte is not None):
        try:
            _rate = float(trocknungsrate(
                pd.Series([float(temp)]), pd.Series([float(feuchte)]),
                pd.Series([float(wind) if wind is not None else 0.0]),
                (pd.Series([float(af['_strahlung'])])
                 if af.get('_strahlung') is not None else None)).iloc[0])
        except Exception:
            _rate = None
        if _rate is not None:
            _mindest = float(af.get('trocknung_min', 0.45))
            if _rate < _mindest * 0.35:
                ueber.append((0.55, f'kaum Trocknung (Index {_rate:.2f})'))
            elif _rate < _mindest * 0.65:
                ueber.append((0.33, f'schwache Trocknung (Index {_rate:.2f})'))
            elif _rate < _mindest:
                ueber.append((0.16, f'verhaltene Trocknung (Index {_rate:.2f})'))

    # --- Alpine Bewölkungskorrektur ---
    # Wenn Standort ≥ 1500 m und Anwendungsfall alpiner Art:
    # Tiefe/mittlere Bewölkung kann Sicht und Orientierung nehmen.
    # --- Allgemeine Bewölkungsregel für Aktivitäten im Freien ---
    # Ein weitgehend bedeckter Himmel mindert den Wert einer Unternehmung
    # spürbar: weniger Licht, keine Aussicht, kühler und feuchter.
    if af.get('outdoor'):
        w_ges = float(af.get('_wolken_gesamt', 0) or 0)
        if af.get('sicht_kritisch'):
            # Bei Sternbeobachtung und Landschaftsfotografie entscheidet
            # die freie Sicht über den Erfolg; Bewölkung wiegt deshalb
            # deutlich schwerer als bei anderen Unternehmungen.
            if w_ges >= 80:
                ueber.append((0.75, f'Himmel bedeckt ({w_ges:.0f} %) — '
                                    f'keine freie Sicht'))
            elif w_ges >= 55:
                ueber.append((0.42, f'stark bewölkt ({w_ges:.0f} %)'))
            elif w_ges >= 35:
                ueber.append((0.22, f'wechselnd bewölkt ({w_ges:.0f} %)'))
            elif w_ges >= 20:
                ueber.append((0.10, f'leicht bewölkt ({w_ges:.0f} %)'))
        elif w_ges >= 90:
            ueber.append((0.30, f'Himmel bedeckt ({w_ges:.0f} %)'))
        elif w_ges >= 70:
            ueber.append((0.16, f'stark bewölkt ({w_ges:.0f} %)'))

        # --- Gewitterpotenzial ---
        # CAPE allein ist ein schwacher Prädiktor: Energie in der
        # Atmosphäre nützt nichts, solange eine Sperrschicht die
        # Auslösung verhindert. Erst das Zusammenspiel entscheidet:
        #   • CAPE  — verfügbare Energie
        #   • CIN   — Sperrschicht, die erst überwunden werden muss
        #   • LI    — Labilität (negativ = labil)
        # Zusätzlich zählt die Auslösewahrscheinlichkeit: hohe CAPE-Werte
        # in der Frühe führen selten zu Gewittern, dieselben Werte am
        # Nachmittag sehr wohl.
        gewitter = gewitterrisiko(af.get('_cape'), af.get('_cin'),
                                  af.get('_lifted_index'),
                                  af.get('_stunde'), af.get('_ns_wahr'))
        if gewitter['stufe'] >= 3:
            ueber.append((0.65, f'hohes Gewitterrisiko ({gewitter["text"]})'))
        elif gewitter['stufe'] == 2:
            ueber.append((0.40, f'Gewitterrisiko ({gewitter["text"]})'))
        elif gewitter['stufe'] == 1:
            ueber.append((0.18, f'erhöhte Gewitterneigung ({gewitter["text"]})'))

        # --- Gefühlte Temperatur ---
        # Wind und Feuchte verschieben die Belastung deutlich gegenüber
        # der reinen Lufttemperatur.
        gef = af.get('_gefuehlt')
        if gef is not None and not (isinstance(gef, float) and np.isnan(gef)):
            gef = float(gef)
            if 'temp_min' in af and gef < af['temp_min'] - 3:
                ueber.append((min(0.45, (af['temp_min'] - 3 - gef) / 10.0),
                              f'gefühlt deutlich kälter ({gef:.0f} °C)'))
            elif 'temp_max' in af and gef > af['temp_max'] + 3:
                ueber.append((min(0.45, (gef - af['temp_max'] - 3) / 10.0),
                              f'gefühlt deutlich wärmer ({gef:.0f} °C)'))

    # ============================================================
    # ALPINE ZUSATZKRITERIEN
    # ============================================================
    hoehe_std = float(af.get('_standorthoehe', 0) or 0)

    # --- Sichtweite ---
    # Für Tourengeher ist Sicht ein Sicherheitsfaktor, nicht Komfort.
    # Unter 200 m wird Orientierung im Gelände zum Problem.
    if af.get('sicht_relevant'):
        _si = af.get('_sicht')
        if _si is not None and not (isinstance(_si, float) and np.isnan(_si)):
            _si = float(_si)
            if _si < 200:
                ueber.append((0.70, f'Sichtweite unter 200 m'))
            elif _si < 1000:
                ueber.append((0.40, f'eingeschränkte Sicht ({_si:.0f} m)'))
            elif _si < 4000:
                ueber.append((0.15, f'diesig ({_si / 1000:.1f} km Sicht)'))

    # --- Nullgradgrenze ---
    # Liegt sie knapp über dem Standort, wechselt Niederschlag zwischen
    # Regen und Schnee, und Nässe bei Temperaturen um null ist die
    # gefährlichste Kombination für Auskühlung.
    if af.get('nullgrad_relevant') and hoehe_std > 0:
        _ng = af.get('_nullgradgrenze')
        if _ng is not None and not (isinstance(_ng, float) and np.isnan(_ng)):
            _ng = float(_ng)
            _abstand = _ng - hoehe_std
            _nass = (af.get('_ns_wahr') or 0) >= 35
            if _nass and -300 <= _abstand <= 400:
                ueber.append((0.45,
                              f'Nullgradgrenze bei {_ng:.0f} m — '
                              f'Wechsel Regen/Schnee am Standort'))
            elif _abstand < -600:
                ueber.append((0.12, f'Nullgradgrenze bei {_ng:.0f} m, '
                                    f'durchgehend Frost'))

    if (af.get('alpin_bewölkung')
            and hoehe_std >= ALPIN_HOEHE_SCHWELLE):
        w_tief = float(af.get('_wolken_tief', 0) or 0)
        w_mittel = float(af.get('_wolken_mittel', 0) or 0)
        # Untergrenze mittlerer Wolken: ca. 2000–3000 m
        # Tief = unter ~2000 m → kritisch für alles ab 1500 m
        if w_tief >= 80:
            ueber.append((0.55, f'Gipfel in Wolken (tiefe Bewölkung {w_tief:.0f} %)'))
        elif w_tief >= 50:
            ueber.append((0.30, f'eingeschränkte Sicht (Bewölkung tief {w_tief:.0f} %)'))
        elif w_tief >= 30:
            ueber.append((0.14, f'Bewölkung tief {w_tief:.0f} %'))
        # Mittlere Wolken: problematisch ab ~2000 m Standort
        if af.get('_standorthoehe', 0) >= 2000:
            if w_mittel >= 75:
                ueber.append((0.40, f'Gipfel in mittlerer Bewölkung ({w_mittel:.0f} %)'))
            elif w_mittel >= 50:
                ueber.append((0.20, f'mittlere Bewölkung {w_mittel:.0f} %'))

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


def bewerte_ensemble_df(ens_df, af, wolken_df=None):
    """
    Bewertet die aggregierten Ensemble-Werte.
    wolken_df: optionales DataFrame mit Bewölkungsdaten aus den Hauptläufen,
    da die Ensemble-Systeme keine Schichtbewölkung liefern.
    """
    af_basis = dict(af)

    # Bewölkung je Zeitpunkt aus den Hauptläufen zuordnen
    wolken_map = {}
    if wolken_df is not None and not wolken_df.empty:
        for _, wz in wolken_df.iterrows():
            wolken_map[wz['time']] = {
                'gesamt': wz.get('wolken'),
                'tief': wz.get('wolken_tief'),
                'mittel': wz.get('wolken_mittel'),
                'cape': wz.get('cape'),
                'gefuehlt': wz.get('gefuehlt'),
                'strahlung': wz.get('strahlung'),
                'cin': wz.get('cin'),
                'lifted_index': wz.get('lifted_index'),
                'nullgradgrenze': wz.get('nullgradgrenze'),
                'sicht': wz.get('sicht'),
            }

    def _bew_ens(r):
        a = dict(af_basis)
        a['_ns_wahr'] = float(r.get('ns_wahrscheinlichkeit', 0) or 0)
        a['_ns_wahr_1mm'] = float(r.get('ns_wahr_1mm', 0) or 0)
        a['_ns_wahr_5mm'] = float(r.get('ns_wahr_5mm', 0) or 0)
        a['_ns_p75'] = float(r.get('ns_p75', 0) or 0)
        w = wolken_map.get(r['time'])
        if w:
            a['_wolken_gesamt'] = float(w['gesamt'] or 0)
            a['_wolken_tief'] = float(w['tief'] or 0)
            a['_wolken_mittel'] = float(w['mittel'] or 0)
            a['_cape'] = float(w.get('cape') or 0)
            a['_gefuehlt'] = w.get('gefuehlt')
            a['_strahlung'] = w.get('strahlung')
            a['_cin'] = w.get('cin')
            a['_lifted_index'] = w.get('lifted_index')
            a['_nullgradgrenze'] = w.get('nullgradgrenze')
            a['_sicht'] = w.get('sicht')
        a['_stunde'] = pd.Timestamp(r['time']).hour
        return bewerte_werte(r['temp_median'], r['feuchte_median'],
                             r['wind_median'], r['ns_median'], a)
    erg = ens_df.apply(_bew_ens, axis=1)
    ens_df = ens_df.copy()
    ens_df['ampel_int'] = [e[0] for e in erg]
    return ens_df


# ============================================================
# LADEN
# ============================================================

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


def lade_alle_daten(lat, lon, af, tage, fortschritt=None, standorthoehe=None,
                    bodenart=BODENART_STANDARD):
    schritte = len(HAUPTLAUFE) + len(ENSEMBLE_MODELLE) + 2
    schritt = 0
    haupt_daten, ensemble_daten = {}, {}

    # Höhendifferenz Standort ↔ Modellgitter bestimmen
    modellhoehe = hole_modellhoehe(lat, lon)
    delta_h = None
    if standorthoehe is not None and modellhoehe is not None:
        delta_h = float(standorthoehe) - float(modellhoehe)

    # Standorthöhe ins af-Dict schreiben damit Bewertung darauf zugreifen kann
    if standorthoehe is not None:
        af['_standorthoehe'] = float(standorthoehe)

    # Erst alle Hauptläufe unkorrigiert laden: Der tatsächliche vertikale
    # Temperaturgradient lässt sich erst bestimmen, wenn die 850-hPa-Werte
    # mehrerer Modelle vorliegen. Die Höhenkorrektur folgt danach.
    roh_haupt = {}
    for name in HAUPTLAUFE:
        schritt += 1
        if fortschritt: fortschritt.progress(schritt/schritte, text=f"Lade {name} …")
        df = hole_hauptlauf(lat, lon, name, tage)
        if df is not None and not df.empty:
            roh_haupt[name] = df

    gradient_df = bestimme_gradient(roh_haupt, modellhoehe)
    inversion_anteil = 0.0
    if gradient_df is not None and len(gradient_df):
        inversion_anteil = float(gradient_df['inversion'].mean())

    for name, df in roh_haupt.items():
        haupt_daten[name] = hoehenkorrektur(df, delta_h,
                                            gradient_df=gradient_df)

    # Bewölkungsdaten aus einem Hauptlauf als Referenz für die Ensembles
    _wolken_ref = None
    for _hd in haupt_daten.values():
        if 'wolken' in _hd.columns and _hd['wolken'].notna().any():
            _spalten_ref = ['time', 'wolken', 'wolken_tief', 'wolken_mittel']
            for _z in ('cape', 'gefuehlt', 'strahlung', 'cin',
                       'lifted_index', 'nullgradgrenze', 'sicht'):
                if _z in _hd.columns:
                    _spalten_ref.append(_z)
            _wolken_ref = _hd[_spalten_ref].copy()
            break

    for name, cfg in ENSEMBLE_MODELLE.items():
        schritt += 1
        if fortschritt:
            fortschritt.progress(schritt/schritte,
                                 text=f"Lade {name} ({cfg['mitglieder']} Mitgl.) …")
        df = hole_ensemble(lat, lon, name, tage)
        if df is not None and not df.empty:
            df = hoehenkorrektur(df, delta_h, gradient_df=gradient_df)
            ensemble_daten[name] = bewerte_ensemble_df(df, af, _wolken_ref)

    schritt += 1
    if fortschritt: fortschritt.progress(schritt/schritte, text="Lade Bodendaten …")
    boden = hole_bodendaten(lat, lon, tage)
    bodenindex = berechne_bodenindex(boden, bodenart)
    infil = berechne_infiltration(boden['hourly']) if boden else None

    schritt += 1
    if fortschritt: fortschritt.progress(schritt/schritte, text="Lade Sonnenstände …")
    sonne = hole_sonnenzeiten(lat, lon, tage)

    return {'haupt': haupt_daten, 'ensemble': ensemble_daten,
            'bodenindex': bodenindex, 'boden': boden, 'infiltration': infil,
            'sonne': sonne, 'modellhoehe': modellhoehe,
            'standorthoehe': standorthoehe, 'delta_h': delta_h,
            'gradient': gradient_df, 'inversion_anteil': inversion_anteil}


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
        gefuehlt_l, cape_l, uv_l, nswm_l = [], [], [], []
        strahlung_l = []
        cin_l, li_l, nullgrad_l, sicht_l, schnee_l, wind80_l = ([] for _ in range(6))
        ns_p75_l, ns_p90_l, ns_mit_l, ns_w1_l, ns_w5_l = [], [], [], [], []
        n_ens_mitglieder = 0
        n_haupt_laeufe = 0
        temp_spread = 0

        # Stimmen werden nach Familie gesammelt, nicht nach Quelle:
        # sechs ICON-Ableger sind eine Meinung, nicht sechs.
        roh_votes = []          # (familie, ampel_int, rohgewicht)
        familien_gesehen = set()

        for name, cfg in ENSEMBLE_MODELLE.items():
            if name not in ensemble: continue
            z = ensemble[name][ensemble[name]['time'] == zeit]
            if z.empty: continue
            z = z.iloc[0]
            fam = MODELL_FAMILIE.get(name, name)
            # Mitgliederzahl geht nur gedämpft ein: ein 51-köpfiges
            # Ensemble ist besser aufgelöst als ein 21-köpfiges, aber
            # nicht zweieinhalbmal so vertrauenswürdig.
            g = (skill_gewicht(_aufloesung_km(cfg['aufloesung']), stunden_voraus)
                 * (cfg['mitglieder'] / 30.0) ** 0.35)
            roh_votes.append((fam, z['ampel_int'], g))
            familien_gesehen.add(fam)
            n_ens_mitglieder += cfg['mitglieder']
            if z['temp_median'] is not None: temps.append(z['temp_median'])
            if z['feuchte_median'] is not None: feuchten.append(z['feuchte_median'])
            if z['wind_median'] is not None: winde.append(z['wind_median'])
            ns_vals.append(z['ns_median'])
            ns_wahrs.append(z['ns_wahrscheinlichkeit'])
            for feld, ziel in (('ns_p75', ns_p75_l), ('ns_p90', ns_p90_l),
                               ('ns_mittel', ns_mit_l),
                               ('ns_wahr_1mm', ns_w1_l),
                               ('ns_wahr_5mm', ns_w5_l)):
                if feld in z.index and z[feld] is not None and not pd.isna(z[feld]):
                    ziel.append(float(z[feld]))
            temp_spread = max(temp_spread, z['temp_spread'])

        for name, df in haupt.items():
            z = df[df['time'] == zeit]
            if z.empty: continue
            z = z.iloc[0]
            _af_h = dict(af)
            _mw = z.get('ns_wahr_modell')
            _af_h['_ns_wahr'] = (
                float(_mw) if _mw is not None and not pd.isna(_mw)
                else (100.0 if (z.get('niederschlag') or 0) > 0.2 else 0.0))
            _af_h['_ns_p75'] = float(z.get('niederschlag') or 0)
            _af_h['_wolken_tief'] = float(z.get('wolken_tief') or 0)
            _af_h['_wolken_mittel'] = float(z.get('wolken_mittel') or 0)
            _af_h['_wolken_gesamt'] = float(z.get('wolken') or 0)
            _af_h['_cape'] = float(z.get('cape') or 0)
            _af_h['_cin'] = z.get('cin')
            _af_h['_lifted_index'] = z.get('lifted_index')
            _af_h['_nullgradgrenze'] = z.get('nullgradgrenze')
            _af_h['_sicht'] = z.get('sicht')
            _af_h['_schneefall'] = z.get('schneefall')
            _af_h['_stunde'] = zeit.hour
            _af_h['_gefuehlt'] = z.get('gefuehlt')
            _af_h['_strahlung'] = z.get('strahlung')
            ai, _ = bewerte_werte(z['temp'], z['feuchte'], z['wind'],
                                  z['niederschlag'], _af_h)
            # Gewichtung nach Maschenweite und Vorlaufzeit statt pauschal:
            # AROME mit 2,5 km schlägt GFS mit 25 km am ersten Tag deutlich,
            # ab Tag fünf gleichen sich beide an.
            _cfg_h = HAUPTLAUFE.get(name, {})
            _gew = skill_gewicht(_aufloesung_km(_cfg_h.get('aufloesung', 25)),
                                 stunden_voraus)
            _fam = MODELL_FAMILIE.get(name, name)
            roh_votes.append((_fam, ai, _gew))
            familien_gesehen.add(_fam)
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
                               ('taupunkt', taupunkt_l),
                               ('gefuehlt', gefuehlt_l), ('cape', cape_l),
                               ('uv', uv_l), ('ns_wahr_modell', nswm_l),
                               ('strahlung', strahlung_l), ('cin', cin_l),
                               ('lifted_index', li_l),
                               ('nullgradgrenze', nullgrad_l),
                               ('sicht', sicht_l), ('schneefall', schnee_l),
                               ('wind80', wind80_l)):
                if feld in z.index and z[feld] is not None and not pd.isna(z[feld]):
                    ziel.append(float(z[feld]))

        if not roh_votes: continue

        # --- Gewichte auf Familienebene normieren ---
        # Jede Familie erhält ihr festes Gesamtgewicht, unabhängig davon,
        # mit wie vielen Ablegern sie vertreten ist. Damit kann ein
        # einzelner Modellkern die Bewertung nicht mehr dominieren.
        fam_summe = {}
        for fam, _a, g in roh_votes:
            fam_summe[fam] = fam_summe.get(fam, 0.0) + g

        votes = []
        fam_urteil = {}      # Familie -> gewichtetes Mittel ihrer Stufe
        for fam, a, g in roh_votes:
            basis = FAMILIEN_GEWICHT.get(fam, 0.6)
            anteil = g / fam_summe[fam] if fam_summe[fam] else 0.0
            votes.append((a, basis * anteil))
            fam_urteil[fam] = fam_urteil.get(fam, 0.0) + a * anteil

        n_familien = len(fam_urteil)
        gesamt_g = sum(g for _, g in votes)
        konsens_wert = sum(a * g for a, g in votes) / gesamt_g

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

        # --- Sicherheit aus der Streuung zwischen den Familien ---
        # Vorher wurde über alle Einzelquellen gestreut; sechs ICON-Ableger
        # sind sich naturgemäß einig, wodurch die Sicherheit systematisch
        # zu hoch ausfiel. Maßgeblich ist die Uneinigkeit unabhängiger Kerne.
        fam_werte = list(fam_urteil.values())
        std = float(np.std(fam_werte)) if len(fam_werte) > 1 else 1.2

        # Mit weniger als drei unabhängigen Familien lässt sich Streuung
        # kaum beurteilen — dann wird bewusst vorsichtiger eingestuft.
        if n_familien < 3:
            deckel = 'mittel'
        elif n_familien < 4:
            deckel = 'hoch'
        else:
            deckel = 'sehr hoch'

        if stunden_voraus <= 48:
            sicherheit = ('sehr hoch' if std < 0.35 else 'hoch' if std < 0.75
                          else 'mittel' if std < 1.25 else 'gering')
        elif stunden_voraus <= 120:
            sicherheit = ('hoch' if std < 0.35 else 'mittel' if std < 0.85
                          else 'gering')
        else:
            sicherheit = 'mittel' if std < 0.5 else 'gering'

        _rang = {'gering': 0, 'mittel': 1, 'hoch': 2, 'sehr hoch': 3}
        if _rang[sicherheit] > _rang[deckel]:
            sicherheit = deckel

        # Begründung aus Medianwerten
        temp_med = float(np.median(temps)) if temps else None
        feuchte_med = float(np.median(feuchten)) if feuchten else None
        wind_med = float(np.median(winde)) if winde else None
        ns_med = float(np.median(ns_vals)) if ns_vals else 0
        ns_wahr = float(np.mean(ns_wahrs)) if ns_wahrs else 0
        # Für die Entscheidung zählt nicht der mittlere Wert, sondern das
        # obere Ende der Verteilung: Wenn ein Viertel der Läufe Regen
        # bringt, ist das Schwad nass — auch wenn der Median null ist.
        ns_p75 = float(np.median(ns_p75_l)) if ns_p75_l else ns_med
        ns_p90 = float(np.median(ns_p90_l)) if ns_p90_l else ns_p75
        ns_mit = float(np.median(ns_mit_l)) if ns_mit_l else ns_med
        ns_w1 = float(np.mean(ns_w1_l)) if ns_w1_l else 0.0
        ns_w5 = float(np.mean(ns_w5_l)) if ns_w5_l else 0.0
        strahlung_med = (float(np.median(strahlung_l))
                         if strahlung_l else None)
        # Begründung mit denselben Zusatzwerten erzeugen wie die Bewertung
        _af_gr = dict(af)
        _af_gr['_ns_wahr'] = ns_wahr
        _af_gr['_ns_p75'] = ns_p75
        _af_gr['_ns_wahr_1mm'] = ns_w1
        _af_gr['_ns_wahr_5mm'] = ns_w5
        _af_gr['_strahlung'] = strahlung_med

        def _med(liste):
            return float(np.median(liste)) if liste else None

        cin_med, li_med = _med(cin_l), _med(li_l)
        nullgrad_med, sicht_med = _med(nullgrad_l), _med(sicht_l)
        schnee_med, wind80_med = _med(schnee_l), _med(wind80_l)
        _af_gr['_cin'] = cin_med
        _af_gr['_lifted_index'] = li_med
        _af_gr['_nullgradgrenze'] = nullgrad_med
        _af_gr['_sicht'] = sicht_med
        _af_gr['_schneefall'] = schnee_med
        _af_gr['_stunde'] = zeit.hour
        _af_gr['_wolken_gesamt'] = float(np.median(wolken_l)) if wolken_l else 0.0
        _af_gr['_wolken_tief'] = float(np.median(wolken_t_l)) if wolken_t_l else 0.0
        _af_gr['_wolken_mittel'] = float(np.median(wolken_m_l)) if wolken_m_l else 0.0
        _af_gr['_cape'] = float(np.median(cape_l)) if cape_l else 0.0
        _af_gr['_gefuehlt'] = (float(np.median(gefuehlt_l))
                               if gefuehlt_l else None)
        _, wetter_gruende = bewerte_werte(temp_med, feuchte_med, wind_med,
                                          ns_med, _af_gr)
        if boden_grund:
            wetter_gruende = [boden_grund] + wetter_gruende

        zeilen.append({
            'time': zeit, 'ampel': k_ampel, 'ampel_int': k_int,
            'sicherheit': sicherheit, 'stunden_voraus': stunden_voraus,
            'n_laeufe': n_ens_mitglieder + n_haupt_laeufe,
            'n_ens_mitglieder': n_ens_mitglieder, 'n_haupt': n_haupt_laeufe,
            'n_familien': n_familien,
            'familien': ', '.join(sorted(FAMILIEN_NAME.get(f, f)
                                         for f in fam_urteil)),
            'familien_std': float(std),
            'temp_median': temp_med, 'temp_spread': float(temp_spread),
            'feuchte_median': feuchte_med, 'wind_median': wind_med,
            'ns_median': ns_med, 'ns_wahrscheinlichkeit': ns_wahr,
            'ns_p75': ns_p75, 'ns_p90': ns_p90, 'ns_mittel': ns_mit,
            'ns_wahr_1mm': ns_w1, 'ns_wahr_5mm': ns_w5,
            'strahlung': strahlung_med, 'cin': cin_med,
            'lifted_index': li_med, 'nullgradgrenze': nullgrad_med,
            'sicht': sicht_med, 'schneefall': schnee_med,
            'wind80': wind80_med,
            'boeen': float(np.median(boeen_l)) if boeen_l else None,
            'windrichtung': (float(np.median(wrichtung_l))
                             if wrichtung_l else None),
            'wolken': float(np.median(wolken_l)) if wolken_l else None,
            'wolken_tief': float(np.median(wolken_t_l)) if wolken_t_l else None,
            'wolken_mittel': float(np.median(wolken_m_l)) if wolken_m_l else None,
            'wolken_hoch': float(np.median(wolken_h_l)) if wolken_h_l else None,
            'taupunkt': float(np.median(taupunkt_l)) if taupunkt_l else None,
            'gefuehlt': float(np.median(gefuehlt_l)) if gefuehlt_l else None,
            'cape': float(np.median(cape_l)) if cape_l else None,
            'uv': float(np.median(uv_l)) if uv_l else None,
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


def _bloecke_der_stufe(konsens_df, mindeststufe):
    """Zusammenhängende Blöcke, in denen die Stufe nicht unterschritten wird."""
    bloecke, akt = [], None
    for _, row in konsens_df.iterrows():
        if int(row['ampel_int']) >= mindeststufe:
            if akt and (row['time'] - akt['ende']) <= timedelta(hours=1):
                akt['ende'] = row['time']
                akt['stunden'] += 1
                akt['summe'] += int(row['ampel_int'])
                if row['sicherheit'] == 'gering':
                    akt['unsicher'] += 1
            else:
                if akt: bloecke.append(akt)
                akt = {'start': row['time'], 'ende': row['time'], 'stunden': 1,
                       'sicherheit': row['sicherheit'],
                       'summe': int(row['ampel_int']),
                       'unsicher': 1 if row['sicherheit'] == 'gering' else 0}
        else:
            if akt: bloecke.append(akt); akt = None
    if akt: bloecke.append(akt)
    return bloecke


def finde_bestes_fenster(konsens_df):
    """
    Sucht das praktisch brauchbarste Zeitfenster.

    Vorher gewann immer die höchste erreichte Stufe, egal wie kurz sie
    anhielt: Eine einzelne Stunde „sehr gut“ schlug zwanzig Stunden
    „gut“. Für jemanden, der ein Feld befahren will, ist das die falsche
    Auskunft — eine Stunde reicht für keine Arbeit.

    Deshalb zählt jetzt Stufe *und* Dauer: Ein besseres Fenster setzt sich
    durch, wenn es lang genug ist, um damit etwas anzufangen; sonst
    gewinnt der längere brauchbare Block.
    """
    if konsens_df is None or konsens_df.empty:
        return None, None, 0

    kandidaten = []
    for mindeststufe, name in ((4, 'sehr gut'), (3, 'gut'), (2, 'bedingt')):
        bloecke = _bloecke_der_stufe(konsens_df, mindeststufe)
        if not bloecke:
            continue
        best = max(bloecke, key=lambda x: (x['stunden'], x['summe']))
        # Nutzwert: Dauer, gedeckelt bei acht Stunden (mehr bringt für die
        # Planung eines Arbeitstages kaum Zusatznutzen), multipliziert mit
        # der mittleren Stufe im Block.
        nutzen = min(best['stunden'], 8) * (best['summe'] / best['stunden'])
        kandidaten.append((nutzen, mindeststufe, name, best, len(bloecke)))

    if not kandidaten:
        return None, None, 0

    kandidaten.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, mindeststufe, mindest_name, best, n = kandidaten[0]

    # Zwei Angaben statt einer, weil beide etwas anderes bedeuten:
    #   • die Untergrenze — so gut ist es mindestens, durchgehend
    #   • die mittlere Stufe — so gut ist es im Schnitt
    # Ein Block mit neun Stunden im Mittel „gut“ nur deshalb
    # „bedingt“ zu nennen, weil eine Stunde darin abfällt, verkauft
    # die Lage unter Wert.
    best['mittelstufe'] = best['summe'] / best['stunden']
    best['mindeststufe'] = mindeststufe
    best['mindest_name'] = mindest_name
    name = INT_AMPEL[int(np.floor(best['mittelstufe']))]
    return name, best, n


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
    Rückgabe: (symbol, kurzbeschreibung, svg_art)
    """
    wolken = 0.0 if wolken is None or pd.isna(wolken) else float(wolken)
    ns_mm = 0.0 if ns_mm is None or pd.isna(ns_mm) else float(ns_mm)
    ns_wahr = 0.0 if ns_wahr is None or pd.isna(ns_wahr) else float(ns_wahr)
    kalt = temp is not None and not pd.isna(temp) and float(temp) <= 1.0

    # Niederschlag dominiert die Darstellung
    if ns_mm >= 1.5 or (ns_wahr >= 70 and ns_mm >= 0.5):
        return (('❄️', 'Schneefall', 'schnee') if kalt
                else ('🌧️', 'Regen', 'regen'))
    if ns_mm >= 0.4 or ns_wahr >= 55:
        return (('🌨️', 'Schneeschauer', 'schneeschauer') if kalt
                else ('🌦️', 'Schauer', 'schauer'))
    if ns_mm >= 0.1 or ns_wahr >= 35:
        return ('🌦️', 'einzelne Schauer', 'schauer')

    # Sonst nach Bewölkung
    if wolken >= 85:
        return ('☁️', 'bedeckt', 'bedeckt')
    if wolken >= 60:
        return ('🌥️', 'stark bewölkt', 'stark')
    if wolken >= 30:
        return (('⛅', 'wechselnd bewölkt', 'wechselnd') if ist_tag
                else ('☁️', 'bewölkt', 'klar_wolken'))
    if wolken >= 12:
        return (('🌤️', 'heiter', 'heiter') if ist_tag
                else ('🌙', 'gering bewölkt', 'klar_wolken'))
    return ('☀️', 'sonnig', 'sonnig') if ist_tag else ('🌙', 'klar', 'klar')


def baue_wetteruebersicht(konsens_df, max_tage=7, sonne=None):
    """
    Baut die Tagesübersicht auf: je Tag Höchst- und Tiefstwert,
    Wetterlage nach Tagesabschnitten sowie Sonnenstunden und
    Niederschlagswahrscheinlichkeit als Tagesbilanz.
    """
    if konsens_df is None or konsens_df.empty:
        return None

    df = konsens_df.copy()
    df['datum'] = df['time'].dt.date
    df['stunde'] = df['time'].dt.hour

    tage = sorted(df['datum'].unique())[:max_tage]
    heute = datetime.now().date()

    spalten = []
    for tag in tage:
        tagdaten = df[df['datum'] == tag]
        if tagdaten.empty:
            continue

        ts = pd.Timestamp(tag)
        if tag == heute:
            titel, unterzeile = 'heute', f"{tag_kurz(ts)}, {ts.strftime('%d.%m.')}"
        elif (tag - heute).days == 1:
            titel, unterzeile = 'morgen', f"{tag_kurz(ts)}, {ts.strftime('%d.%m.')}"
        else:
            titel, unterzeile = WOCHENTAGE_LANG.get(
                ts.strftime('%A'), tag_kurz(ts)), ts.strftime('%d.%m.')

        temps = pd.to_numeric(tagdaten['temp_median'], errors='coerce').dropna()
        t_max = float(temps.max()) if not temps.empty else None
        t_min = float(temps.min()) if not temps.empty else None

        # Tagesbilanz Niederschlag: höchste Wahrscheinlichkeit im Tagesverlauf
        ns_w = pd.to_numeric(tagdaten['ns_wahrscheinlichkeit'],
                             errors='coerce').fillna(0)
        ns_wahr_tag = float(ns_w.max()) if not ns_w.empty else 0.0
        ns_summe = float(pd.to_numeric(tagdaten['ns_median'],
                                       errors='coerce').fillna(0).sum())

        # Sonnenstunden: Tagesstunden mit geringer Bewölkung innerhalb
        # der astronomisch möglichen Sonnenscheindauer
        sonnenstunden = _sonnenstunden(tagdaten, tag, sonne)

        abschnitte = []
        for label, von, bis in TAGESABSCHNITTE:
            if von <= bis:
                teil = tagdaten[(tagdaten['stunde'] >= von)
                                & (tagdaten['stunde'] <= bis)]
            else:
                teil = tagdaten[(tagdaten['stunde'] >= von)
                                | (tagdaten['stunde'] <= bis)]
            if teil.empty:
                abschnitte.append({'label': label, 'leer': True})
                continue

            wolken = (float(pd.to_numeric(teil['wolken'], errors='coerce').mean())
                      if 'wolken' in teil.columns else np.nan)
            ns_mm = float(pd.to_numeric(teil['ns_median'],
                                        errors='coerce').fillna(0).max())
            ns_wahr = float(pd.to_numeric(teil['ns_wahrscheinlichkeit'],
                                          errors='coerce').fillna(0).mean())
            temp_mit = float(pd.to_numeric(teil['temp_median'],
                                           errors='coerce').mean())
            wind = (float(pd.to_numeric(teil['wind_median'],
                                        errors='coerce').mean())
                    if 'wind_median' in teil.columns else np.nan)
            richtung = (pd.to_numeric(teil['windrichtung'],
                                      errors='coerce').median()
                        if 'windrichtung' in teil.columns else None)
            ist_tag = von <= 18
            sym, beschr, svg_art = _wettersymbol(
                wolken, ns_mm, ns_wahr, temp_mit, ist_tag)

            abschnitte.append({
                'label': label, 'leer': False, 'svg_art': svg_art,
                'beschreibung': beschr, 'temp': temp_mit,
                'ns_mm': ns_mm, 'ns_wahr': ns_wahr,
                'wind': wind, 'richtung': richtung})

        spalten.append({
            'titel': titel, 'unterzeile': unterzeile,
            't_max': t_max, 't_min': t_min,
            'ns_wahr_tag': ns_wahr_tag, 'ns_summe': ns_summe,
            'sonnenstunden': sonnenstunden,
            'abschnitte': abschnitte,
        })

    return spalten


def _sonnenstunden(tagdaten, tag, sonne):
    """
    Schätzt die Sonnenscheindauer eines Tages.
    Grundlage ist die astronomisch mögliche Tageslänge, gewichtet
    mit dem stündlichen Bewölkungsgrad.
    """
    if 'wolken' not in tagdaten.columns:
        return None

    # Tageslichtfenster bestimmen
    auf_h, unter_h = 7.0, 19.0
    if sonne is not None and not sonne.empty:
        zeile = sonne[sonne['datum'] == tag]
        if not zeile.empty:
            auf = zeile.iloc[0]['aufgang']
            unter = zeile.iloc[0]['untergang']
            if pd.notna(auf) and pd.notna(unter):
                auf_h = auf.hour + auf.minute / 60.0
                unter_h = unter.hour + unter.minute / 60.0

    tags = tagdaten[(tagdaten['stunde'] >= int(auf_h))
                    & (tagdaten['stunde'] <= int(unter_h))]
    if tags.empty:
        return None

    wolken = pd.to_numeric(tags['wolken'], errors='coerce').fillna(50.0)
    # Anteil möglicher Sonnenscheindauer je Stunde
    anteil = (1.0 - (wolken / 100.0) ** 1.4).clip(0, 1)
    return float(anteil.sum())


def _svg_wettersymbol(art, groesse=52):
    """
    Realistisch wirkendes Wettersymbol als Vektorgrafik.
    Weiche Verläufe, plastische Wolken, klare Formen.
    """
    g = groesse
    uid = "s%d" % (abs(hash(art)) % 100000)

    defs = (
        '<defs>'
        '<radialGradient id="sun_' + uid + '" cx="35%" cy="30%" r="72%">'
        '<stop offset="0%" stop-color="#fff3c4"/>'
        '<stop offset="45%" stop-color="#ffd24a"/>'
        '<stop offset="100%" stop-color="#f39c12"/>'
        '</radialGradient>'
        '<radialGradient id="glow_' + uid + '" cx="50%" cy="50%" r="50%">'
        '<stop offset="55%" stop-color="#ffd24a" stop-opacity="0.28"/>'
        '<stop offset="100%" stop-color="#ffd24a" stop-opacity="0"/>'
        '</radialGradient>'
        '<linearGradient id="cl_' + uid + '" x1="20%" y1="0%" x2="60%" y2="100%">'
        '<stop offset="0%" stop-color="#ffffff"/>'
        '<stop offset="55%" stop-color="#f2f6fa"/>'
        '<stop offset="100%" stop-color="#cfd9e4"/>'
        '</linearGradient>'
        '<linearGradient id="cld_' + uid + '" x1="20%" y1="0%" x2="60%" y2="100%">'
        '<stop offset="0%" stop-color="#c2ccd8"/>'
        '<stop offset="55%" stop-color="#9daabb"/>'
        '<stop offset="100%" stop-color="#78879a"/>'
        '</linearGradient>'
        '<linearGradient id="mn_' + uid + '" x1="20%" y1="10%" x2="80%" y2="90%">'
        '<stop offset="0%" stop-color="#fdfdff"/>'
        '<stop offset="100%" stop-color="#c4d0de"/>'
        '</linearGradient>'
        '<linearGradient id="dr_' + uid + '" x1="50%" y1="0%" x2="50%" y2="100%">'
        '<stop offset="0%" stop-color="#7fb8e8"/>'
        '<stop offset="100%" stop-color="#2f7fc4"/>'
        '</linearGradient>'
        '<filter id="sh_' + uid + '" x="-30%" y="-30%" width="170%" height="170%">'
        '<feDropShadow dx="0" dy="' + ('%.2f' % (g * 0.018)) + '" '
        'stdDeviation="' + ('%.2f' % (g * 0.022)) + '" '
        'flood-color="#7d8b9c" flood-opacity="0.30"/>'
        '</filter>'
        '</defs>')

    def sonne(cx, cy, r, mit_strahlen=True):
        teile = ['<circle cx="%.1f" cy="%.1f" r="%.1f" fill="url(#glow_%s)"/>'
                 % (cx, cy, r * 2.0, uid)]
        if mit_strahlen:
            for a in range(0, 360, 30):
                rad = np.radians(a)
                x1, y1 = cx + r * 1.32 * np.cos(rad), cy + r * 1.32 * np.sin(rad)
                x2, y2 = cx + r * 1.70 * np.cos(rad), cy + r * 1.70 * np.sin(rad)
                teile.append(
                    '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" '
                    'stroke="#f7c035" stroke-width="%.1f" stroke-linecap="round"/>'
                    % (x1, y1, x2, y2, r * 0.20))
        teile.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="url(#sun_%s)"/>'
                     % (cx, cy, r, uid))
        teile.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#fff8dc" '
                     'opacity="0.5"/>' % (cx - r * 0.28, cy - r * 0.32, r * 0.34))
        return ''.join(teile)

    def mond(cx, cy, r):
        return ('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="url(#glow_%s)" '
                'opacity="0.5"/>' % (cx, cy, r * 1.9, uid)
                + '<path d="M %.1f %.1f a %.1f %.1f 0 1 0 %.1f %.1f '
                  'a %.1f %.1f 0 1 1 %.1f %.1f z" fill="url(#mn_%s)"/>'
                  % (cx + r * 0.34, cy - r * 0.92, r, r, r * 0.10, r * 1.84,
                     r * 0.82, r * 0.82, -r * 0.10, -r * 1.84, uid))

    def wolke(cx, cy, b, dunkel=False, schatten=True):
        grad = 'url(#cld_%s)' % uid if dunkel else 'url(#cl_%s)' % uid
        f = ' filter="url(#sh_%s)"' % uid if schatten else ''
        h = b * 0.52
        return (
            '<g%s>' % f
            + '<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" fill="%s"/>'
              % (cx - b * 0.22, cy + h * 0.10, b * 0.30, h * 0.46, grad)
            + '<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" fill="%s"/>'
              % (cx + b * 0.06, cy - h * 0.16, b * 0.34, h * 0.54, grad)
            + '<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" fill="%s"/>'
              % (cx + b * 0.28, cy + h * 0.06, b * 0.28, h * 0.44, grad)
            + '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" '
              'fill="%s"/>'
              % (cx - b * 0.50, cy + h * 0.02, b * 1.00, h * 0.50, h * 0.25, grad)
            + '</g>')

    def tropfen(anzahl, y0=None, breit=False):
        y0 = y0 if y0 is not None else g * 0.70
        out = []
        pos = ([0.28, 0.44, 0.60, 0.76] if breit else [0.34, 0.52, 0.70])[:anzahl]
        for i, px in enumerate(pos):
            x = g * px
            versatz = (i % 2) * g * 0.045
            l = g * 0.13
            out.append(
                '<path d="M %.1f %.1f c %.1f %.1f %.1f %.1f 0 %.1f '
                'c %.1f %.1f %.1f %.1f 0 %.1f z" fill="url(#dr_%s)"/>'
                % (x, y0 + versatz, g * 0.042, l * 0.55, g * 0.042, l * 0.80, l,
                   -g * 0.042, -l * 0.20, -g * 0.042, -l * 0.45, -l, uid))
        return ''.join(out)

    def flocken(anzahl, y0=None):
        y0 = y0 if y0 is not None else g * 0.76
        out = []
        pos = [0.34, 0.52, 0.70][:anzahl]
        for i, px in enumerate(pos):
            x = g * px
            y = y0 + (i % 2) * g * 0.05
            r = g * 0.058
            arme = []
            for a in (0, 60, 120):
                rad = np.radians(a)
                dx, dy = r * np.cos(rad), r * np.sin(rad)
                arme.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
                            % (x - dx, y - dy, x + dx, y + dy))
            out.append('<g stroke="#a9d3f0" stroke-width="%.1f" '
                       'stroke-linecap="round">%s</g>'
                       % (g * 0.026, ''.join(arme)))
        return ''.join(out)

    varianten = {
        'sonnig':        sonne(g * 0.50, g * 0.47, g * 0.20),
        'heiter':        sonne(g * 0.37, g * 0.35, g * 0.155)
                         + wolke(g * 0.58, g * 0.62, g * 0.62),
        'wechselnd':     sonne(g * 0.34, g * 0.33, g * 0.145, False)
                         + wolke(g * 0.55, g * 0.58, g * 0.76),
        'stark':         wolke(g * 0.40, g * 0.42, g * 0.58, False, False)
                         + wolke(g * 0.56, g * 0.60, g * 0.80),
        'bedeckt':       wolke(g * 0.40, g * 0.40, g * 0.60, True, False)
                         + wolke(g * 0.55, g * 0.58, g * 0.84, True),
        'schauer':       sonne(g * 0.32, g * 0.30, g * 0.135, False)
                         + wolke(g * 0.54, g * 0.50, g * 0.74) + tropfen(2),
        'regen':         wolke(g * 0.50, g * 0.44, g * 0.82, True)
                         + tropfen(4, breit=True),
        'schnee':        wolke(g * 0.50, g * 0.44, g * 0.82, True) + flocken(3),
        'schneeschauer': sonne(g * 0.32, g * 0.30, g * 0.135, False)
                         + wolke(g * 0.54, g * 0.50, g * 0.74) + flocken(2),
        'klar':          mond(g * 0.50, g * 0.46, g * 0.20),
        'klar_wolken':   mond(g * 0.38, g * 0.36, g * 0.165)
                         + wolke(g * 0.58, g * 0.62, g * 0.66),
    }
    inhalt = varianten.get(art, wolke(g * 0.50, g * 0.50, g * 0.78))

    return ('<svg viewBox="0 0 %d %d" width="%d" height="%d" '
            'xmlns="http://www.w3.org/2000/svg">%s%s</svg>'
            % (g, g, g, g, defs, inhalt))


def _windpfeil(richtung, staerke, groesse=22):
    """Kleiner Windpfeil, Richtung meteorologisch (woher der Wind weht)."""
    if richtung is None or (isinstance(richtung, float) and np.isnan(richtung)):
        return ''
    g = groesse
    dreh = float(richtung) + 180.0
    if staerke is None or (isinstance(staerke, float) and np.isnan(staerke)):
        farbe = '#9aa7b4'
    elif staerke >= 40:
        farbe = '#c0392b'
    elif staerke >= 25:
        farbe = '#d98324'
    else:
        farbe = '#9aa7b4'
    return ('<svg width="%d" height="%d" viewBox="0 0 24 24" '
            'xmlns="http://www.w3.org/2000/svg">'
            '<g transform="rotate(%.0f 12 12)">'
            '<path d="M12 4 L17 18 L12 15 L7 18 Z" fill="%s"/>'
            '</g></svg>' % (g, g, dreh, farbe))


def wetteruebersicht_html(spalten):
    """
    Tagesübersicht im Stil klassischer Wetterportale:
    Spalten je Tag, oben Höchst- und Tiefstwert, darunter die
    Tagesabschnitte mit Symbol und Wind, unten Sonnenstunden
    und Niederschlagswahrscheinlichkeit.
    """
    if not spalten:
        return ''

    css = """
    <style>
      /* display:flex + margin:auto am Kind zentriert den Streifen,
         schneidet ihn beim Überlaufen aber nicht links ab
         (anders als justify-content:center) */
      .wo-wrap {display:flex; overflow-x:auto; overflow-y:hidden;
                -webkit-overflow-scrolling:touch;
                padding-bottom:8px; width:100%;
                scrollbar-width:thin;}
      .wo-wrap::-webkit-scrollbar {height:6px;}
      .wo-wrap::-webkit-scrollbar-thumb {background:#9aabbe; border-radius:3px;}
      /* inline-flex: der Streifen wächst mit dem Inhalt, dadurch bleibt
         auch die letzte Spalte vollständig erreichbar */
      .wo-row {display:inline-flex; gap:0; flex:0 0 auto; margin:0 auto;
               border:1px solid #e2e9f1; border-radius:14px;
               background:#ffffff;
               box-shadow:0 1px 2px rgba(13,34,61,.04),
                          0 6px 20px rgba(13,34,61,.06);
               font-family:'Raleway','Segoe UI',system-ui,sans-serif;}
      .wo-col {flex:0 0 auto; width:136px; padding:13px 6px 11px 6px;
               border-right:1px solid #eef3f8; text-align:center;
               transition:background .15s ease;}
      .wo-col:hover {background:#f9fcff;}
      .wo-col:first-child {border-radius:14px 0 0 14px;}
      .wo-col:last-child {border-radius:0 14px 14px 0;}
      .wo-col:last-child {border-right:none;}
      .wo-tag {font-family:'Montserrat','Segoe UI',system-ui,sans-serif;
               font-size:1.0rem; font-weight:700; color:#0D223D;
               line-height:1.25; letter-spacing:-0.2px;}
      .wo-datum {font-size:0.75rem; color:#5b6b7d; margin-bottom:9px;
                 font-weight:500;}
      .wo-temps {display:flex; flex-direction:column; align-items:center;
                 gap:1px; padding:7px 0 9px 0;
                 border-bottom:1px solid #eef3f8; margin-bottom:6px;}
      .wo-trow {display:flex; align-items:baseline; gap:6px;}
      .wo-tlbl {font-size:0.63rem; color:#5b6b7d; width:30px;
                text-align:right; font-weight:600; white-space:nowrap;
                text-transform:uppercase; letter-spacing:.3px;}
      .wo-tmax {font-family:'Montserrat',sans-serif;
                font-size:1.45rem; font-weight:700; color:#E65100;
                line-height:1.1;}
      .wo-tmin {font-family:'Montserrat',sans-serif;
                font-size:1.12rem; font-weight:600; color:#1E88E5;
                line-height:1.1;}
      .wo-tline {width:58px; height:1px; background:#e2e9f1; margin:2px 0;}
      .wo-abs {display:grid; grid-template-columns:46px 1fr 24px;
               align-items:center; column-gap:4px; padding:4px 2px;
               justify-items:center;}
      .wo-abs-sym {width:46px; height:46px; display:flex;
                   align-items:center; justify-content:center;}
      .wo-abs-txt {text-align:left; line-height:1.15; width:100%;
                   padding-left:2px;}
      .wo-abs-lbl {font-size:0.67rem; color:#55697f; font-weight:600;
                   white-space:nowrap;}
      .wo-abs-t {font-family:'Montserrat',sans-serif;
                 font-size:0.86rem; font-weight:700; color:#16324f;}
      .wo-abs-wind {width:24px; height:24px; display:flex;
                    align-items:center; justify-content:center; opacity:.85;}
      .wo-foot {border-top:1px solid #eef3f8; margin-top:7px; padding-top:8px;
                display:flex; flex-direction:column; gap:5px;
                align-items:center;}
      .wo-fitem {display:flex; align-items:center; gap:5px;
                 font-size:0.82rem; font-weight:600; color:#33506e;}
      @media (max-width: 640px) {
        .wo-col {width:118px; padding:9px 4px 8px 4px;}
        .wo-tmax {font-size:1.22rem;}
        .wo-tmin {font-size:0.98rem;}
        .wo-tag {font-size:0.9rem;}
        .wo-abs-t {font-size:0.76rem;}
        .wo-abs {grid-template-columns:40px 1fr 20px;}
        .wo-abs-sym {width:40px; height:40px;}
        .wo-abs-wind {width:20px; height:20px;}
        .wo-fitem {font-size:0.74rem;}
        .wo-tlbl {width:26px; font-size:0.58rem; letter-spacing:.2px;}
      }
    </style>
    """

    sonne_icon = ('<svg width="16" height="16" viewBox="0 0 24 24" '
                  'xmlns="http://www.w3.org/2000/svg" fill="none" '
                  'stroke="#F5A623" stroke-width="1.9" stroke-linecap="round">'
                  '<circle cx="12" cy="12" r="4.2" fill="#FFC107" '
                  'stroke="none"/>'
                  '<line x1="12" y1="2.5" x2="12" y2="5"/>'
                  '<line x1="12" y1="19" x2="12" y2="21.5"/>'
                  '<line x1="2.5" y1="12" x2="5" y2="12"/>'
                  '<line x1="19" y1="12" x2="21.5" y2="12"/>'
                  '<line x1="5.4" y1="5.4" x2="7.2" y2="7.2"/>'
                  '<line x1="16.8" y1="16.8" x2="18.6" y2="18.6"/>'
                  '<line x1="5.4" y1="18.6" x2="7.2" y2="16.8"/>'
                  '<line x1="16.8" y1="7.2" x2="18.6" y2="5.4"/>'
                  '</svg>')

    tropfen_icon = ('<svg width="16" height="16" viewBox="0 0 24 24" '
                    'xmlns="http://www.w3.org/2000/svg">'
                    '<path d="M12 3.2 C12 3.2 5.8 11 5.8 15 '
                    'a6.2 6.2 0 0 0 12.4 0 C18.2 11 12 3.2 12 3.2 z" '
                    'fill="none" stroke="#1E88E5" stroke-width="1.8"/>'
                    '</svg>')

    html = [css, '<div class="wo-wrap"><div class="wo-row">']

    for sp in spalten:
        html.append('<div class="wo-col">')
        html.append('<div class="wo-tag">%s</div>' % sp['titel'])
        html.append('<div class="wo-datum">%s</div>' % sp['unterzeile'])

        # Höchst- und Tiefstwert
        html.append('<div class="wo-temps">')
        if sp['t_max'] is not None:
            html.append(
                '<div class="wo-trow"><span class="wo-tlbl">max</span>'
                '<span class="wo-tmax">%.0f&deg;</span></div>' % sp['t_max'])
        html.append('<div class="wo-tline"></div>')
        if sp['t_min'] is not None:
            html.append(
                '<div class="wo-trow"><span class="wo-tlbl">min</span>'
                '<span class="wo-tmin">%.0f&deg;</span></div>' % sp['t_min'])
        html.append('</div>')

        # Tagesabschnitte
        for ab in sp['abschnitte']:
            if ab.get('leer'):
                html.append(
                    '<div class="wo-abs" style="opacity:.25">'
                    '<div class="wo-abs-txt"><div class="wo-abs-lbl">%s</div>'
                    '<div class="wo-abs-t">&ndash;</div></div></div>'
                    % ab['label'])
                continue
            sym = _svg_wettersymbol(ab['svg_art'], 42)
            pfeil = _windpfeil(ab.get('richtung'), ab.get('wind'), 20)
            html.append(
                '<div class="wo-abs" title="%s">'
                '<div class="wo-abs-sym">%s</div>'
                '<div class="wo-abs-txt">'
                '<div class="wo-abs-t">%.0f&deg;</div>'
                '<div class="wo-abs-lbl">%s</div>'
                '</div>'
                '<div class="wo-abs-wind">%s</div>'
                '</div>'
                % (ab.get('beschreibung', ''), sym, ab['temp'],
                   ab['label'], pfeil))

        # Tagesbilanz
        html.append('<div class="wo-foot">')
        if sp.get('sonnenstunden') is not None:
            html.append('<div class="wo-fitem">%s%.0f h</div>'
                        % (sonne_icon, sp['sonnenstunden']))
        html.append('<div class="wo-fitem">%s%.0f %%</div>'
                    % (tropfen_icon, sp.get('ns_wahr_tag', 0)))
        html.append('</div>')

        html.append('</div>')

    html.append('</div></div>')
    return ''.join(html)


# ============================================================
# KARTENDARSTELLUNG (Ventusky)
# ============================================================
# Ventusky stellt eine offizielle Einbettung bereit, die Radar,
# Niederschlag, Temperatur, Wind und Bewölkung mit eigener Zeitleiste
# darstellt. Die Darstellung ist hochwertig gerendert und benötigt
# keinen Zugangsschlüssel; die Ortsmarkierung wird als Parameter
# übergeben.
VENTUSKY_EMBED = 'https://embed.ventusky.com/'

# Einstiegsebene der Karte. Alle weiteren Ansichten erreicht man über
# die runde Schaltfläche in der Karte selbst — eine eigene Auswahlleiste
# in der App wäre eine doppelte Bedienung.
VENTUSKY_START_EBENE = 'radar'

# Nur zur Beschriftung: was sich in der Karte umschalten lässt.
VENTUSKY_WEITERE = ('Niederschlagsmenge', 'Temperatur', 'Wind', 'Windböen',
                    'Bewölkung', 'Schneefall', 'Luftdruck')


def ventusky_karte(ebene, lat, lon, ortsname, zoom=7, hoehe=560,
                   animation='soft'):
    """
    Bettet eine Ventusky-Karte ein.

    Die Zeitleiste, das Zoomen und der Ebenenwechsel werden von
    Ventusky selbst bereitgestellt. Der ausgewählte Standort erscheint
    als Markierung mit Beschriftung.
    """
    if lat is None or lon is None:
        pass  # Kein Standort verfügbar
        return

    # Beschriftung von Zeichen befreien, die die URL stören würden
    label = re.sub(r'[;&#?]', ' ', str(ortsname)).strip()[:40]

    url = (f"{VENTUSKY_EMBED}"
           f"?p={lat:.4f};{lon:.4f};{zoom}"
           f"&l={ebene}"
           f"&w={animation}"
           f"&pin={lat:.4f};{lon:.4f};dot;{quote(label)}")

    components.iframe(url, height=hoehe)






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

    y_labels = [f'{tag_kurz(pd.Timestamp(t))}  {pd.Timestamp(t).strftime("%d.%m.%Y")}'
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
                zahl('ns_p75', 1), zahl('ns_wahrscheinlichkeit'),
                int(r.get('n_familien') or 0), r.get('gruende', ''),
                zahl('taupunkt'), zahl('boeen'),
                (himmelsrichtung(r.get('windrichtung'))
                 if r.get('windrichtung') is not None else '–'),
                zahl('ns_wahr_1mm'),
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
            '<b>%{y}  %{x}:00 Uhr</b><br>'
            '<b>Eignung: %{customdata[0]}</b> '
            '(Vorhersagesicherheit %{customdata[1]})<br>'
            'Temperatur: %{customdata[2]} °C (±%{customdata[3]})<br>'
            'Taupunkt: %{customdata[10]} °C · Feuchte: %{customdata[5]} %<br>'
            'Wind: %{customdata[4]} km/h · Böen: %{customdata[11]} km/h '
            '(%{customdata[12]})<br>'
            'Regenrisiko: %{customdata[7]} % · über 1 mm: %{customdata[13]} %<br>'
            'Menge (75. Perzentil): %{customdata[6]} mm/h<br>'
            'Unabhängige Modellfamilien: %{customdata[8]}<br>'
            '<i>%{customdata[9]}</i><extra></extra>')))

    if unsicher_x:
        fig.add_trace(go.Scatter(
            x=unsicher_x, y=unsicher_y, mode='markers',
            marker=dict(symbol='x-thin', size=10,
                        line=dict(color='rgba(255,255,255,0.95)', width=2.0)),
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

    # X-Achse: Datum bei Mitternacht, Uhrzeit bei den anderen Ticks
    tick_vals = list(range(0, 24, 2))
    tick_text = []
    for h in tick_vals:
        if h == 0:
            # Datum aus dem ersten Tag der Daten
            erster_tag = tage_list[0] if tage_list else None
            if erster_tag:
                dt = pd.Timestamp(erster_tag)
                tick_text.append(f"{tag_kurz(dt)}<br>{dt.strftime('%d.%m.')}")
            else:
                tick_text.append("00:00")
        else:
            tick_text.append(f"{h:02d}:00")

    # Für jeden Tag nach dem ersten eine Datumslinie bei Stunde 0 setzen
    for i, tag in enumerate(tage_list[1:], 1):
        dt = pd.Timestamp(tag)
        fig.add_shape(type='line', x0=-0.5, x1=-0.5,
                      y0=i - 0.5, y1=i + 0.5,
                      line=dict(color='rgba(0,0,0,0)', width=0))
        # Datum-Annotation links außen je Zeile (im Y-Label bereits drin,
        # aber für Klarheit zusätzlich in der Achse)

    fig.update_layout(
        height=max(320, len(tage_list) * 62 + 130),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title='', tickmode='array',
                   tickvals=tick_vals,
                   ticktext=tick_text,
                   side='top', fixedrange=True),
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
        font=dict(family='Raleway, Segoe UI, sans-serif',
                  size=12, color='#33506e'),
        hoverlabel=dict(font=dict(family='Raleway, Segoe UI, sans-serif',
                                  size=12),
                        bgcolor='rgba(255,255,255,0.96)',
                        bordercolor='#e2e9f1'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(gridcolor='rgba(13,34,61,0.09)', zeroline=False)
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

    # Von klarem Himmelblau über diesig zu geschlossener Wolkendecke
    colorscale = [[0.0, '#cfe6f7'], [0.20, '#e3eef5'],
                  [0.45, '#dfe4e8'], [0.70, '#b3bcc5'],
                  [0.88, '#8e98a3'], [1.0, '#66707b']]

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


