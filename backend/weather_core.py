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

