"""
Automatisierte Tests für die Kernlogik.
Starten: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import numpy as np
import pandas as pd
import pytest
from weather_core import (
    gewitterrisiko, dampfdruckdefizit, trocknungsrate,
    bewerte_werte, berechne_konsens, wende_trockenfenster_an,
    finde_bestes_fenster, im_inca_gebiet, ANWENDUNGSFAELLE,
    STUFEN, INT_AMPEL, bewerte_ensemble_df
)


# ── Gewitterlogik ────────────────────────────────────────────────────────────

class TestGewitterrisiko:
    def test_keine_labilität(self):
        r = gewitterrisiko(200, -5, 3)
        assert r["stufe"] == 0

    def test_hohe_cape_kein_deckel(self):
        r = gewitterrisiko(2200, -10, -5, stunde=15)
        assert r["stufe"] >= 2

    def test_hohe_cape_starker_deckel(self):
        r = gewitterrisiko(2200, -250, -5, stunde=15)
        assert r["stufe"] <= 1  # Deckel verhindert Auslösung

    def test_nacht_dämpft(self):
        tag = gewitterrisiko(1500, -20, -4, stunde=15)
        nacht = gewitterrisiko(1500, -20, -4, stunde=3)
        assert tag["stufe"] >= nacht["stufe"]


# ── Physikalische Modelle ────────────────────────────────────────────────────

class TestVPD:
    def test_gesättigt_null(self):
        vpd = dampfdruckdefizit(pd.Series([20.0]), pd.Series([100.0]))
        assert float(vpd.iloc[0]) < 0.01

    def test_trocken_groß(self):
        vpd = dampfdruckdefizit(pd.Series([30.0]), pd.Series([30.0]))
        assert float(vpd.iloc[0]) > 2.5

    def test_kein_negativwert(self):
        vpd = dampfdruckdefizit(pd.Series([-10.0, 40.0]),
                                pd.Series([110.0, 0.0]))
        assert all(vpd >= 0)


class TestTrocknungsrate:
    def test_warm_trocken_hoch(self):
        r = trocknungsrate(pd.Series([28.0]), pd.Series([40.0]),
                           pd.Series([15.0]), pd.Series([700.0]))
        assert float(r.iloc[0]) > 1.5

    def test_kalt_feucht_niedrig(self):
        r = trocknungsrate(pd.Series([10.0]), pd.Series([92.0]),
                           pd.Series([2.0]), pd.Series([50.0]))
        assert float(r.iloc[0]) < 0.15


# ── Bewertungslogik ──────────────────────────────────────────────────────────

def _basis_af():
    af = dict(ANWENDUNGSFAELLE["Grünland & Feldfutter"]
              ["Heuernte (bodengetrocknet)"])
    af["name"] = "Heuernte"; af["modus"] = "standard"
    return af


class TestBewertung:
    def test_optimale_bedingungen(self):
        af = _basis_af()
        af.update({"_ns_wahr": 2.0, "_ns_p75": 0.0,
                   "_ns_wahr_1mm": 0.0, "_strahlung": 700.0})
        stufe, _ = bewerte_werte(24, 52, 12, 0.0, af)
        assert stufe >= 3

    def test_regenrisiko_erkennt_median_null(self):
        """35% Regenrisiko bei Median 0 muss bewertet werden."""
        af = _basis_af()
        af.update({"_ns_wahr": 35.0, "_ns_p75": 0.0,
                   "_ns_wahr_1mm": 18.0, "_strahlung": 600.0})
        stufe, gruende = bewerte_werte(24, 55, 12, 0.0, af)
        assert stufe <= 2
        assert any("Regenrisiko" in g or "wahrscheinlich" in g
                   for g in gruende)

    def test_zu_kalt(self):
        af = _basis_af()
        af.update({"_ns_wahr": 2.0, "_ns_p75": 0.0,
                   "_strahlung": 500.0})
        stufe, _ = bewerte_werte(5, 55, 10, 0.0, af)
        assert stufe <= 2

    def test_stufen_vollständig(self):
        for k, v in STUFEN.items():
            assert "name" in v and "farbe" in v


# ── Gesamtdurchlauf ──────────────────────────────────────────────────────────

def _test_daten(n=72):
    h = np.arange(n)
    tg = np.clip(np.sin((h % 24 - 6) / 12 * np.pi), 0, None)
    zeiten = pd.date_range("2026-09-07", periods=n, freq="h")
    rng = np.random.default_rng(42)
    return zeiten, tg, rng


def _haupt_df(zeiten, tg, rng, off=0.0):
    n = len(zeiten)
    return pd.DataFrame({
        "time": zeiten, "temp": 16 + 9 * tg + off,
        "feuchte": 88 - 33 * tg, "niederschlag": np.zeros(n),
        "wind": 8 + 6 * tg, "boeen": 15 + 10 * tg,
        "windrichtung": rng.random(n) * 360,
        "wolken": 15 + rng.random(n) * 20,
        "wolken_tief": rng.random(n) * 20,
        "wolken_mittel": rng.random(n) * 20,
        "wolken_hoch": rng.random(n) * 20,
        "taupunkt": 10 + np.zeros(n), "gefuehlt": 16 + 9 * tg,
        "cape": 900 * tg, "uv": 6 * tg,
        "ns_wahr_modell": 5 + rng.random(n) * 8,
        "strahlung": 750 * tg, "t850": 9.0 + np.zeros(n),
        "cin": -30 + np.zeros(n), "lifted_index": -2 + np.zeros(n),
        "nullgradgrenze": 3200 + np.zeros(n),
        "sicht": 20000 + np.zeros(n), "schneefall": np.zeros(n),
        "wind80": 12 + 8 * tg,
        "ist_tag": (tg > 0.05).astype(int),
    })


def _ens_df(zeiten, tg):
    n = len(zeiten)
    return pd.DataFrame({
        "time": zeiten, "temp_median": 16 + 9 * tg,
        "temp_p10": 15 + 9 * tg, "temp_p90": 17 + 9 * tg,
        "temp_spread": 2 + np.zeros(n),
        "feuchte_median": 88 - 33 * tg,
        "wind_median": 8 + 6 * tg,
        "ns_median": np.zeros(n), "ns_mittel": np.zeros(n),
        "ns_p75": np.zeros(n), "ns_p90": np.zeros(n),
        "ns_wahrscheinlichkeit": 4 + np.zeros(n),
        "ns_wahr_1mm": np.zeros(n), "ns_wahr_5mm": np.zeros(n),
        "n_mitglieder": 40,
    })


class TestAlleAnwendungsfaelle:
    @pytest.mark.parametrize(
        "kat,fall",
        [(kat, fall)
         for kat, faelle in ANWENDUNGSFAELLE.items()
         for fall in faelle]
    )
    def test_durchlauf(self, kat, fall):
        zeiten, tg, rng = _test_daten()
        af = dict(ANWENDUNGSFAELLE[kat][fall])
        af["name"] = fall; af["modus"] = "standard"
        af.setdefault("_standorthoehe", 800)

        roh = {
            "ICON EU 7km": _haupt_df(zeiten, tg, rng),
            "ECMWF IFS 9km": _haupt_df(zeiten, tg, rng, 0.8),
            "GFS 25km": _haupt_df(zeiten, tg, rng, -0.6),
            "GeoSphere AROME 2.5km": _haupt_df(zeiten, tg, rng, 0.2),
        }
        e = bewerte_ensemble_df(_ens_df(zeiten, tg), af, roh["ICON EU 7km"])
        alle = {
            "haupt": roh,
            "ensemble": {"ICON-EU-EPS 13km": e, "ECMWF IFS ENS 9km": e,
                         "GFS ENS 25km": e},
            "bodenindex": None,
        }
        k = wende_trockenfenster_an(berechne_konsens(alle, af, 3), af)
        assert len(k) > 0
        assert "ampel_int" in k.columns
        assert "n_familien" in k.columns

        ziel, best, _ = finde_bestes_fenster(k)
        if best:
            assert best["stunden"] >= 1
            assert ziel in INT_AMPEL.values()


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

class TestHilfen:
    def test_inca_gebiet(self):
        assert im_inca_gebiet(48.13, 15.14)   # Wieselburg
        assert im_inca_gebiet(47.32, 12.79)   # Zell am See
        assert not im_inca_gebiet(53.55, 9.99)  # Hamburg
        assert not im_inca_gebiet(41.90, 12.50)  # Rom
