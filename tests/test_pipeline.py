"""Unit-Tests ohne Netzwerk und ohne KI: pruefen die regelbasierten Teile der Pipeline."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from src.ai import parse_json_list, parse_json_object
from src.analyze import (
    apply_chart_filters,
    compute_chart_facts,
    ground_annotations,
    ground_title,
    profile_dataframe,
    quality_gate,
)
from src.config import load_settings
from src.history import story_similarity
from src.main import create_run_output_dir
from src.visualize import _axis_str, build_chart


@pytest.fixture
def kantone() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Kanton": ["ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "FR"],
            "Miete": [2100, 1450, 1600, 1200, 1750, 1400, 1650, 1150, 2300, 1350],
        }
    )


@pytest.fixture
def zeitreihe() -> pd.DataFrame:
    return pd.DataFrame({"Jahr": list(range(2010, 2022)), "Eistage": [30, 25, 28, 20, 22, 18, 15, 17, 12, 10, 8, 9]})


# ---------------------------------------------------------------- config


def test_ai_provider_liste_filtert_unbekannte(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq, foo ,ollama")
    settings = load_settings()
    assert settings.ai_providers == ("groq", "ollama")


def test_ohne_provider_werden_keys_erkannt_gratis_zuerst(monkeypatch):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AI_PROVIDER", "none")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GROQ_API_KEY", "y")
    assert load_settings().ai_providers == ("groq", "gemini")


def test_ungueltige_zahlen_fallen_auf_default(monkeypatch):
    monkeypatch.setenv("MAX_TOPICS", "viele")
    assert load_settings().max_topics == 5


# ---------------------------------------------------------------- ai json


@pytest.mark.parametrize(
    "antwort",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Hier ist das Ergebnis: {"a": 1} Viel Erfolg!',
    ],
)
def test_parse_json_object_toleriert_prosa_und_codefences(antwort):
    assert parse_json_object(antwort) == {"a": 1}


def test_parse_json_list_mit_wrapper_und_innerem_objekt():
    assert parse_json_list('{"topics": [{"x": 1}]}', ("topics",)) == [{"x": 1}]
    assert parse_json_list('Liste: [{"x": 1}, {"x": 2}]') == [{"x": 1}, {"x": 2}]
    assert parse_json_list("kein json") == []


# ---------------------------------------------------------------- analyse


def test_profile_erkennt_zahlen_und_kategorien(kantone):
    profile = profile_dataframe(kantone)
    types = {col["name"]: col["semantic_type"] for col in profile["columns"]}
    assert types == {"Kanton": "category", "Miete": "number"}
    assert profile["n_rows"] == 10


def test_quality_gate_lehnt_flache_daten_ab():
    df = pd.DataFrame({"x": list("abcdefghij"), "y": [100.0] * 9 + [100.5]})
    ok, reason = quality_gate(df, {"chart_type": "bar", "x_column": "x", "y_column": "y"}, 0.9, 0.5)
    assert not ok
    assert "constant" in reason


def test_quality_gate_lehnt_zu_wenige_zeilen_ab():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    ok, _ = quality_gate(df, {"chart_type": "bar", "x_column": "x", "y_column": "y"}, 0.9, 0.5)
    assert not ok


def test_quality_gate_akzeptiert_gute_daten(kantone):
    ok, reason = quality_gate(kantone, {"chart_type": "bar", "x_column": "Kanton", "y_column": "Miete"}, 0.8, 0.55)
    assert ok, reason


def test_chart_facts_rangliste_und_veraenderung(zeitreihe):
    facts = compute_chart_facts(zeitreihe, {"chart_type": "line", "x_column": "Jahr", "y_column": "Eistage"})
    assert facts["top"][0] == {"label": "2010", "value": 30.0}
    assert facts["first"]["value"] == 30.0
    assert facts["last"]["value"] == 9.0
    assert facts["change_pct"] == -70.0


def test_superlativ_titel_wird_ersetzt_wenn_daten_ihn_nicht_stuetzen(kantone):
    plan = {
        "chart_type": "bar",
        "x_column": "Kanton",
        "y_column": "Miete",
        "title": "Bern hat die höchsten Mieten",
        "highlight": {"mode": "category", "values": ["BE"]},
    }
    facts = compute_chart_facts(kantone, plan)
    ground_title(plan, facts)
    assert "höchsten" not in plan["title"]
    assert "title_grounding_note" in plan


def test_superlativ_titel_bleibt_wenn_er_stimmt(kantone):
    plan = {
        "chart_type": "bar",
        "x_column": "Kanton",
        "y_column": "Miete",
        "title": "Zug hat die höchsten Mieten",
        "highlight": {"mode": "category", "values": ["ZG"]},
    }
    ground_title(plan, compute_chart_facts(kantone, plan))
    assert plan["title"] == "Zug hat die höchsten Mieten"


def test_falsche_rekord_annotation_wird_entfernt(kantone):
    plan = {
        "chart_type": "bar",
        "x_column": "Kanton",
        "y_column": "Miete",
        "annotations": [{"x": "BE", "text": "Rekord!"}, {"x": "ZG", "text": "Spitzenreiter"}, {"x": "UR", "text": "tief"}],
    }
    ground_annotations(plan, compute_chart_facts(kantone, plan))
    assert [a["x"] for a in plan["annotations"]] == ["ZG", "UR"]


def test_filter_wird_nur_angewendet_wenn_genug_zeilen_bleiben():
    df = pd.DataFrame({"Geschlecht": ["m"] * 5 + ["w"] * 2, "y": range(7)})
    plan = {"filters": [{"column": "Geschlecht", "value": "W"}]}
    assert len(apply_chart_filters(df, plan)) == 7
    assert plan["filters"] == []

    plan = {"filters": [{"column": "Geschlecht", "value": "M"}]}
    assert len(apply_chart_filters(df, plan)) == 5


# ---------------------------------------------------------------- history


def test_gleiche_story_wird_erkannt():
    sig = {"resource_url": "u", "dataset_id": "d", "chart_type": "line", "x_column": "Jahr", "y_column": "y"}
    assert story_similarity(sig, dict(sig)) == 1.0


def test_fehlende_ids_gelten_nicht_als_gleiche_story():
    leer = {"resource_url": None, "dataset_id": None, "chart_type": None, "x_column": None, "y_column": None}
    assert story_similarity(leer, dict(leer)) == 0.0


# ---------------------------------------------------------------- output


def test_output_ordner_bekommt_suffix_bei_kollision(tmp_path):
    _, first = create_run_output_dir(tmp_path, "2026-01-01")
    (first.parent / f"{first.name}-2").mkdir()
    _, second = create_run_output_dir(tmp_path, "2026-01-01")
    assert second != first
    assert second.exists()


def test_jahreszahlen_ohne_nachkommastellen():
    assert list(_axis_str(pd.Series([2019.0, 2020.0]))) == ["2019", "2020"]


@pytest.mark.parametrize("chart_type", ["line", "bar", "horizontal_bar", "lollipop"])
def test_chart_wird_als_png_geschrieben(tmp_path, kantone, zeitreihe, chart_type):
    df, x, y = (zeitreihe, "Jahr", "Eistage") if chart_type == "line" else (kantone, "Kanton", "Miete")
    plan = {"chart_type": chart_type, "x_column": x, "y_column": y, "category_column": x, "title": "Test"}
    path = build_chart(df, plan, tmp_path / "chart.png")
    assert path.exists() and path.stat().st_size > 10_000


# ---------------------------------------------------------------- download


class _FakeResponse:
    def __init__(self, chunks, headers=None):
        self._chunks = chunks
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_download_bricht_bei_zu_grosser_datei_ab(monkeypatch):
    from src import datasets

    monkeypatch.setattr(datasets.requests, "get", lambda *a, **k: _FakeResponse([b"x" * 600, b"x" * 600]))
    with pytest.raises(ValueError):
        datasets.download_limited("http://example.test", max_bytes=1000)


def test_download_bricht_bei_gemeldeter_groesse_sofort_ab(monkeypatch):
    from src import datasets

    monkeypatch.setattr(
        datasets.requests, "get", lambda *a, **k: _FakeResponse([], {"Content-Length": "999999"})
    )
    with pytest.raises(ValueError):
        datasets.download_limited("http://example.test", max_bytes=1000)


def test_download_liefert_inhalt(monkeypatch):
    from src import datasets

    monkeypatch.setattr(datasets.requests, "get", lambda *a, **k: _FakeResponse([b"a;b\n", b"1;2\n"]))
    assert datasets.download_limited("http://example.test") == b"a;b\n1;2\n"
