from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from .ai import generate, parse_json_object
from .config import Settings


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": [],
    }

    for column in df.columns:
        series = df[column]
        col_profile: dict[str, Any] = {
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_ratio": round(float(series.isna().mean()), 3),
            "n_unique": int(series.nunique(dropna=True)),
            "sample_values": [str(value)[:80] for value in series.dropna().head(5).tolist()],
        }
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.65:
            col_profile.update(
                {
                    "semantic_type": "number",
                    "min": float(numeric.min()) if numeric.notna().any() else None,
                    "max": float(numeric.max()) if numeric.notna().any() else None,
                    "mean": float(numeric.mean()) if numeric.notna().any() else None,
                }
            )
        elif _looks_temporal(series, str(column)):
            col_profile["semantic_type"] = "time"
        else:
            col_profile["semantic_type"] = "category"
        profile["columns"].append(col_profile)

    return profile


def choose_chart_plan(
    topic: dict[str, Any],
    dataset: dict[str, Any],
    profile: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    ai_plan = _try_ai_chart_plan(topic, dataset, profile, settings)
    ai_plan = _normalize_chart_plan(ai_plan, profile)
    if _valid_chart_plan(ai_plan, profile):
        return ai_plan
    return _choose_chart_with_rules(topic, dataset, profile)


def score_story(
    topic: dict[str, Any],
    dataset: dict[str, Any],
    profile: dict[str, Any],
    chart_plan: dict[str, Any],
) -> float:
    rows = profile["n_rows"]
    columns = profile["n_columns"]
    data_quality = 0.35 + min(0.35, rows / 500) + min(0.2, columns / 25)
    if rows < 20:
        data_quality -= 0.25
    if chart_plan["chart_type"] in {"line", "area"}:
        chartability = 0.95
    elif chart_plan["chart_type"] in {"lollipop", "heatmap", "grouped_bar", "stacked_bar"}:
        chartability = 0.85
    elif chart_plan["chart_type"] in {"bar", "horizontal_bar"}:
        chartability = 0.8
    else:
        chartability = 0.6
    topic_score = float(topic.get("score", 0.5))
    dataset_score = float(dataset.get("dataset_score", 0.5))
    simplicity = 0.8 if chart_plan.get("x_column") and chart_plan.get("y_column") else 0.45
    variation = _variation_score(profile, chart_plan.get("y_column"))

    final = (
        0.25 * max(0.0, min(1.0, data_quality))
        + 0.2 * dataset_score
        + 0.2 * chartability
        + 0.2 * topic_score
        + 0.05 * simplicity
        + 0.1 * variation
    )
    return round(max(0.0, min(1.0, final)), 3)


def _variation_score(profile: dict[str, Any], y_column: Any) -> float:
    col = next((c for c in profile.get("columns", []) if c["name"] == y_column), None)
    if not col or col.get("semantic_type") != "number":
        return 0.5
    col_min = col.get("min")
    col_max = col.get("max")
    if col_min is None or col_max is None:
        return 0.5
    scale = max(abs(col_min), abs(col_max))
    if scale == 0:
        return 0.0
    spread = (col_max - col_min) / scale
    return round(min(1.0, spread), 3)


def quality_gate(df: pd.DataFrame, chart_plan: dict[str, Any], score: float, min_score: float) -> tuple[bool, str]:
    if len(df) < 8:
        return False, "Too few rows for a useful public chart."
    if score < min_score:
        return False, f"Story score {score} is below threshold {min_score}."
    if chart_plan.get("y_column") not in df.columns:
        return False, "Y column is missing from the loaded dataframe."
    if chart_plan.get("chart_type") != "histogram" and chart_plan.get("x_column") not in df.columns:
        return False, "X column is missing from the loaded dataframe."
    if chart_plan.get("chart_type") in CATEGORY_REQUIRED_TYPES and chart_plan.get("category_column") not in df.columns:
        return False, "Category column is missing for a chart type that requires one."
    if not _has_renderable_data(df, chart_plan):
        return False, "Chart plan does not produce renderable numeric data."
    if _is_flat_data(df, chart_plan):
        return False, "Y values are nearly constant; the chart would not be interesting."
    return True, "OK"


def _is_flat_data(df: pd.DataFrame, chart_plan: dict[str, Any]) -> bool:
    if chart_plan.get("aggregation") == "count":
        return False
    values = pd.to_numeric(df[chart_plan.get("y_column")], errors="coerce").dropna()
    if len(values) < 3:
        return False
    if values.nunique() <= 2:
        return True
    scale = max(abs(float(values.min())), abs(float(values.max())))
    if scale == 0:
        return True
    return (float(values.max()) - float(values.min())) / scale < 0.02


def _choose_chart_with_rules(topic: dict[str, Any], dataset: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    columns = profile["columns"]
    numeric = [col for col in columns if col["semantic_type"] == "number" and col["missing_ratio"] < 0.6]
    temporal = [col for col in columns if col["semantic_type"] == "time" and col["missing_ratio"] < 0.6]
    categories = [
        col
        for col in columns
        if col["semantic_type"] == "category" and 2 <= col["n_unique"] <= 30 and col["missing_ratio"] < 0.6
    ]

    y_col = _pick_best_numeric(numeric)
    if not y_col:
        y_col = next((col for col in numeric), None)
    if not y_col:
        first = columns[0]["name"]
        return {
            "title": f"{_theme_name(topic)}: Datenüberblick",
            "subtitle": "Automatisch erzeugte Übersicht aus einem offiziellen Open-Data-Datensatz.",
            "chart_type": "bar",
            "x_column": first,
            "y_column": first,
            "category_column": None,
            "aggregation": "count",
            "highlight": {
                "mode": "top",
                "n": 1,
                "reason": "Häufigste Kategorie hervorgehoben.",
            },
            "why_interesting": "Der Datensatz ist passend zum Thema, aber die Spalten eignen sich nur begrenzt für Kennzahlen.",
            "caveat": "Die Grafik ist explorativ und sollte vor einer Veröffentlichung fachlich geprüft werden.",
            "source_note": f"{dataset.get('organization', 'opendata.swiss')}: {dataset.get('title', '')}",
        }

    if temporal:
        x_col = temporal[0]["name"]
        chart_type = "line"
        category_col = None
    elif categories:
        x_col = categories[0]["name"]
        chart_type = "horizontal_bar"
        category_col = x_col
    else:
        x_col = y_col["name"]
        chart_type = "histogram"
        category_col = None

    y_name = y_col["name"]
    dataset_title = str(dataset.get("title") or "").strip()
    if y_name.lower().startswith(("column_", "wert ")) and dataset_title:
        title = dataset_title
    else:
        title = f"{dataset_title or topic['topic']}: {y_name[:1].upper() + y_name[1:]}"
    return {
        "title": title[:95],
        "subtitle": f"Aktuelles Thema in den Schweizer News: {_theme_name(topic)}. Passende offizielle Daten, automatisch ausgewählt.",
        "chart_type": chart_type,
        "x_column": x_col,
        "y_column": y_name,
        "category_column": category_col,
        "aggregation": "mean",
        "highlight": _default_highlight(chart_type),
        "why_interesting": "Die Daten enthalten eine robuste Kennzahl und eine einfache Vergleichsachse.",
        "caveat": "Automatisch erzeugte Analyse: Aussage und Kontext vor Publikation prüfen.",
        "source_note": f"{dataset.get('organization', 'opendata.swiss')}: {dataset.get('title', '')}",
    }


def _try_ai_chart_plan(
    topic: dict[str, Any],
    dataset: dict[str, Any],
    profile: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    prompt = (
        "Du bist Datenjournalist:in und gestaltest die Grafik fuer eine Schweizer Datenstory. "
        "Waehle die Chartform nach der AUSSAGE der Story, nicht nach Bequemlichkeit:\n"
        "- Entwicklung ueber Zeit -> line (mehrere Serien via category_column) oder area (eine Serie)\n"
        "- Rangliste / Vergleich von Kategorien -> horizontal_bar oder lollipop\n"
        "- Eine Kategorie ist die Story, der Rest ist Kontext -> highlight mode 'category' mit values\n"
        "- Vergleich weniger Gruppen ueber wenige Zeitpunkte -> grouped_bar\n"
        "- Zusammensetzung / Teil-vom-Ganzen -> stacked_bar\n"
        "- Muster ueber zwei Dimensionen (z.B. Kanton x Jahr) -> heatmap\n"
        "- Zusammenhang zweier Kennzahlen -> scatter; Verteilung -> histogram\n"
        "Gib ausschliesslich JSON mit diesen Feldern zurueck: title, subtitle, chart_type, "
        "x_column, y_column, category_column, x_label, y_label, unit, aggregation, sort, top_n, "
        "filters, highlight, annotations, reference_line, why_interesting, caveat, source_note.\n"
        "Erlaubte chart_type: line, area, bar, horizontal_bar, lollipop, grouped_bar, stacked_bar, "
        "heatmap, histogram, scatter. Erlaubte sort: value_desc, value_asc, x_asc, none. "
        "x_label und y_label sind PFLICHT: kurze, verstaendliche deutsche Achsenbeschriftungen "
        "(nie rohe Spaltencodes wie GEO_NAME oder VALUE). unit ist die Einheit der Kennzahl (z.B. CHF, %, Personen). "
        "filters ist eine optionale Liste von Objekten mit column und value (value exakt wie in sample_values), "
        "um eine klare Aussage zu isolieren (z.B. nur Total-Zeilen oder eine Region). "
        "highlight ist ein Objekt mit mode, n, values und reason; mode: max, min, top, bottom, category, none. "
        "Bei mode 'category' enthaelt values die hervorzuhebenden Kategorienwerte. "
        "annotations ist eine optionale Liste (max 2) von Objekten mit x (exakter Achsenwert) und text "
        "(kurze Einordnung, z.B. ein historisches Ereignis zu einem Jahr) - nutze sie, wenn sie die Story staerkt. "
        "reference_line ist optional: {mode: 'mean'} oder {mode: 'value', value: zahl, label: text}. "
        "Ein Titel soll die Kernaussage formulieren ('X steigt seit Y'), nicht den Datensatz beschreiben. "
        "Waehle die Kennzahl mit dem staerksten Bezug zum aktuellen Thema. "
        "Verwende nur Spalten aus dem Profil und behaupte keine Kausalitaet.\n\n"
        f"Thema:\n{json.dumps(topic, ensure_ascii=False)}\n\n"
        f"Datensatz:\n{json.dumps(_compact_dataset(dataset), ensure_ascii=False)}\n\n"
        f"Profil:\n{json.dumps(profile, ensure_ascii=False)}"
    )
    return parse_json_object(generate(settings, prompt, "chart_plan"))


CHART_TYPES = {
    "line", "area", "bar", "horizontal_bar", "lollipop", "grouped_bar",
    "stacked_bar", "heatmap", "histogram", "scatter",
}
CATEGORY_REQUIRED_TYPES = {"grouped_bar", "stacked_bar", "heatmap"}


def _valid_chart_plan(plan: dict[str, Any], profile: dict[str, Any]) -> bool:
    if not plan:
        return False
    columns = {col["name"] for col in profile["columns"]}
    semantic_types = {col["name"]: col["semantic_type"] for col in profile["columns"]}
    unique_counts = {col["name"]: col.get("n_unique", 0) for col in profile["columns"]}
    chart_type = plan.get("chart_type")
    if chart_type not in CHART_TYPES:
        return False
    if plan.get("y_column") not in columns:
        return False
    if chart_type != "histogram" and plan.get("x_column") not in columns:
        return False
    category = plan.get("category_column")
    if category and category not in columns:
        return False
    if chart_type in CATEGORY_REQUIRED_TYPES:
        if not category or category == plan.get("x_column"):
            return False
        if semantic_types.get(category) == "number" and unique_counts.get(category, 0) > 12:
            return False

    x_column = plan.get("x_column")
    y_column = plan.get("y_column")
    x_type = semantic_types.get(x_column)
    y_type = semantic_types.get(y_column)

    if chart_type == "histogram":
        return y_type == "number"
    if chart_type == "scatter":
        return x_type == "number" and y_type == "number"
    return y_type == "number" and x_type in {"category", "time", "number"}


def _normalize_chart_plan(plan: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    if not plan:
        return {}

    normalized = plan.copy()
    semantic_types = {col["name"]: col["semantic_type"] for col in profile["columns"]}
    x_column = normalized.get("x_column")
    y_column = normalized.get("y_column")
    x_type = semantic_types.get(x_column)
    y_type = semantic_types.get(y_column)
    chart_type = normalized.get("chart_type")

    if chart_type in {"bar", "horizontal_bar", "line"} and x_type == "number" and y_type in {"category", "time"}:
        normalized["x_column"] = y_column
        normalized["y_column"] = x_column
        x_column = normalized["x_column"]

    if normalized.get("x_column") == "Kategorie" and _has_column(profile, "Kategorie Detail"):
        category_profile = _column_profile(profile, "Kategorie")
        if category_profile and category_profile.get("n_unique", 0) < profile.get("n_rows", 0):
            normalized["x_column"] = "Kategorie Detail"
            x_column = "Kategorie Detail"

    if chart_type == "horizontal_bar" and not normalized.get("category_column"):
        if semantic_types.get(x_column) == "category":
            normalized["category_column"] = x_column
    elif chart_type == "horizontal_bar" and normalized.get("category_column") == "Kategorie" and x_column == "Kategorie Detail":
        normalized["category_column"] = "Kategorie Detail"

    if normalized.get("aggregation") in {None, "", "null"}:
        normalized["aggregation"] = "mean"
    normalized["highlight"] = _normalize_highlight(normalized.get("highlight"), chart_type)
    normalized["filters"] = _normalize_filters(normalized.get("filters"), profile)
    if normalized.get("sort") not in {"value_desc", "value_asc", "x_asc", "none", None}:
        normalized["sort"] = None
    try:
        normalized["top_n"] = min(30, max(3, int(normalized["top_n"]))) if normalized.get("top_n") else None
    except (TypeError, ValueError):
        normalized["top_n"] = None
    for label_key in ("x_label", "y_label", "unit"):
        value = normalized.get(label_key)
        normalized[label_key] = str(value).strip()[:80] if value else None
    if not isinstance(normalized.get("annotations"), list):
        normalized["annotations"] = []
    if not isinstance(normalized.get("reference_line"), dict):
        normalized["reference_line"] = None

    return normalized


def _normalize_filters(filters: Any, profile: dict[str, Any]) -> list[dict[str, str]]:
    if not isinstance(filters, list):
        return []
    columns = {col["name"] for col in profile.get("columns", [])}
    normalized = []
    for item in filters:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column", ""))
        value = item.get("value")
        if column in columns and value is not None:
            normalized.append({"column": column, "value": str(value)})
    return normalized[:3]


def compute_chart_facts(df: pd.DataFrame, chart_plan: dict[str, Any]) -> dict[str, Any]:
    """Berechnet die Fakten aus den tatsaechlich geplotteten Daten.

    Diese Zahlen sind die einzige erlaubte Quelle fuer Aussagen in Titel und
    Caption - so koennen Titelbehauptungen gegen die Daten geprueft werden.
    """
    y = chart_plan.get("y_column")
    chart_type = chart_plan.get("chart_type", "bar")
    key = chart_plan.get("x_column")
    if chart_type in {"horizontal_bar", "lollipop"} and chart_plan.get("category_column"):
        key = chart_plan.get("category_column")

    facts: dict[str, Any] = {}
    if y not in df.columns:
        return facts
    values = pd.to_numeric(df[y], errors="coerce")
    if values.notna().sum() == 0:
        return facts

    if key and key in df.columns and key != y:
        work = df[[key]].copy()
        work["__value"] = values
        work = work.dropna()
        aggregation = chart_plan.get("aggregation", "mean")
        grouped = work.groupby(key)["__value"]
        series = {"sum": grouped.sum, "median": grouped.median, "count": grouped.count}.get(
            aggregation, grouped.mean
        )()
        ranked = series.sort_values(ascending=False)
        facts["n_groups"] = int(len(ranked))
        facts["top"] = [
            {"label": str(label), "value": round(float(value), 2)} for label, value in ranked.head(3).items()
        ]
        facts["bottom"] = [
            {"label": str(label), "value": round(float(value), 2)} for label, value in ranked.tail(1).items()
        ]

        highlight = chart_plan.get("highlight") or {}
        if highlight.get("mode") == "category" and highlight.get("values"):
            wanted = {str(v).strip().lower() for v in highlight["values"]}
            highlighted = []
            for rank, (label, value) in enumerate(ranked.items(), start=1):
                if str(label).strip().lower() in wanted:
                    highlighted.append(
                        {"label": str(label), "value": round(float(value), 2), "rank": rank, "of": len(ranked)}
                    )
            facts["highlighted"] = highlighted

        if chart_type in {"line", "area"}:
            ordered = series.sort_index()
            if len(ordered) >= 2:
                first_value, last_value = float(ordered.iloc[0]), float(ordered.iloc[-1])
                facts["first"] = {"label": str(ordered.index[0]), "value": round(first_value, 2)}
                facts["last"] = {"label": str(ordered.index[-1]), "value": round(last_value, 2)}
                if first_value:
                    facts["change_pct"] = round((last_value - first_value) / abs(first_value) * 100, 1)

    clean = values.dropna()
    facts["value_mean"] = round(float(clean.mean()), 2)
    facts["value_min"] = round(float(clean.min()), 2)
    facts["value_max"] = round(float(clean.max()), 2)
    return facts


SUPERLATIVE_PATTERN = re.compile(
    r"h(ö|oe)chst|gr(ö|oe)sst|meist|spitzenreiter|nummer\s*1|rekord|am\s+h(ö|oe)chsten",
    re.IGNORECASE,
)


def ground_title(chart_plan: dict[str, Any], facts: dict[str, Any]) -> None:
    """Ersetzt Superlativ-Titel, die die berechneten Daten nicht stuetzen."""
    highlighted = facts.get("highlighted") or []
    if not highlighted:
        return
    title = str(chart_plan.get("title", ""))
    if not SUPERLATIVE_PATTERN.search(title):
        return
    if min(item["rank"] for item in highlighted) == 1:
        return
    labels = ", ".join(item["label"] for item in highlighted)
    metric = chart_plan.get("y_label") or chart_plan.get("y_column") or ""
    chart_plan["title"] = f"{metric} im Vergleich: Wo {labels} steht"[:95]
    chart_plan["title_grounding_note"] = (
        f"Original-Titel entfernt: Superlativ nicht durch Daten gedeckt (Rang "
        f"{min(item['rank'] for item in highlighted)} von {highlighted[0]['of']})."
    )


def ground_annotations(chart_plan: dict[str, Any], facts: dict[str, Any]) -> None:
    """Entfernt Annotationen mit Superlativ-Behauptungen, die nicht auf dem Spitzenwert sitzen."""
    annotations = chart_plan.get("annotations")
    top = facts.get("top") or []
    if not isinstance(annotations, list) or not top:
        return
    top_label = str(top[0]["label"]).strip().lower()
    kept = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        text = str(annotation.get("text", ""))
        target = str(annotation.get("x", "")).strip().lower()
        if SUPERLATIVE_PATTERN.search(text) and target != top_label:
            continue
        kept.append(annotation)
    chart_plan["annotations"] = kept


def apply_chart_filters(df: pd.DataFrame, chart_plan: dict[str, Any]) -> pd.DataFrame:
    filters = chart_plan.get("filters") or []
    filtered = df
    applied: list[dict[str, str]] = []
    for item in filters:
        column = item.get("column")
        value = str(item.get("value", "")).strip().lower()
        if column not in filtered.columns:
            continue
        mask = filtered[column].astype(str).str.strip().str.lower() == value
        candidate = filtered[mask]
        if len(candidate) >= 3:
            filtered = candidate
            applied.append(item)
    chart_plan["filters"] = applied
    return filtered.reset_index(drop=True)


def _theme_name(topic: dict[str, Any]) -> str:
    """'Wohnen: Tages-anzeiger / Wohnung' -> 'Wohnen' (nur das lesbare Oberthema)."""
    return str(topic.get("topic", "")).split(":")[0].split(" / ")[0].strip() or "Schweiz"


def _has_column(profile: dict[str, Any], column_name: str) -> bool:
    return any(col["name"] == column_name for col in profile.get("columns", []))


def _column_profile(profile: dict[str, Any], column_name: str) -> dict[str, Any] | None:
    return next((col for col in profile.get("columns", []) if col["name"] == column_name), None)


def _default_highlight(chart_type: str) -> dict[str, Any]:
    if chart_type in {"bar", "horizontal_bar", "line"}:
        return {
            "mode": "max",
            "n": 1,
            "reason": "Höchster Wert hervorgehoben.",
        }
    if chart_type == "histogram":
        return {
            "mode": "none",
            "n": 0,
            "reason": "Histogramme werden im MVP nicht einzeln hervorgehoben.",
        }
    return {
        "mode": "max",
        "n": 1,
        "reason": "Auffälligster Wert hervorgehoben.",
    }


def _normalize_highlight(highlight: Any, chart_type: str) -> dict[str, Any]:
    if not isinstance(highlight, dict):
        return _default_highlight(chart_type)

    mode = str(highlight.get("mode", "max")).lower()
    if mode not in {"max", "min", "top", "bottom", "category", "none"}:
        mode = "max"
    values = highlight.get("values")
    if not isinstance(values, list):
        values = []
    values = [str(v).strip() for v in values if str(v).strip()][:5]
    if mode == "category" and not values:
        mode = "max"
    try:
        n = int(highlight.get("n", 1))
    except (TypeError, ValueError):
        n = 1

    return {
        "mode": mode,
        "n": max(0, min(5, n)),
        "values": values,
        "reason": str(highlight.get("reason") or _default_highlight(chart_type)["reason"]),
    }


def _has_renderable_data(df: pd.DataFrame, chart_plan: dict[str, Any]) -> bool:
    y_column = chart_plan.get("y_column")
    if y_column not in df.columns:
        return False

    values = pd.to_numeric(df[y_column], errors="coerce")
    if values.notna().sum() == 0:
        return False

    chart_type = chart_plan.get("chart_type")
    if chart_type == "histogram":
        return values.notna().sum() >= 3

    x_column = chart_plan.get("x_column")
    if x_column not in df.columns:
        return False

    renderable = df[[x_column, y_column]].copy()
    renderable[y_column] = values
    renderable = renderable.dropna(subset=[x_column, y_column])
    return len(renderable) >= 3


def _looks_temporal(series: pd.Series, column_name: str) -> bool:
    sample = series.dropna().astype(str).head(30)
    if sample.empty:
        return False
    date_like = sample.str.match(r"^\d{4}([\-/.]\d{1,2}([\-/.]\d{1,2})?)?$").mean()
    swiss_date_like = sample.str.match(r"^\d{1,2}\.\d{1,2}\.\d{4}$").mean()
    if max(date_like, swiss_date_like) >= 0.65:
        return True
    # Der Spaltenname allein genuegt nicht: 'Periode' kann auch Textwerte wie
    # '1. semestre' oder 'Octobre' enthalten, die keine sortierbare Zeitachse ergeben.
    if re.search(r"(date|datum|jahr|year|periode|time)", column_name, re.IGNORECASE):
        return max(date_like, swiss_date_like) >= 0.4 or pd.to_numeric(series, errors="coerce").notna().mean() >= 0.65
    return False


def _pick_best_numeric(columns: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not columns:
        return None
    preferred = ["anzahl", "rate", "quote", "preis", "kosten", "wert", "total", "mean", "value"]
    for token in preferred:
        for column in columns:
            if token in column["name"].lower():
                return column
    return max(columns, key=lambda col: col.get("n_unique", 0))


def _compact_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": dataset.get("id"),
        "title": dataset.get("title"),
        "description": dataset.get("description", "")[:700],
        "organization": dataset.get("organization"),
        "license": dataset.get("license"),
        "url": dataset.get("url"),
    }

