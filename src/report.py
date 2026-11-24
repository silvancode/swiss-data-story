from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ai import generate, parse_json_object
from .config import Settings


def build_caption(
    topic: dict[str, Any],
    dataset: dict[str, Any],
    resource: dict[str, Any],
    chart_plan: dict[str, Any],
    story_score: float,
    settings: Settings,
    chart_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ai_caption = _try_ai_caption(topic, dataset, chart_plan, story_score, settings, chart_facts or {})
    if ai_caption:
        ai_caption["resource_url"] = resource.get("url")
        return ai_caption

    source_text = f"{dataset.get('organization', 'opendata.swiss')}: {dataset.get('title', '')}"
    return {
        "post_title": chart_plan.get("title", topic.get("topic", "Swiss Data Story")),
        "short_description": (
            f"Zum aktuellen Thema '{topic.get('topic')}' wurde ein passender offizieller "
            f"Datensatz auf opendata.swiss gefunden. Die Grafik zeigt eine automatisch "
            f"ausgewählte, einfache Darstellung der Spalte '{chart_plan.get('y_column')}'."
        ),
        "key_takeaways": [
            chart_plan.get("why_interesting", "Der Datensatz eignet sich für eine kompakte Datenstory."),
            f"Automatischer Story-Score: {story_score}.",
            "Die Ausgabe ist als Kandidat gedacht und sollte vor einer Publikation geprüft werden.",
        ],
        "caveat": chart_plan.get("caveat", "Keine kausale Aussage; Kontext und Datenstand prüfen."),
        "source_text": source_text,
        "alt_text": (
            f"Automatisch erzeugte Grafik mit dem Titel {chart_plan.get('title')}. "
            f"Quelle: {source_text}."
        ),
        "resource_url": resource.get("url"),
    }


def write_story_files(
    output_dir: Path,
    caption: dict[str, Any],
    topic: dict[str, Any],
    dataset: dict[str, Any],
    resource: dict[str, Any],
    chart_plan: dict[str, Any],
    report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    caption_path = output_dir / "caption.md"
    caption_path.write_text(_caption_markdown(caption), encoding="utf-8")

    sources = {
        "topic": topic,
        "dataset": _strip_heavy_resources(dataset),
        "resource": resource,
        "chart_plan": chart_plan,
    }
    (output_dir / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _caption_markdown(caption: dict[str, Any]) -> str:
    takeaways = "\n".join(f"- {item}" for item in caption.get("key_takeaways", []))
    return (
        f"# {caption.get('post_title', 'Swiss Data Story')}\n\n"
        f"{caption.get('short_description', '')}\n\n"
        f"## Kernaussagen\n{takeaways}\n\n"
        f"## Einordnung\n{caption.get('caveat', '')}\n\n"
        f"## Quelle\n{caption.get('source_text', '')}\n\n"
        f"Ressource: {caption.get('resource_url', '')}\n\n"
        f"## Alt-Text\n{caption.get('alt_text', '')}\n"
    )


def _try_ai_caption(
    topic: dict[str, Any],
    dataset: dict[str, Any],
    chart_plan: dict[str, Any],
    story_score: float,
    settings: Settings,
    chart_facts: dict[str, Any],
) -> dict[str, Any]:
    prompt = (
        "Erstelle eine sachliche deutsche Caption fuer eine Schweizer Datenvisualisierung. "
        "Gib ausschliesslich JSON mit post_title, short_description, key_takeaways, caveat, "
        "source_text und alt_text zurueck. "
        "Die 'berechneten Fakten' unten sind aus den tatsaechlich geplotteten Daten berechnet "
        "und sind die EINZIGE erlaubte Quelle fuer Zahlen- und Rangaussagen. Keine erfundenen "
        "Zahlen, keine Superlative, die dort nicht belegt sind, keine Kausalitaet.\n\n"
        f"Thema:\n{json.dumps(topic, ensure_ascii=False)}\n\n"
        f"Datensatz:\n{json.dumps(_strip_heavy_resources(dataset), ensure_ascii=False)}\n\n"
        f"Chartplan:\n{json.dumps(chart_plan, ensure_ascii=False)}\n\n"
        f"Berechnete Fakten:\n{json.dumps(chart_facts, ensure_ascii=False)}\n\n"
        f"Story score: {story_score}"
    )
    data = parse_json_object(generate(settings, prompt, "caption"))
    required = {"post_title", "short_description", "key_takeaways", "caveat", "source_text", "alt_text"}
    if not required.issubset(data):
        return {}
    return data



def _strip_heavy_resources(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dataset.items()
        if key not in {"resources"} and not (key == "loadable_resources" and isinstance(value, list) and len(value) > 5)
    }
