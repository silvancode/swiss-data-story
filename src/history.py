from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_story_history(output_dir: Path, current_output_dir: Path, limit: int) -> list[dict[str, Any]]:
    reports = [
        path
        for path in output_dir.glob("*/*/run_report.json")
        if current_output_dir not in path.parents
    ]
    reports.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    history: list[dict[str, Any]] = []
    for report_path in reports[:limit]:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        for attempt in report.get("attempts", []):
            if not attempt.get("success"):
                continue
            history.append(
                {
                    "run_date": report.get("run_date"),
                    "run_id": report.get("run_id"),
                    "report_path": str(report_path),
                    "signature": build_story_signature(
                        attempt.get("topic", {}),
                        {
                            "id": attempt.get("dataset_id"),
                            "title": attempt.get("dataset_title"),
                        },
                        {"url": attempt.get("resource_url")},
                        attempt.get("chart_plan", {}),
                    ),
                }
            )
    return history[:limit]


def build_story_signature(
    topic: dict[str, Any],
    dataset: dict[str, Any],
    resource: dict[str, Any],
    chart_plan: dict[str, Any],
) -> dict[str, Any]:
    topic_text = " ".join(
        [
            str(topic.get("topic", "")),
            str(topic.get("summary", "")),
            " ".join(str(keyword) for keyword in topic.get("keywords", [])),
        ]
    )
    dataset_text = " ".join(
        [
            str(dataset.get("title", "")),
            str(dataset.get("description", "")),
            str(dataset.get("organization", "")),
        ]
    )
    chart_text = " ".join(
        [
            str(chart_plan.get("title", "")),
            str(chart_plan.get("x_column", "")),
            str(chart_plan.get("y_column", "")),
            str(chart_plan.get("chart_type", "")),
        ]
    )
    return {
        "topic_id": topic.get("id"),
        "topic": topic.get("topic"),
        "dataset_id": dataset.get("id"),
        "dataset_title": dataset.get("title"),
        "resource_url": resource.get("url"),
        "chart_type": chart_plan.get("chart_type"),
        "x_column": chart_plan.get("x_column"),
        "y_column": chart_plan.get("y_column"),
        "tokens": sorted(_tokens(f"{topic_text} {dataset_text} {chart_text}")),
    }


def most_similar_story(
    candidate: dict[str, Any], history: list[dict[str, Any]]
) -> tuple[float, dict[str, Any] | None]:
    best_score = 0.0
    best_match = None

    for item in history:
        score = story_similarity(candidate, item["signature"])
        if score > best_score:
            best_score = score
            best_match = item

    return round(best_score, 3), best_match


def story_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    score = 0.0
    # Fehlende IDs/URLs (None == None) duerfen nicht als "gleiche Story" zaehlen.
    same_resource = bool(left.get("resource_url")) and left.get("resource_url") == right.get("resource_url")
    same_dataset = bool(left.get("dataset_id")) and left.get("dataset_id") == right.get("dataset_id")

    if same_resource:
        score = max(score, 0.72)
    if same_dataset:
        score = max(score, 0.68)
    if same_dataset and left.get("y_column") == right.get("y_column"):
        score = max(score, 0.94)
    if (
        same_resource
        and left.get("chart_type") == right.get("chart_type")
        and left.get("x_column") == right.get("x_column")
        and left.get("y_column") == right.get("y_column")
    ):
        score = max(score, 1.0)

    left_tokens = set(left.get("tokens", []))
    right_tokens = set(right.get("tokens", []))
    if left_tokens and right_tokens:
        jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        score = max(score, min(0.92, jaccard))

    return score


def _tokens(value: str) -> set[str]:
    words = re.findall(r"[a-zA-ZäöüÄÖÜéèàÉÈÀ0-9]{4,}", value.lower())
    stopwords = {
        "schweiz",
        "swiss",
        "suisse",
        "daten",
        "data",
        "quelle",
        "total",
        "durchschnittlicher",
    }
    return {word for word in words if word not in stopwords}
