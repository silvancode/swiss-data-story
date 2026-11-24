from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from .ai import collected_ai_errors
from .analyze import (
    apply_chart_filters,
    choose_chart_plan,
    compute_chart_facts,
    ground_annotations,
    ground_title,
    profile_dataframe,
    quality_gate,
    score_story,
)
from .config import load_settings
from .datasets import load_first_supported_resource, search_datasets_for_topic
from .history import build_story_signature, load_story_history, most_similar_story
from .report import build_caption, write_story_files
from .topics import build_topic_candidates, fetch_news_items, run_started_at
from .visualize import build_chart


def create_run_output_dir(base_output_dir: Path, run_date: str) -> tuple[str, Path]:
    run_id = datetime.now().strftime("%H%M%S")
    day_dir = base_output_dir / run_date
    output_dir = day_dir / run_id

    suffix = 2
    while output_dir.exists():
        output_dir = day_dir / f"{run_id}-{suffix}"
        suffix += 1

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.name, output_dir


def main() -> None:
    load_dotenv()
    settings = load_settings()
    run_date = date.today().isoformat()
    run_id, output_dir = create_run_output_dir(settings.output_dir, run_date)

    report = {
        "started_at": run_started_at(),
        "run_date": run_date,
        "run_id": run_id,
        "output_dir": str(output_dir),
        "ai_provider": settings.ai_provider,
        "status": "running",
        "attempts": [],
    }
    story_history = load_story_history(settings.output_dir, output_dir, settings.history_limit)
    report["history_items_checked"] = len(story_history)

    print("1. Fetching current Swiss topics...")
    news_items = fetch_news_items()
    topics = build_topic_candidates(news_items, settings)
    report["news_items_count"] = len(news_items)
    report["topics"] = topics

    for topic in topics:
        print(f"2. Searching opendata.swiss datasets for: {topic['topic']}")
        datasets = search_datasets_for_topic(topic, settings)
        if not datasets:
            report["attempts"].append({"topic": topic, "success": False, "reason": "No datasets found."})
            continue

        for dataset in datasets:
            attempt = {
                "topic": topic,
                "dataset_id": dataset.get("id"),
                "dataset_title": dataset.get("title"),
                "dataset_score": dataset.get("dataset_score"),
                "success": False,
            }
            print(f"3. Trying dataset: {dataset.get('title')}")
            resource, df, load_error = load_first_supported_resource(dataset)
            if df is None or resource is None:
                attempt["reason"] = load_error
                report["attempts"].append(attempt)
                continue

            print("4. Profiling data and choosing chart...")
            profile = profile_dataframe(df)
            chart_plan = choose_chart_plan(topic, dataset, profile, settings)
            df = apply_chart_filters(df, chart_plan)
            chart_facts = compute_chart_facts(df, chart_plan)
            ground_title(chart_plan, chart_facts)
            ground_annotations(chart_plan, chart_facts)
            story_score = score_story(topic, dataset, profile, chart_plan)
            ok, reason = quality_gate(df, chart_plan, story_score, settings.min_story_score)
            story_signature = build_story_signature(topic, dataset, resource, chart_plan)
            similarity, similar_story = most_similar_story(story_signature, story_history)

            attempt.update(
                {
                    "resource_url": resource.get("url"),
                    "profile": profile,
                    "chart_plan": chart_plan,
                    "chart_facts": chart_facts,
                    "story_score": story_score,
                    "quality_gate": reason,
                    "story_signature": story_signature,
                    "novelty_similarity": similarity,
                    "novelty_threshold": settings.max_story_similarity,
                }
            )
            if similar_story:
                attempt["most_similar_previous_story"] = {
                    "run_date": similar_story.get("run_date"),
                    "run_id": similar_story.get("run_id"),
                    "report_path": similar_story.get("report_path"),
                    "dataset_title": similar_story["signature"].get("dataset_title"),
                    "chart_type": similar_story["signature"].get("chart_type"),
                    "x_column": similar_story["signature"].get("x_column"),
                    "y_column": similar_story["signature"].get("y_column"),
                }

            if not ok:
                attempt["reason"] = reason
                report["attempts"].append(attempt)
                continue

            if similarity >= settings.max_story_similarity:
                attempt["reason"] = (
                    f"Story too similar to previous run "
                    f"({similarity} >= {settings.max_story_similarity})."
                )
                report["attempts"].append(attempt)
                continue

            print("5. Building chart and report files...")
            chart_path = build_chart(df, chart_plan, output_dir / "chart.png")
            caption = build_caption(topic, dataset, resource, chart_plan, story_score, settings, chart_facts)
            attempt["success"] = True
            attempt["chart_path"] = str(chart_path)
            attempt["caption"] = caption
            report["attempts"].append(attempt)
            report["status"] = "success"
            report["selected_story"] = {
                "topic": topic,
                "dataset": dataset.get("title"),
                "resource_url": resource.get("url"),
                "story_score": story_score,
                "chart_path": str(chart_path),
            }

            report["ai_errors"] = collected_ai_errors()
            write_story_files(output_dir, caption, topic, dataset, resource, chart_plan, report)
            print(f"Done. Story written to {output_dir}")
            return

    report["status"] = "no_story_found"
    report["ai_errors"] = collected_ai_errors()
    (output_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"No usable story found. Report written to {output_dir / 'run_report.json'}")


if __name__ == "__main__":
    main()
