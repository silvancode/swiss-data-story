from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from io import BytesIO, StringIO
from typing import Any

import pandas as pd
import requests

from .ai import generate, parse_json_list
from .config import Settings

CKAN_SEARCH_URL = "https://opendata.swiss/api/3/action/package_search"
SUPPORTED_FORMATS = ("csv", "json", "xls", "xlsx")
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_DOWNLOAD_SECONDS = 60.0


def search_datasets_for_topic(topic: dict[str, Any], settings: Settings) -> list[dict[str, Any]]:
    queries = _queries_for_topic(topic)
    datasets: dict[str, dict[str, Any]] = {}

    for query in queries:
        params = {
            "q": query,
            "rows": settings.max_datasets_per_topic,
            "sort": "score desc, metadata_modified desc",
        }
        try:
            response = requests.get(CKAN_SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        for package in payload.get("result", {}).get("results", []):
            dataset = _normalize_dataset(package, query)
            if dataset["loadable_resources"]:
                datasets[dataset["id"]] = dataset

    ranked = rank_datasets(topic, list(datasets.values()), settings)
    return ranked[: settings.max_datasets_per_topic]


def rank_datasets(
    topic: dict[str, Any], datasets: list[dict[str, Any]], settings: Settings
) -> list[dict[str, Any]]:
    ai_ranked = _try_ai_dataset_ranking(topic, datasets, settings)
    if ai_ranked:
        by_id = {dataset["id"]: dataset for dataset in datasets}
        ranked = []
        for item in ai_ranked:
            dataset = by_id.get(item.get("dataset_id"))
            if not dataset:
                continue
            dataset = dataset.copy()
            dataset["dataset_score"] = _bounded_float(item.get("score", dataset["dataset_score"]))
            dataset["ranking_reason"] = item.get("reason", dataset.get("ranking_reason", "AI ranking"))
            ranked.append(dataset)
        if ranked:
            return sorted(ranked, key=lambda item: item["dataset_score"], reverse=True)

    return sorted(datasets, key=lambda item: item["dataset_score"], reverse=True)


def load_first_supported_resource(dataset: dict[str, Any]) -> tuple[dict[str, Any] | None, pd.DataFrame | None, str | None]:
    for resource in dataset.get("loadable_resources", []):
        url = resource.get("url")
        fmt = str(resource.get("format") or "").lower()
        if not url:
            continue
        try:
            df = load_resource(url, fmt)
        except Exception:
            continue
        if df is not None and len(df) >= 8 and len(df.columns) >= 2:
            return resource, df, None
    return None, None, "No supported resource could be loaded with enough rows and columns."


def download_limited(url: str, max_bytes: int = MAX_DOWNLOAD_BYTES, max_seconds: float = MAX_DOWNLOAD_SECONDS) -> bytes:
    """Laedt eine Datei mit Obergrenze fuer Groesse und Gesamtdauer.

    Der `timeout` von requests gilt nur pro Lesevorgang: Ein Server, der langsam
    tropfenweise liefert, wuerde den Lauf sonst beliebig lange blockieren.
    """
    started = time.monotonic()
    with requests.get(url, timeout=30, stream=True) as response:
        response.raise_for_status()
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > max_bytes:
            raise ValueError(f"Resource too large ({declared} bytes > {max_bytes}).")
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"Resource too large (> {max_bytes} bytes).")
            if time.monotonic() - started > max_seconds:
                raise TimeoutError(f"Download took longer than {max_seconds}s.")
            chunks.append(chunk)
    return b"".join(chunks)


def load_resource(url: str, file_format: str) -> pd.DataFrame | None:
    content = download_limited(url)
    fmt = (file_format or "").lower()
    lower_url = url.lower()

    if "csv" in fmt or lower_url.endswith(".csv"):
        text = _decode_text(content)
        return pd.read_csv(StringIO(text), sep=None, engine="python")

    if "json" in fmt or lower_url.endswith(".json"):
        return pd.read_json(BytesIO(content))

    if "xls" in fmt or lower_url.endswith((".xls", ".xlsx")):
        raw = pd.read_excel(BytesIO(content), header=None)
        return _normalize_spreadsheet_table(raw)

    return None


def _normalize_spreadsheet_table(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    raw = raw.reset_index(drop=True)
    raw.columns = range(len(raw.columns))
    if raw.empty:
        return raw

    data_start = _find_data_start(raw)
    if data_start <= 0:
        cleaned = raw.copy()
        cleaned.columns = [f"column_{idx + 1}" for idx in range(len(cleaned.columns))]
        return _coerce_numeric_columns(cleaned)

    header_block = raw.iloc[max(0, data_start - 5) : data_start]
    data = raw.iloc[data_start:].copy().dropna(axis=0, how="all")
    data.columns = _build_spreadsheet_columns(raw, header_block)
    data = data.dropna(axis=1, how="all")
    data = data.loc[:, ~pd.Index(data.columns).duplicated()]

    first_col = data.columns[0]
    data = data[data[first_col].notna()]
    data = data[
        ~data[first_col].astype(str).str.contains(r"^(?:quelle|source|©|copyright|datenstand)", case=False, regex=True)
    ]
    data = _coerce_numeric_columns(data.reset_index(drop=True))
    data = _preserve_spreadsheet_group_headers(data)
    if len(data.columns) > 1:
        data = data[data.iloc[:, 1:].notna().any(axis=1)]
    return data.reset_index(drop=True)


def _preserve_spreadsheet_group_headers(data: pd.DataFrame) -> pd.DataFrame:
    if len(data.columns) < 2:
        return data

    first_col = data.columns[0]
    value_cols = [column for column in data.columns[1:] if pd.to_numeric(data[column], errors="coerce").notna().any()]
    if not value_cols:
        return data

    has_values = data[value_cols].apply(lambda row: pd.to_numeric(row, errors="coerce").notna().any(), axis=1)
    group_headers = data[first_col].notna() & ~has_values
    if group_headers.sum() < 2:
        return data

    current_group = "Gesamt"
    groups: list[str | None] = []
    keep_rows: list[bool] = []

    for _, row in data.iterrows():
        label = _clean_text(row[first_col])
        row_has_values = pd.to_numeric(row[value_cols], errors="coerce").notna().any()
        if label and not row_has_values:
            current_group = label
            groups.append(None)
            keep_rows.append(False)
            continue
        groups.append(current_group)
        keep_rows.append(bool(row_has_values))

    cleaned = data.loc[keep_rows].copy().reset_index(drop=True)
    row_groups = [group for group, keep in zip(groups, keep_rows) if keep]
    if not row_groups:
        return data

    cleaned.insert(0, "Merkmal", row_groups)
    cleaned.insert(
        1,
        "Kategorie Detail",
        [
            _combine_group_and_category(group, category)
            for group, category in zip(cleaned["Merkmal"], cleaned[first_col])
        ],
    )
    return cleaned


def _combine_group_and_category(group: Any, category: Any) -> str:
    group_text = _clean_text(group)
    category_text = _clean_text(category)
    if not group_text or group_text == "Gesamt":
        return category_text
    return f"{group_text} - {category_text}"


def _find_data_start(raw: pd.DataFrame) -> int:
    for idx in range(len(raw)):
        row = raw.iloc[idx]
        numeric_count = pd.to_numeric(row, errors="coerce").notna().sum()
        text_count = row.astype(str).str.strip().replace("nan", "").astype(bool).sum()
        first_value = row.iloc[0]
        if pd.notna(first_value) and numeric_count >= 2 and text_count >= 3:
            return idx
    return 0


def _build_spreadsheet_columns(raw: pd.DataFrame, header_block: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    title = _clean_text(raw.iloc[0, 0]) if pd.notna(raw.iloc[0, 0]) else "Kategorie"

    for col_idx in range(raw.shape[1]):
        if col_idx == 0:
            columns.append(_short_category_name(title))
            continue
        parts = []
        for value in header_block.iloc[:, col_idx].tolist():
            if pd.isna(value):
                continue
            text = _clean_text(value)
            if text and text.lower() != "nan" and text not in parts:
                parts.append(text)
        if not parts:
            parts = [f"Wert {col_idx}"]
        name = " - ".join(parts)
        name = name.replace("Durch-schnittlicher", "Durchschnittlicher")
        name = name.replace("\n", " ")
        columns.append(name[:120])
    return columns


def _short_category_name(title: str) -> str:
    lowered = title.lower()
    if "kanton" in lowered:
        return "Kanton"
    if "gemeinde" in lowered:
        return "Gemeinde"
    if "jahr" in lowered:
        return "Jahr"
    return "Kategorie"


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype != object:
            continue
        normalized = (
            cleaned[column]
            .astype(str)
            .str.replace("'", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        numeric = pd.to_numeric(normalized, errors="coerce")
        if numeric.notna().mean() >= 0.65:
            cleaned[column] = numeric
    return cleaned


def _queries_for_topic(topic: dict[str, Any]) -> list[str]:
    raw_terms = list(topic.get("keywords", [])) + [topic.get("topic", "")]
    terms = []
    for value in raw_terms:
        cleaned = re.sub(r"[^\w\s\-äöüÄÖÜéèà]", " ", str(value))
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned and cleaned.lower() not in {t.lower() for t in terms}:
            terms.append(cleaned)
    return terms[:5]


def _normalize_dataset(package: dict[str, Any], matched_query: str) -> dict[str, Any]:
    resources = package.get("resources", []) or []
    loadable = [resource for resource in resources if _is_supported_resource(resource)]
    title = _localized_text(package.get("title")) or package.get("name") or "Untitled dataset"
    notes = _clean_text(_localized_text(package.get("notes", "")))
    organization = package.get("organization") or {}
    organization_title = _localized_text(organization.get("title")) or organization.get("name") or "Unknown"
    tags = [_localized_text(tag.get("display_name") or tag.get("name")) for tag in package.get("tags", []) if tag]

    dataset = {
        "id": package.get("id") or package.get("name"),
        "name": package.get("name"),
        "title": _clean_text(title),
        "description": notes[:1500],
        "organization": _clean_text(organization_title),
        "metadata_modified": package.get("metadata_modified"),
        "license": package.get("license_id") or package.get("license_title"),
        "url": package.get("url") or f"https://opendata.swiss/dataset/{package.get('name', '')}",
        "tags": [tag for tag in tags if tag][:12],
        "resources": resources,
        "loadable_resources": loadable,
        "matched_query": matched_query,
    }
    dataset["dataset_score"] = _score_dataset(dataset, matched_query)
    dataset["ranking_reason"] = "Rule-based score from query match, supported formats, recency metadata and official source."
    return dataset


def _score_dataset(dataset: dict[str, Any], query: str) -> float:
    haystack = " ".join(
        [
            dataset.get("title", ""),
            dataset.get("description", ""),
            dataset.get("organization", ""),
            " ".join(dataset.get("tags", [])),
        ]
    ).lower()
    query_terms = [term.lower() for term in re.findall(r"\w{4,}", query)]
    match_score = sum(1 for term in query_terms if term in haystack) / max(1, len(query_terms))
    resource_score = min(1.0, len(dataset.get("loadable_resources", [])) / 3)
    official_score = 1.0 if any(token in dataset.get("organization", "").lower() for token in ["bund", "federal", "bfs", "admin"]) else 0.65
    recency_score = _recency_score(dataset.get("metadata_modified"))
    return round(0.42 * match_score + 0.25 * resource_score + 0.2 * official_score + 0.13 * recency_score, 3)


def _recency_score(metadata_modified: Any) -> float:
    if not metadata_modified:
        return 0.4
    try:
        modified = datetime.fromisoformat(str(metadata_modified).replace("Z", "+00:00"))
    except ValueError:
        return 0.45
    if modified.tzinfo is None:
        modified = modified.replace(tzinfo=UTC)
    age_days = (datetime.now(UTC) - modified).days
    if age_days <= 90:
        return 1.0
    if age_days <= 365:
        return 0.85
    if age_days <= 3 * 365:
        return 0.6
    return 0.35


def _try_ai_dataset_ranking(
    topic: dict[str, Any], datasets: list[dict[str, Any]], settings: Settings
) -> list[dict[str, Any]]:
    if not datasets:
        return []

    compact = [
        {
            "id": item["id"],
            "title": item["title"],
            "description": item["description"][:500],
            "organization": item["organization"],
            "formats": [resource.get("format") for resource in item.get("loadable_resources", [])[:5]],
        }
        for item in datasets[:12]
    ]
    prompt = (
        "Ranke Schweizer Open-Data-Datensaetze fuer eine sachliche Datenstory. "
        "Gib ausschliesslich JSON zurueck: eine Liste mit dataset_id, score und reason. "
        "Bevorzuge offizielle, tabellarische, aktuelle und visualisierbare Daten.\n\n"
        f"Thema:\n{json.dumps(topic, ensure_ascii=False)}\n\n"
        f"Datensaetze:\n{json.dumps(compact, ensure_ascii=False)}"
    )
    return parse_json_list(generate(settings, prompt, "dataset_ranking"), wrapper_keys=("ranked_datasets", "datasets"))


def _is_supported_resource(resource: dict[str, Any]) -> bool:
    fmt = str(resource.get("format") or "").lower()
    url = str(resource.get("url") or "").lower()
    if not url.startswith(("http://", "https://")):
        return False
    return any(token in fmt for token in SUPPORTED_FORMATS) or url.endswith(SUPPORTED_FORMATS)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _localized_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("de", "en", "fr", "it"):
            if value.get(key):
                return str(value[key])
        for nested in value.values():
            if nested:
                return str(nested)
        return ""
    return str(value or "")


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _bounded_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return round(max(0.0, min(1.0, number)), 3)
