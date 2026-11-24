from __future__ import annotations

import calendar
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from html import unescape
from typing import Any

import feedparser
import requests

from .ai import generate, parse_json_list
from .config import SWISS_FEEDS, TRENDS_FEEDS, WORLD_FEEDS, Settings

SWISS_FALLBACK_TOPICS = [
    {
        "id": "fallback-mieten",
        "topic": "Wohnkosten und Mieten in der Schweiz",
        "summary": "Fallback-Thema fuer den MVP, wenn keine Newsquelle erreichbar ist.",
        "keywords": ["Mieten", "Wohnkosten", "Wohnung", "Haushalte", "Kanton"],
        "score": 0.55,
        "source_titles": ["MVP fallback topic"],
    },
    {
        "id": "fallback-energie",
        "topic": "Energieverbrauch und Strompreise in der Schweiz",
        "summary": "Fallback-Thema fuer den MVP, wenn keine Newsquelle erreichbar ist.",
        "keywords": ["Energie", "Strom", "Elektrizitaet", "Verbrauch", "Preise"],
        "score": 0.52,
        "source_titles": ["MVP fallback topic"],
    },
    {
        "id": "fallback-klima",
        "topic": "Wetter, Klima und Hitzetage in der Schweiz",
        "summary": "Fallback-Thema fuer den MVP, wenn keine Newsquelle erreichbar ist.",
        "keywords": ["Klima", "Wetter", "Temperatur", "Hitze", "Niederschlag"],
        "score": 0.50,
        "source_titles": ["MVP fallback topic"],
    },
    {
        "id": "fallback-praemien",
        "topic": "Krankenkassenpraemien und Gesundheitskosten",
        "summary": "Fallback-Thema fuer den MVP, wenn keine Newsquelle erreichbar ist.",
        "keywords": ["Praemien", "Krankenkasse", "Gesundheitskosten", "Kanton"],
        "score": 0.54,
        "source_titles": ["MVP fallback topic"],
    },
    {
        "id": "fallback-migration",
        "topic": "Zuwanderung und Bevoelkerungsentwicklung in der Schweiz",
        "summary": "Fallback-Thema fuer den MVP, wenn keine Newsquelle erreichbar ist.",
        "keywords": ["Bevoelkerung", "Migration", "Zuwanderung", "Wanderungssaldo"],
        "score": 0.53,
        "source_titles": ["MVP fallback topic"],
    },
    {
        "id": "fallback-loehne",
        "topic": "Loehne und Kaufkraft in der Schweiz",
        "summary": "Fallback-Thema fuer den MVP, wenn keine Newsquelle erreichbar ist.",
        "keywords": ["Loehne", "Lohnentwicklung", "Kaufkraft", "Teuerung"],
        "score": 0.53,
        "source_titles": ["MVP fallback topic"],
    },
    {
        "id": "fallback-verkehr",
        "topic": "Mobilitaet und oeffentlicher Verkehr in der Schweiz",
        "summary": "Fallback-Thema fuer den MVP, wenn keine Newsquelle erreichbar ist.",
        "keywords": ["Verkehr", "Mobilitaet", "Bahn", "Personenverkehr"],
        "score": 0.51,
        "source_titles": ["MVP fallback topic"],
    },
    {
        "id": "fallback-tourismus",
        "topic": "Tourismus und Logiernaechte in der Schweiz",
        "summary": "Fallback-Thema fuer den MVP, wenn keine Newsquelle erreichbar ist.",
        "keywords": ["Tourismus", "Logiernaechte", "Hotellerie", "Gaeste"],
        "score": 0.51,
        "source_titles": ["MVP fallback topic"],
    },
]


# Brueckt global/lokal trendende Begriffe auf deutsche Suchbegriffe fuer opendata.swiss.
# Damit bekommt ein weltweites Thema (Krieg, Inflation, KI, ...) einen Schweizer Datenwinkel.
THEME_MAP: list[tuple[tuple[str, ...], list[str], str]] = [
    (("inflation", "teuerung", "preis", "price"), ["Konsumentenpreise", "Landesindex", "Teuerung", "Preise"], "Preise und Teuerung"),
    (("energie", "strom", "energy", "benzin", "erdgas", "erdoel", "oil"), ["Energie", "Strom", "Energieverbrauch", "Strompreise"], "Energie"),
    (("klima", "climate", "hitze", "heat", "hochwasser", "flut", "duerre", "unwetter", "wetter", "waldbrand", "gletscher"), ["Klima", "Temperatur", "Hitzetage", "Niederschlag"], "Klima und Wetter"),
    (("migration", "asyl", "fluechtling", "refugee", "einwanderung", "zuwanderung"), ["Migration", "Asyl", "Auslaendische Bevoelkerung", "Zuwanderung"], "Migration"),
    (("kuenstliche", "artificial", "intelligence", "chatgpt", "digital", "cyber", "internet"), ["Digitalisierung", "Informatik", "Internetnutzung"], "Digitalisierung und KI"),
    (("gesundheit", "health", "grippe", "virus", "pandemie", "covid", "spital", "krankenkasse", "praemie"), ["Gesundheit", "Spital", "Krankenversicherung", "Praemien"], "Gesundheit"),
    (("miete", "wohnung", "housing", "immobilien", "wohnen"), ["Mieten", "Wohnungen", "Leerwohnungen", "Bautaetigkeit"], "Wohnen"),
    (("arbeitslos", "unemployment", "stellen", "lohn", "loehne", "wage", "arbeitsmarkt"), ["Arbeitslosigkeit", "Erwerbstaetige", "Loehne", "Arbeitsmarkt"], "Arbeitsmarkt"),
    (("tourismus", "tourism", "reisen", "travel", "flug", "ferien"), ["Tourismus", "Logiernaechte", "Hotellerie"], "Tourismus"),
    (("verkehr", "bahn", "traffic", "stau", "pendler", "velo", "auto"), ["Verkehr", "Mobilitaet", "Personenverkehr", "Pendler"], "Verkehr und Mobilitaet"),
    (("krieg", "konflikt", "ukraine", "nahost", "iran", "israel", "gaza", "raketen", "sanktionen", "military"), ["Aussenhandel", "Schutzstatus", "Asyl", "Armee"], "Geopolitik und Schweizer Folgen"),
    (("wahl", "wahlen", "election", "abstimmung", "referendum", "initiative"), ["Abstimmungen", "Wahlen", "Stimmbeteiligung"], "Politik und Abstimmungen"),
    (("bank", "banken", "boerse", "zins", "franken", "hypothek", "finanz", "aktien", "bitcoin", "krypto"), ["Banken", "Zinssaetze", "Wechselkurs", "Hypotheken"], "Finanzen"),
    (("bevoelkerung", "population", "geburten", "demografie", "alterung"), ["Bevoelkerung", "Geburten", "Demografie", "Altersstruktur"], "Bevoelkerung"),
    (("kriminalitaet", "crime", "polizei", "gewalt", "einbruch", "betrug"), ["Kriminalitaet", "Straftaten", "Polizeiliche Kriminalstatistik"], "Kriminalitaet"),
    (("schule", "bildung", "education", "universitaet", "lehrstellen", "studium"), ["Bildung", "Schueler", "Studierende", "Lehrstellen"], "Bildung"),
]


# Begriffe, die auf Sport-/Unterhaltungsnews hindeuten: selten gute Open-Data-Storys
LOW_DATA_VALUE_TERMS = {
    "spiel", "match", "sieg", "niederlage", "tor", "trainer", "final", "halbfinal",
    "meister", "turnier", "etappe", "rennen", "olympia", "wm-titel", "song", "film",
    "serie", "star", "royals", "hochzeit", "konzert",
}


SCOPE_WEIGHTS = {"swiss": 1.0, "world": 0.8, "trend": 1.6}


STOPWORDS = {
    # deutsche Funktionswoerter
    "der", "die", "das", "den", "dem", "des", "und", "oder", "mit", "ohne",
    "fuer", "für", "zur", "zum", "eine", "einer", "einen", "einem", "eines",
    "von", "vom", "im", "in", "am", "auf", "aus", "bei", "beim", "ist", "sind",
    "war", "waren", "wird", "wurde", "wurden", "werden", "sein", "seine",
    "seiner", "seinen", "ihre", "ihren", "ihrer", "dass", "nicht", "noch",
    "schon", "jetzt", "immer", "alle", "alles", "beide", "viele", "mehr",
    "weniger", "sehr", "ganz", "auch", "aber", "denn", "dann", "wenn", "weil",
    "nach", "ueber", "über", "unter", "wieder", "sich", "kann", "koennen",
    "können", "muss", "muessen", "müssen", "soll", "sollen", "will", "wollen",
    "darf", "gibt", "kommt", "kommen", "geht", "gehen", "macht", "machen",
    "sagt", "sagen", "sorgt", "sorgen", "meldet", "melden", "zeigt", "zeigen",
    "steht", "stehen", "kurz", "kurzer", "kurze", "neue", "neuer", "neues",
    "neuen", "erste", "ersten", "erstmals", "heute", "morgen", "gestern",
    "diese", "dieser", "dieses", "diesen", "warum", "damit", "trotz", "wegen",
    # generische News-/Zahlwoerter
    "jahr", "jahre", "jahren", "woche", "wochen", "monat", "monate", "tage",
    "video", "live", "news", "artikel", "bericht", "interview", "kommentar",
    "analyse", "ticker", "update", "fotos", "bilder", "gestorben", "stirbt",
    "tote", "toten", "franken", "millionen", "milliarden", "prozent",
    "montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag",
    # Orts-/Quellenmarker
    "schweiz", "swiss", "switzerland", "suisse", "schweizer", "schweizerin",
    "medienmitteilung", "blick", "watson", "minuten", "tagesanzeiger",
    "bluewin", "suchvolumen",
    # englische Funktionswoerter
    "with", "from", "that", "this", "have", "says", "said", "after",
    "before", "about", "first", "being", "other", "these", "those", "where",
    "which", "while", "during", "against", "amid", "over", "their", "there",
    "could", "would", "should", "world", "years",
    # bekannte False-Positive-Marker
    "gegen", "einsatz", "kraefte", "kräfte", "boehmischer", "boehmische",
    "böhmischer", "böhmische", "tschechien", "sachsen", "grossbrand", "großbrand",
}


def fetch_news_items(timeout: int = 20) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    scoped_feeds = (
        [(url, "swiss") for url in SWISS_FEEDS]
        + [(url, "world") for url in WORLD_FEEDS]
        + [(url, "trend") for url in TRENDS_FEEDS]
    )
    for feed_url, scope in scoped_feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            continue

        for entry in parsed.entries[:25]:
            title = _clean_text(entry.get("title", ""))
            summary = _clean_text(entry.get("summary", ""))
            if not title:
                continue
            if scope == "trend":
                summary = _trend_summary(entry) or summary
            items.append(
                {
                    "source": parsed.feed.get("title", feed_url),
                    "scope": scope,
                    "title": title,
                    "summary": summary,
                    "url": entry.get("link", feed_url),
                    "published": entry.get("published", ""),
                    "published_ts": _entry_timestamp(entry),
                }
            )

    gdelt_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": '(Schweiz OR Suisse OR Switzerland OR Swiss) -"Bohemian Switzerland" -"Böhmische Schweiz" -"Böhmischer Schweiz"',
        "mode": "ArtList",
        "format": "json",
        "maxrecords": 50,
        "sort": "HybridRel",
    }
    try:
        response = requests.get(gdelt_url, params=params, timeout=timeout)
        response.raise_for_status()
        for article in response.json().get("articles", []):
            title = _clean_text(article.get("title", ""))
            summary = _clean_text(article.get("seendate", ""))
            if not title or not _is_relevant_swiss_news(title, summary):
                continue
            items.append(
                {
                    "source": article.get("sourceCommonName", "GDELT"),
                    "scope": "swiss",
                    "title": title,
                    "summary": summary,
                    "url": article.get("url", ""),
                    "published": article.get("seendate", ""),
                    "published_ts": _gdelt_timestamp(article.get("seendate", "")),
                }
            )
    except Exception:
        pass

    return items


def _trend_summary(entry: Any) -> str:
    parts = []
    traffic = _clean_text(entry.get("ht_approx_traffic", ""))
    if traffic:
        parts.append(f"Suchvolumen Schweiz: {traffic}")
    news_items = entry.get("ht_news_items") or []
    if isinstance(news_items, dict):
        news_items = [news_items]
    for news in news_items[:3]:
        title = _clean_text(news.get("ht_news_item_title", ""))
        if title:
            parts.append(title)
    return " | ".join(parts)


def _entry_timestamp(entry: Any) -> float | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return calendar.timegm(parsed)
            except (TypeError, ValueError):
                continue
    return None


def _gdelt_timestamp(seendate: str) -> float | None:
    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()
    except ValueError:
        return None


def _item_weight(item: dict[str, Any]) -> float:
    weight = SCOPE_WEIGHTS.get(item.get("scope", "swiss"), 1.0)
    ts = item.get("published_ts")
    if ts:
        age_hours = max(0.0, (datetime.now(UTC).timestamp() - ts) / 3600)
        if age_hours <= 24:
            weight *= 1.25
        elif age_hours <= 48:
            weight *= 1.0
        elif age_hours <= 96:
            weight *= 0.7
        else:
            weight *= 0.4
    return weight


def _theme_for_terms(terms: list[str]) -> tuple[list[str], str | None]:
    matched_keywords: list[str] = []
    matched_label: str | None = None
    lowered = [term.lower() for term in terms]
    for patterns, keywords, label in THEME_MAP:
        if any(pattern in term for term in lowered for pattern in patterns):
            for keyword in keywords:
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)
            if matched_label is None:
                matched_label = label
    return matched_keywords, matched_label


def build_topic_candidates(news_items: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    if not news_items:
        return _rotating_fallback_topics(settings.max_topics)

    ai_topics = _try_ai_topics(news_items, settings)
    if ai_topics:
        return [_enrich_topic_with_theme(topic) for topic in ai_topics[: settings.max_topics]]

    return _build_topics_with_rules(news_items, settings.max_topics)


def _rotating_fallback_topics(max_topics: int) -> list[dict[str, Any]]:
    pool = SWISS_FALLBACK_TOPICS
    offset = date.today().toordinal() % len(pool)
    rotated = pool[offset:] + pool[:offset]
    return rotated[:max_topics]


def _enrich_topic_with_theme(topic: dict[str, Any]) -> dict[str, Any]:
    terms = [topic.get("topic", "")] + [str(k) for k in topic.get("keywords", [])]
    theme_keywords, _ = _theme_for_terms(terms)
    merged = list(topic.get("keywords", []))
    for keyword in theme_keywords:
        if keyword not in merged:
            merged.append(keyword)
    topic["keywords"] = merged[:12]
    return topic


def _build_topics_with_rules(news_items: list[dict[str, Any]], max_topics: int) -> list[dict[str, Any]]:
    documents = []
    keyword_weights: dict[str, float] = defaultdict(float)
    keyword_scopes: dict[str, set[str]] = defaultdict(set)

    for item in news_items:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        keywords = _extract_keywords(text)
        weight = _item_weight(item)
        for keyword in keywords:
            keyword_weights[keyword] += weight
            keyword_scopes[keyword].add(item.get("scope", "swiss"))
        documents.append((item, keywords))

    boosted: dict[str, float] = {}
    for keyword, weight in keyword_weights.items():
        scopes = keyword_scopes[keyword]
        theme_keywords, _ = _theme_for_terms([keyword])
        if {"swiss", "world"} <= scopes:
            weight *= 1.35
        if theme_keywords:
            weight *= 1.2
        elif keyword.lower() in LOW_DATA_VALUE_TERMS:
            weight *= 0.35
        elif scopes == {"trend"}:
            # Trend-only Begriffe ohne Datenwinkel sind meist Sport-/Promi-Namen
            weight *= 0.5
        boosted[keyword] = weight

    top_terms = sorted(boosted, key=boosted.get, reverse=True)[:25]
    max_weight = max(boosted.values()) if boosted else 1.0
    topics: list[dict[str, Any]] = []
    used_titles: set[str] = set()

    for term in top_terms:
        matching = [(item, kws) for item, kws in documents if term in kws and item["title"] not in used_titles]
        if not matching:
            continue
        source_titles = [item["title"] for item, _ in matching[:4]]
        for title in source_titles:
            used_titles.add(title)
        merged_keywords = Counter(keyword for _, kws in matching for keyword in kws)
        keywords = [keyword for keyword, _ in merged_keywords.most_common(8)]
        theme_keywords, theme_label = _theme_for_terms([term] + keywords)
        search_keywords = theme_keywords + [k for k in keywords if k not in theme_keywords]
        label = _topic_label(term, keywords)
        if theme_label:
            label = f"{theme_label}: {label}"
        scopes = sorted(keyword_scopes[term])
        summary = f"Aktuelles Cluster aus {len(matching)} Meldungen mit Bezug zu '{term}'."
        if "world" in scopes and "swiss" in scopes:
            summary += " Das Thema trendet weltweit und in der Schweiz."
        elif "world" in scopes:
            summary += " Weltweit trendendes Thema mit Schweizer Datenwinkel."
        elif "trend" in scopes:
            summary += " Aktuell stark gesucht auf Google Trends Schweiz."
        topics.append(
            {
                "id": _stable_id(label),
                "topic": label,
                "summary": summary,
                "keywords": search_keywords[:12],
                "score": round(min(1.0, 0.4 + 0.6 * boosted[term] / max_weight), 3),
                "source_titles": source_titles,
                "scopes": scopes,
            }
        )
        if len(topics) >= max_topics:
            break

    return topics or _rotating_fallback_topics(max_topics)


def _try_ai_topics(news_items: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    sample = _balanced_sample(news_items, per_scope=15)
    prompt = (
        "Du kuratierst Datenstorys fuer ein Schweizer Publikum. Die News enthalten "
        "Schweizer Meldungen (scope=swiss), weltweite Meldungen (scope=world) und "
        "Google-Trends-Suchbegriffe aus der Schweiz (scope=trend). "
        "Extrahiere maximal 5 Datenthemen, die JETZT interessieren. Bevorzuge Themen, "
        "die weltweit UND in der Schweiz trenden. Finde fuer globale Themen einen "
        "Schweizer Datenwinkel (z.B. Krieg -> Asylgesuche/Aussenhandel, Inflation -> "
        "Konsumentenpreise, Hitzewelle -> Hitzetage). Meide reine Sport- und Promi-News, "
        "ausser es gibt einen klaren Statistikbezug. "
        "Gib ausschliesslich JSON zurueck: eine Liste mit Objekten mit "
        "topic, summary, keywords, score und source_titles. "
        "Keywords muessen deutsche Suchbegriffe sein, die auf opendata.swiss Treffer liefern "
        "(amtliche Statistik-Begriffe, keine Eigennamen).\n\n"
        + json.dumps(sample, ensure_ascii=False)
    )
    parsed = parse_json_list(generate(settings, prompt, "topics"), wrapper_keys=("topics",))
    topics = []
    for item in parsed:
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        keywords = item.get("keywords") or _extract_keywords(topic)
        topics.append(
            {
                "id": _stable_id(topic),
                "topic": topic,
                "summary": str(item.get("summary", ""))[:500],
                "keywords": [str(k) for k in keywords][:10],
                "score": _bounded_float(item.get("score", 0.7)),
                "source_titles": [str(t) for t in item.get("source_titles", [])][:5],
            }
        )
    return topics


def _balanced_sample(news_items: list[dict[str, Any]], per_scope: int) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for scope in ("trend", "world", "swiss"):
        scoped = [item for item in news_items if item.get("scope", "swiss") == scope]
        scoped.sort(key=lambda item: item.get("published_ts") or 0, reverse=True)
        sample.extend(
            {key: item[key] for key in ("scope", "source", "title", "summary") if key in item}
            for item in scoped[:per_scope]
        )
    return sample


def _clean_text(value: str) -> str:
    value = re.sub("<[^<]+?>", " ", unescape(str(value)))
    return re.sub(r"\s+", " ", value).strip()


def _is_relevant_swiss_news(title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()
    false_positive_markers = [
        "bohemian switzerland",
        "böhmische schweiz",
        "böhmischer schweiz",
        "boehmische schweiz",
        "boehmischer schweiz",
        "sächsische schweiz",
        "saechsische schweiz",
    ]
    if any(marker in text for marker in false_positive_markers):
        return False
    swiss_markers = [
        "swiss",
        "schweiz",
        "suisse",
        "svizzera",
        "zurich",
        "zürich",
        "geneva",
        "genf",
        "bern",
        "basel",
        "lausanne",
        "ubs",
        "credit suisse",
        "bundesrat",
        "parlament",
    ]
    return any(marker in text for marker in swiss_markers)


def _extract_keywords(text: str) -> list[str]:
    normalized = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("Ä", "Ae")
        .replace("Ö", "Oe")
        .replace("Ü", "Ue")
        .replace("ß", "ss")
    )
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", normalized)
    keywords = []
    for word in words:
        lower = word.lower().strip("-")
        if lower in STOPWORDS or lower.replace("-", "") in STOPWORDS or len(lower) < 4:
            continue
        keywords.append(lower.capitalize())
    return list(dict.fromkeys(keywords))[:12]


def _topic_label(seed: str, keywords: list[str]) -> str:
    title_terms = [seed.capitalize()]
    for keyword in keywords:
        if keyword.lower() != seed.lower() and len(title_terms) < 4:
            title_terms.append(keyword)
    return " / ".join(title_terms)


def _stable_id(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"topic-{digest}"


def _bounded_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return round(max(0.0, min(1.0, number)), 3)


def run_started_at() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
