from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


# Schweizer Newsquellen (scope "swiss")
SWISS_FEEDS = [
    "https://www.srf.ch/news/bnf/rss/1646",
    "https://news.google.com/rss?hl=de-CH&gl=CH&ceid=CH:de",
    "https://www.watson.ch/api/1.0/rss/index.xml",
    "https://partner-feeds.beta.20min.ch/rss/20minuten",
    "https://www.admin.ch/gov/de/start/dokumentation/medienmitteilungen.rss",
]

# Weltweite Newsquellen (scope "world") - fuer global trending Themen mit Schweizer Datenwinkel
WORLD_FEEDS = [
    "https://news.google.com/rss/headlines/section/topic/WORLD?hl=de-CH&gl=CH&ceid=CH:de",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]

# Google Trends Schweiz (scope "trend") - was die Schweizer Bevoelkerung gerade sucht
TRENDS_FEEDS = [
    "https://trends.google.com/trending/rss?geo=CH",
]

# Rueckwaertskompatibilitaet
DEFAULT_FEEDS = SWISS_FEEDS


@dataclass(frozen=True)
class Settings:
    ai_provider: str = "none"
    ai_providers: tuple[str, ...] = ()
    ai_timeout: float = 120.0
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    ollama_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:7b"
    hf_token: str | None = None
    hf_model: str = "meta-llama/Llama-3.3-70B-Instruct"
    max_topics: int = 5
    max_datasets_per_topic: int = 8
    min_story_score: float = 0.55
    max_story_similarity: float = 0.86
    history_limit: int = 40
    output_dir: Path = OUTPUT_DIR


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


KNOWN_PROVIDERS = ("openai", "gemini", "groq", "ollama", "hf")


def _detect_providers(
    openai_key: str | None, gemini_key: str | None, groq_key: str | None, hf_token: str | None
) -> tuple[str, ...]:
    """Automatische Reihenfolge, wenn AI_PROVIDER nicht gesetzt ist: Gratis-Provider zuerst."""
    providers = []
    if groq_key:
        providers.append("groq")
    if openai_key:
        providers.append("openai")
    if gemini_key:
        providers.append("gemini")
    if hf_token:
        providers.append("hf")
    return tuple(providers)


def load_settings() -> Settings:
    raw_provider = os.getenv("AI_PROVIDER", "none").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY") or None
    gemini_key = os.getenv("GEMINI_API_KEY") or None
    groq_key = os.getenv("GROQ_API_KEY") or None
    hf_token = os.getenv("HF_TOKEN") or None

    # AI_PROVIDER kann eine Prioritaetenliste sein, z.B. "groq,ollama,gemini"
    if raw_provider in {"", "none"}:
        providers = _detect_providers(openai_key, gemini_key, groq_key, hf_token)
    else:
        providers = tuple(
            p.strip() for p in raw_provider.split(",") if p.strip() in KNOWN_PROVIDERS
        )

    return Settings(
        ai_provider=",".join(providers) if providers else "none",
        ai_providers=providers,
        ai_timeout=_float_env("AI_TIMEOUT", 120.0),
        openai_api_key=openai_key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        gemini_api_key=gemini_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        groq_api_key=groq_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        hf_token=hf_token,
        hf_model=os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        max_topics=_int_env("MAX_TOPICS", 5),
        max_datasets_per_topic=_int_env("MAX_DATASETS_PER_TOPIC", 8),
        min_story_score=_float_env("MIN_STORY_SCORE", 0.55),
        max_story_similarity=_float_env("MAX_STORY_SIMILARITY", 0.86),
        history_limit=_int_env("HISTORY_LIMIT", 40),
    )
