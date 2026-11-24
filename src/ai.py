from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import Settings

_errors: list[str] = []


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str | None  # None = offizielle OpenAI-API bzw. Spezialfall gemini
    key_attr: str
    model_attr: str


# Groq, Ollama und Hugging Face sprechen alle das OpenAI-Chat-Protokoll,
# darum genuegt ein Codepfad mit unterschiedlicher base_url.
OPENAI_COMPATIBLE = {
    "openai": ProviderSpec("openai", None, "openai_api_key", "openai_model"),
    "groq": ProviderSpec("groq", "https://api.groq.com/openai/v1", "groq_api_key", "groq_model"),
    "ollama": ProviderSpec("ollama", None, "ollama_url", "ollama_model"),  # base_url kommt aus settings
    "hf": ProviderSpec("hf", "https://router.huggingface.co/v1", "hf_token", "hf_model"),
}


def generate(settings: Settings, prompt: str, purpose: str) -> str | None:
    """Einzige Stelle, die AI-Provider aufruft.

    settings.ai_providers ist eine Prioritaetenliste (z.B. ["groq", "ollama"]);
    der erste Provider, der antwortet, gewinnt. Fehler werden gesammelt statt verschluckt.
    """
    for provider in settings.ai_providers:
        content = _call_provider(provider, settings, prompt, purpose)
        if content and content.strip():
            return content
    return None


def _call_provider(provider: str, settings: Settings, prompt: str, purpose: str) -> str | None:
    if provider == "gemini":
        if not settings.gemini_api_key:
            return None
        try:
            from google import genai

            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(model=settings.gemini_model, contents=prompt)
            return response.text
        except Exception as error:
            _record_error(purpose, provider, error)
            return None

    spec = OPENAI_COMPATIBLE.get(provider)
    if spec is None:
        return None

    if provider == "ollama":
        api_key = "ollama"  # Platzhalter, Ollama prueft keinen Key
        base_url = settings.ollama_url
    else:
        api_key = getattr(settings, spec.key_attr, None)
        base_url = spec.base_url
        if not api_key:
            return None

    model = getattr(settings, spec.model_attr)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=settings.ai_timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as error:
        _record_error(purpose, provider, error)
        return None


def _record_error(purpose: str, provider: str, error: Exception) -> None:
    message = f"{purpose} ({provider}): {str(error)[:300]}"
    _errors.append(message)
    print(f"WARNUNG: AI-Aufruf fehlgeschlagen - {message}")


def collected_ai_errors() -> list[str]:
    return list(_errors)


def parse_json_object(content: str | None) -> dict[str, Any]:
    data = _parse_json(content)
    return data if isinstance(data, dict) else {}


def parse_json_list(content: str | None, wrapper_keys: tuple[str, ...] = ()) -> list[Any]:
    data = _parse_json(content)
    if isinstance(data, dict):
        for key in wrapper_keys:
            if isinstance(data.get(key), list):
                return data[key]
        return []
    return data if isinstance(data, list) else []


def _parse_json(content: str | None) -> Any:
    """Robustes Parsen: kleinere Modelle liefern JSON oft mit Prosa oder Codefences drumherum."""
    if not content:
        return None
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    pairs = [("{", "}"), ("[", "]")]
    # Der Klammertyp, der zuerst im Text auftaucht, gewinnt - sonst wird aus
    # einer Liste "[{...}]" faelschlich nur das innere Objekt extrahiert.
    pairs.sort(key=lambda pair: cleaned.find(pair[0]) if cleaned.find(pair[0]) != -1 else len(cleaned))
    for opener, closer in pairs:
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None
