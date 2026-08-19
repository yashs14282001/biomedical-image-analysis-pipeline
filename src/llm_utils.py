from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field, ValidationError

from config import PROMPT_DIR, TEXT_MODEL, VISION_MODEL


class DirectVLMRecord(BaseModel):
    modality: str = Field(description="Visible imaging modality, or 'uncertain'.")
    tissue_type: str = Field(description="Visible tissue/sample type, or 'uncertain'.")
    notable_features: list[str]
    image_quality: Literal["good", "acceptable", "poor", "uncertain"]


class NumbersFirstRecord(BaseModel):
    n_objects: int
    density_class: Literal["low", "medium", "high", "uncertain"]
    shape_regularity: Literal["regular", "mixed", "irregular", "uncertain"]
    quality_flag: Literal["acceptable", "review", "poor", "uncertain"]


class HybridRecord(BaseModel):
    image_id: str
    n_objects: int
    mean_area: float
    density_class: Literal["low", "medium", "high", "uncertain"]
    quality_flag: Literal["acceptable", "review", "poor", "uncertain"]


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _content(response) -> str:
    # Current ollama-python exposes response.message.content; the fallback keeps the code
    # tolerant of older dict-like responses.
    if hasattr(response, "message") and hasattr(response.message, "content"):
        return response.message.content
    if isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    return str(response)


def parse_json_loose(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def run_naive_vlm(image_path: Path, model: str = VISION_MODEL, temperature: float = 0.7) -> str:
    prompt = load_prompt("naive_vlm.txt")
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        options={"temperature": temperature},
    )
    return _content(response)


def run_optimised_vlm(
    image_path: Path,
    model: str = VISION_MODEL,
    temperature: float = 0.6,
) -> tuple[dict, str]:
    prompt = load_prompt("optimized_vlm.txt")
    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        format=DirectVLMRecord.model_json_schema(),
        options={"temperature": temperature},
    )
    raw = _content(response)
    record = DirectVLMRecord.model_validate_json(raw).model_dump()
    return record, raw


def run_numbers_first(
    feature_summary: str,
    facts: dict,
    model: str = TEXT_MODEL,
    temperature: float = 0.3,
) -> tuple[dict, str, str]:
    template = load_prompt("numbers_first.txt")
    prompt = template.replace("{{FEATURE_SUMMARY}}", feature_summary)
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=NumbersFirstRecord.model_json_schema(),
        options={"temperature": temperature},
    )
    raw = _content(response)
    parsed = NumbersFirstRecord.model_validate_json(raw).model_dump()

    # Audit guard: counts are measured values, not LLM opinions.
    parsed["n_objects"] = int(facts["n_objects"])

    narrative_prompt = load_prompt("numbers_first_narrative.txt")
    narrative_prompt = narrative_prompt.replace("{{JSON_RECORD}}", json.dumps(parsed, indent=2))
    narrative_prompt = narrative_prompt.replace("{{FEATURE_SUMMARY}}", feature_summary)
    narrative_response = chat(
        model=model,
        messages=[{"role": "user", "content": narrative_prompt}],
        options={"temperature": 0.2},
    )
    narrative = _content(narrative_response).strip()
    return parsed, raw, narrative


def run_hybrid_record(
    image_id: str,
    feature_summary: str,
    facts: dict,
    model: str = TEXT_MODEL,
    temperature: float = 0.2,
) -> tuple[dict, str]:
    template = load_prompt("hybrid_record.txt")
    prompt = (
        template.replace("{{IMAGE_ID}}", image_id)
        .replace("{{FEATURE_SUMMARY}}", feature_summary)
    )
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=HybridRecord.model_json_schema(),
        options={"temperature": temperature},
    )
    raw = _content(response)
    record = HybridRecord.model_validate_json(raw).model_dump()

    # Force directly measured quantities back to their deterministic values.
    record["image_id"] = image_id
    record["n_objects"] = int(facts["n_objects"])
    record["mean_area"] = round(float(facts["mean_area"]), 3)
    return record, raw


def narrative_from_record(record: dict, feature_summary: str, model: str = TEXT_MODEL) -> str:
    template = load_prompt("hybrid_narrative.txt")
    prompt = (
        template.replace("{{JSON_RECORD}}", json.dumps(record, indent=2))
        .replace("{{FEATURE_SUMMARY}}", feature_summary)
    )
    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return _content(response).strip()


def ollama_error_message(exc: Exception) -> str:
    return (
        f"Ollama call failed: {exc}\n"
        "Make sure the Ollama app/server is running and the required models are pulled:\n"
        "  ollama pull llama3.2-vision\n"
        "  ollama pull llama3.2:3b"
    )
