"""Definitions for advisor run modes with their cost/performance trade-offs."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RunMode:
    label: str
    description: str
    model: str
    num_rounds: int
    fast_path: bool
    enable_pubmed: bool
    enable_web_search: bool


RUN_MODES: Dict[str, RunMode] = {
    "budget": RunMode(
        label="Budget",
        description="Single-round completions, lightweight model, no external lookups.",
        model="gpt-4.1-mini",
        num_rounds=1,
        fast_path=True,
        enable_pubmed=False,
        enable_web_search=False,
    ),
    "balanced": RunMode(
        label="Balanced",
        description="Fast completions with targeted web/PubMed context for better recall.",
        model="gpt-4.1-mini",
        num_rounds=1,
        fast_path=True,
        enable_pubmed=True,
        enable_web_search=True,
    ),
    "thorough": RunMode(
        label="Thorough",
        description="Full multi-agent Assistants run (higher quality and cost).",
        model="gpt-4.1",
        num_rounds=2,
        fast_path=False,
        enable_pubmed=True,
        enable_web_search=True,
    ),
}


DEFAULT_MODE_KEY = "budget"
