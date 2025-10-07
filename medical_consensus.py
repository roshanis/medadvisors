"""CLI entry point for running advisor meetings without the Streamlit UI."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from advisors.prompts import PROMPT_GOAL_SUFFIX_LEAD, PROMPT_GOAL_SUFFIX_MEMBER, ACTIONABILITY_RULE, ADVICE_RULE
from advisors.presets import (
    CATEGORY_PRESETS,
    CATEGORY_QUESTIONS,
    CATEGORY_RULES,
)
from advisors.services.context import build_pubmed_context, build_web_context
from advisors.services.meeting_fast import run_fast_completions
from advisors.services.run_modes import DEFAULT_MODE_KEY, RUN_MODES
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

SAVE_DIR = BASE_DIR / "medical_meetings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _build_team(category: str, model: str) -> tuple[Agent, tuple[Agent, ...]]:
    preset = CATEGORY_PRESETS[category]
    lead_cfg = preset["lead"]
    members_cfg = preset["members"]
    team_lead = Agent(
        title=lead_cfg["title"],
        expertise=lead_cfg["expertise"],
        goal=lead_cfg["goal"] + PROMPT_GOAL_SUFFIX_LEAD,
        role=lead_cfg["role"],
        model=model,
    )
    team_members = tuple(
        Agent(
            title=m["title"],
            expertise=m["expertise"],
            goal=m["goal"] + PROMPT_GOAL_SUFFIX_MEMBER,
            role=m["role"],
            model=model,
        )
        for m in members_cfg
    )
    return team_lead, team_members


def _read_agenda(args: argparse.Namespace) -> str:
    if args.agenda_file:
        return args.agenda_file.read_text(encoding="utf-8").strip()
    if args.agenda:
        return args.agenda.strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a medical advisor consensus meeting from the CLI.")
    parser.add_argument("--agenda", help="Case description. You can also use --agenda-file or pipe input.")
    parser.add_argument("--agenda-file", type=Path, help="Path to a text/markdown file containing the case.")
    parser.add_argument(
        "--category",
        choices=sorted(CATEGORY_PRESETS.keys()),
        default="Medical",
        help="Advisor category to use (defaults to Medical).",
    )
    parser.add_argument(
        "--mode",
        choices=list(RUN_MODES.keys()),
        default=DEFAULT_MODE_KEY,
        help="Run mode controlling cost/quality trade-offs.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override number of discussion rounds (falls back to the chosen run mode).",
    )
    parser.add_argument(
        "--model",
        help="Override the model name used for advisors (defaults to the run mode's model).",
    )
    parser.add_argument(
        "--context",
        action="append",
        default=[],
        help="Additional context blocks to prepend (can be repeated).",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable DuckDuckGo context even if the run mode enables it.",
    )
    parser.add_argument(
        "--no-pubmed",
        action="store_true",
        help="Disable PubMed lookups even if the run mode enables them.",
    )
    parser.add_argument(
        "--save-name",
        help="Optional transcript base name. If omitted, a timestamped name is generated.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agenda = _read_agenda(args)
    if not agenda:
        raise SystemExit("Provide an agenda via --agenda, --agenda-file, or stdin.")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in the environment or .env file.")

    run_mode = RUN_MODES[args.mode]
    model = args.model or run_mode.model
    num_rounds = args.rounds or run_mode.num_rounds
    enable_web = run_mode.enable_web_search and not args.no_web
    enable_pubmed = run_mode.enable_pubmed and not args.no_pubmed

    additional_contexts: List[str] = [c.strip() for c in args.context if c.strip()]
    if enable_web:
        web_ctx = build_web_context(args.category, agenda)
        if web_ctx:
            additional_contexts.append(web_ctx)
    pm_query = pm_md = ""
    if enable_pubmed:
        pm_query, pm_md, _, _ = build_pubmed_context(agenda)
        if pm_md:
            additional_contexts.append(pm_md)

    team_lead, team_members = _build_team(args.category, model)
    agenda_questions = tuple(CATEGORY_QUESTIONS.get(args.category, []))
    agenda_rules = tuple(list(CATEGORY_RULES.get(args.category, [])) + [ACTIONABILITY_RULE, ADVICE_RULE])
    save_name = args.save_name or f"cli_{int(time.time())}"

    contexts_tuple = tuple(ctx for ctx in additional_contexts if ctx)

    if run_mode.fast_path:
        lead_spec = {
            "title": team_lead.title,
            "expertise": team_lead.expertise,
            "goal": team_lead.goal,
        }
        member_specs = tuple(
            {"title": m.title, "expertise": m.expertise, "goal": m.goal, "role": m.role}
            for m in team_members
        )
        summary = run_fast_completions(
            agenda=agenda,
            contexts=contexts_tuple,
            lead_spec=lead_spec,
            member_specs=member_specs,
            model_name=model,
            num_rounds=num_rounds,
        )
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        (SAVE_DIR / f"{save_name}.md").write_text(
            "\n\n".join(
                [
                    "# Medical Advisors — Transcript (CLI Fast Path)",
                    f"## Agenda\n\n{agenda}",
                    *(f"## Context\n\n{ctx}" for ctx in contexts_tuple),
                    "## Consensus Summary\n\n" + (summary or "(No summary generated)"),
                ]
            ),
            encoding="utf-8",
        )
        print(summary)
        return

    summary = run_meeting(
        meeting_type="team",
        agenda=agenda,
        save_dir=SAVE_DIR,
        save_name=save_name,
        team_lead=team_lead,
        team_members=team_members,
        agenda_questions=agenda_questions,
        agenda_rules=agenda_rules,
        contexts=contexts_tuple,
        num_rounds=num_rounds,
        temperature=0.7,
        pubmed_search=enable_pubmed,
        return_summary=True,
    )
    print(summary)


if __name__ == "__main__":
    main()
