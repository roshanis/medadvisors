from typing import Dict, List, Tuple
from openai import OpenAI

from advisors.prompts import ACTIONABILITY_RULE, ADVICE_RULE


def _chat(model_name: str, system: str, user: str) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content if resp.choices else ""


def run_fast_completions(
    agenda: str,
    contexts: Tuple[str, ...],
    lead_spec: Dict[str, str],
    member_specs: Tuple[Dict[str, str], ...],
    model_name: str,
    num_rounds: int = 1,
) -> str:
    context_block = "\n\n".join(contexts) if contexts else ""

    def member_prompt(m: Dict[str, str]) -> Tuple[str, str]:
        system = (
            f"You are {m['title']}. Expertise: {m['expertise']}. Goal: {m['goal']}. "
            f"{ADVICE_RULE} {ACTIONABILITY_RULE}"
        )
        user = (
            f"Agenda:\n{agenda}\n\n"
            + (f"Context:\n{context_block}\n\n" if context_block else "")
            + "Provide your actionable advice now. Be concise."
        )
        return system, user

    # Run members sequentially (could be parallelized by caller if needed)
    member_outputs: List[str] = []
    for m in member_specs:
        sys, usr = member_prompt(m)
        try:
            member_outputs.append(_chat(model_name, sys, usr) or "")
        except Exception:
            member_outputs.append("")

    # Lead synthesis
    lead_system = (
        f"You are {lead_spec['title']}. Expertise: {lead_spec['expertise']}. Goal: {lead_spec['goal']}. "
        f"{ACTIONABILITY_RULE}"
    )
    members_block = "\n\n".join(
        f"[member {i+1}]\n{out}" for i, out in enumerate(member_outputs) if out.strip()
    )
    lead_user = (
        f"Agenda:\n{agenda}\n\n"
        + (f"Context:\n{context_block}\n\n" if context_block else "")
        + (f"Team member advice:\n{members_block}\n\n" if members_block else "")
        + "Think step by step.Produce the final consensus in markdown."
    )
    summary_md = _chat(model_name, lead_system, lead_user)
    return summary_md or "(No summary generated)"
