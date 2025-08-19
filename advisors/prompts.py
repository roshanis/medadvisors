PROMPT_GOAL_SUFFIX_LEAD = (
    " Produce a final consensus under the headings: Assumptions; Options (pros/cons); Recommendation; "
    "Risks & Mitigations; Next Steps. The Recommendation MUST be a short numbered action plan (3–7 items). "
    "For EACH action, include these fields explicitly: Action (strong verb), Owner, Deadline (date or timeframe), "
    "Steps (how to execute), Tools/Resources (links if mentioned), Success Metric (target), Risk & Mitigation. "
    "Avoid vague language (no 'leverage', 'optimize' without details). Be concrete and succinct."
)

PROMPT_GOAL_SUFFIX_MEMBER = (
    " Provide concrete, verifiable details; quantify where possible; explicitly state uncertainty; "
    "cite sources when literature/search is enabled. Focus on advising, not just critiquing: "
    "propose specific actions with rationale, offer alternatives and tradeoffs, and suggest next steps."
)

ACTIONABILITY_RULE = (
    "Recommendation must be a numbered action plan (3–7 items). For each action, specify: Action, Owner, "
    "Deadline, Steps, Tools/Resources, Success Metric, Risk & Mitigation. Avoid vague language."
)

ADVICE_RULE = (
    "Advisors must provide actionable advice (specific actions and why), not just critique. "
    "Include at least one concrete recommended action and an alternative with tradeoffs, when applicable."
)
