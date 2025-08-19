# Medical Advisors

Medical Advisors is an AI‑assisted, multi‑agent app that coordinates medical specialists to analyze a case and synthesize a clear, actionable consensus plan.

## Features
- Multi‑agent “team meeting” to analyze a case and synthesize a consensus plan
- Transparent artifacts: full transcript (.md) and raw messages (.json)
- Streamlit UI for agenda, questions, rules, and team configuration
- Clarity Assistant: auto‑suggests clarifying questions; your answers guide the agents
- Optional PubMed search (via the underlying framework)

## Safety & Scope
- Educational/prototype use only; not medical advice
- Avoid PHI; comply with institutional policies
- Human oversight and guideline verification required

## Quickstart
1) Setup
```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate venv
uv venv
source .venv/bin/activate

# Install deps
uv pip install -r requirements.txt
```

2) API key
- UI script: create `./.streamlit/secrets.toml`
```toml
OPENAI_API_KEY = "sk-..."
```
- CLI script: create `.env` (used by `medical_consensus.py`)
```bash
cp .env.example .env
# edit .env and set: OPENAI_API_KEY=sk-...
```

3) Run UI
```bash
streamlit run app.py
# open http://localhost:8501
```

## Using Medical Advisors
1) Describe your medical case in “Case Description (Agenda)”
2) Optionally click “Suggest Questions” and answer clarifiers (Clarity Assistant)
3) Review the prefilled team (leader + specialists) and edit titles/expertise if desired
4) Click “Run Advisors”, then review tabs: Consensus Summary, Transcript, Raw JSON
5) Saved artifacts: `advisor_meetings/<session>.md` and `<session>.json`

## Configuration Tips
- Models: Clarifying questions use your selection (e.g., gpt‑5‑nano). Team meeting uses GPT‑4.1‑nano when a gpt‑5* model is selected (Assistants requirement).
- Fast mode: Uses 1 round and smaller specialist models for lower latency/cost.
- Web search: DuckDuckGo summaries can provide background context; disable for offline runs.
- Caching: Enable “Cache outputs” to reuse recent results; turn off when iterating prompts.
- Actionability: Recommendations are short, numbered action plans with owners, deadlines, steps, tools, metrics, risks.

## CLI Example
```bash
python medical_consensus.py
```
Edit `AGENDA`, `AGENDA_QUESTIONS`, and `AGENDA_RULES` in the script as needed.

## Deploy
- Local: Streamlit run as above
- Container: build a minimal Python image and expose port 8501
- Internal use: protect behind SSO/reverse proxy; store `.env` securely (not in git)

## License
- App code: MIT (adjust as needed)
- Virtual Lab: see license in the upstream repository