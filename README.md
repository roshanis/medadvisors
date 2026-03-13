# Medical Advisors

Medical Advisors is a multi-agent application that stages a clinical case review across specialist roles and returns a concise plan, transcript, and raw artifacts. It is designed for exploratory decision support workflows, not autonomous diagnosis.

## What it does

- Runs a structured advisor meeting around a medical case
- Uses specialist presets such as attending, emergency medicine, internal medicine, radiology, cardiology, insurance, and pharmacy
- Produces a summary, transcript, and JSON output for later review
- Supports clarifying questions before the advisor meeting starts
- Offers fast mode, caching, and optional web search

## Safety and scope

- Prototype and educational use only
- Not medical advice
- Do not enter PHI unless you control the environment and policy requirements
- Human review is required before acting on the output

## Quick start

### Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### API key

For the UI:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

For CLI usage:

```bash
cp .env.example .env
```

Then add `OPENAI_API_KEY` to `.env`.

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## Typical workflow

1. Describe the case.
2. Answer clarifying questions.
3. Review or edit the specialist team.
4. Run the advisor meeting.
5. Read the summary, transcript, and raw JSON output.

Saved outputs land in `advisor_meetings/`.

## Status

Current status: active prototype for multi-agent medical case analysis.

## License

MIT for this app code. Any upstream framework keeps its own license.
