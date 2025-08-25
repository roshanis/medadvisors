import os
import random
import time
import secrets
import concurrent.futures
from collections import defaultdict, deque
from pathlib import Path
import streamlit as st
import json
from typing import List, Dict, Tuple
from openai import OpenAI
try:
    from duckduckgo_search import DDGS  # type: ignore
except Exception:
    DDGS = None  # optional dependency
import streamlit.components.v1 as components
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting

BASE_DIR = Path(__file__).resolve().parent

# ---- Category presets (leader + specialists with goals/roles) ----
CATEGORY_PRESETS: Dict[str, Dict] = {
    "Medical": {
        "lead": {
            "title": "Attending Physician",
            "expertise": "evidence-based medicine, multidisciplinary care",
            "goal": "synthesize differential diagnosis, diagnostics, and initial management plan with risks and contingencies",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Emergency Medicine",
                "expertise": "triage, resuscitation, stabilization",
                "goal": "prioritize ABCs, immediate stabilization steps, and initial orders",
                "role": "emergency",
            },
            {
                "title": "Internal Medicine",
                "expertise": "differential diagnosis, inpatient management",
                "goal": "construct prioritized differential and inpatient plan",
                "role": "hospitalist",
            },
            {
                "title": "Radiology",
                "expertise": "imaging selection and interpretation",
                "goal": "recommend appropriate imaging and interpret key findings",
                "role": "radiology",
            },
            {
                "title": "Cardiology",
                "expertise": "ACS workup, arrhythmias, heart failure",
                "goal": "assess cardiac risks, tests, and management",
                "role": "cardiology",
            },
            {
                "title": "Insurance Expert",
                "expertise": "coverage criteria, prior authorization, coding/billing",
                "goal": "identify coverage constraints, recommend documentation for approvals, and estimate patient cost",
                "role": "insurance",
            },
            {
                "title": "Clinical Pharmacist",
                "expertise": "dosing, interactions, renal/hepatic adjustments",
                "goal": "optimize medications, dosing, and monitoring parameters",
                "role": "pharmacy",
            },
        ],
    },
}

# Standard prompt suffixes for consistent outputs
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

# Additional guardrail to ensure actionable outputs
ACTIONABILITY_RULE = (
    "Recommendation must be a numbered action plan (3–7 items). For each action, specify: Action, Owner, "
    "Deadline, Steps, Tools/Resources, Success Metric, Risk & Mitigation. Avoid vague language."
)

# Guardrail to ensure advisors provide advice, not only critique
ADVICE_RULE = (
    "Advisors must provide actionable advice (specific actions and why), not just critique. "
    "Include at least one concrete recommended action and an alternative with tradeoffs, when applicable."
)

# ---- Lightweight rate limiter (in‑memory, per session/user) ----
@st.cache_resource
def _rate_limit_store() -> Dict[str, deque]:
    return defaultdict(lambda: deque(maxlen=100))

def _rate_limit_ok(user_id: str, window_s: int = 60, max_calls: int = 3) -> bool:
    store = _rate_limit_store()
    now = int(time.time())
    q = store[user_id]
    # Drop timestamps outside window
    while q and now - q[0] >= window_s:
        q.popleft()
    if len(q) >= max_calls:
        return False
    q.append(now)
    return True

# Icons and subtitles per category for the hero header
CATEGORY_EMOJI = {
    "Medical": "🩺"
}

CATEGORY_SUBTITLE = {
    "Medical": "Attending physician leading a multidisciplinary discussion to form a safe, guideline‑aware diagnostic and treatment plan.",
    }

st.set_page_config(page_title="Medical Advisors", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none !important; }
    .block-container { padding-top: 0.5rem !important; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background:#eef3ff; color:#2952ff; font-size: 12px; margin-right:8px; }
    .hero-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
    .hero-subtitle { color: #6b7280; margin-top: 0; }
    .chip { display:inline-block; padding:4px 10px; margin:2px 6px 2px 0; border-radius:999px; background:#1f2937; color:#e5e7eb; font-size:12px; }
    .runbar { position:sticky; bottom:0; z-index:50; background:rgba(17,24,39,.85); backdrop-filter:saturate(180%) blur(8px); padding:10px 12px; border-top:1px solid #2b2f36; }
    .skel { background:linear-gradient(90deg, #1f2937 25%, #374151 37%, #1f2937 63%); background-size:400% 100%; animation:sh 1.2s ease-in-out infinite; border-radius:8px; }
    @keyframes sh { 0%{background-position:100% 0} 100%{background-position:0 0} }
    /* Ensure Streamlit element containers span full width */
    [data-testid="stElementContainer"] { width: 100% !important; max-width: 100% !important; margin-left: 0 !important; margin-right: 0 !important; }
    /* Solid color dividers */
    hr { border: 0 !important; height: 2px !important; background-color: #e8ebf3 !important; width: 100% !important; margin-left: 0 !important; margin-right: 0 !important; }
    [data-testid="stDivider"] hr { border: 0 !important; height: 2px !important; background-color: #e8ebf3 !important; width: 100% !important; margin-left: 0 !important; margin-right: 0 !important; }
    [role="separator"] { border: 0 !important; height: 2px !important; background-color: #e8ebf3 !important; width: 100% !important; margin-left: 0 !important; margin-right: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

left_h, right_h = st.columns([3, 1])
with left_h:
    # Advisor Category at top
    selected_category = "Medical"
    _emoji = CATEGORY_EMOJI.get(selected_category, "🩺")
    _subtitle = CATEGORY_SUBTITLE.get(selected_category, "Leader‑led expert panel tailored to the domain to deliver a clear, actionable plan.")
    st.markdown(f"<div class='hero-title'>{_emoji} Medical Advisors</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-subtitle'>{_subtitle}</div>", unsafe_allow_html=True)
    st.divider()
with right_h:
    pass

# Safety notice
st.warning(
    "Educational prototype. Not medical advice. "
    "For information only; consult qualified clinicians for decisions. "
    "Do not submit confidential, personal, or protected health information (PHI).",
    icon="⚠️",
)

# Initialize session state containers
if "clarifying_questions" not in st.session_state:
    st.session_state.clarifying_questions = []
if "clarifying_answers" not in st.session_state:
    st.session_state.clarifying_answers = {}

col_cfg, col_prev = st.columns([3, 2])
with col_cfg:
    # Settings removed from UI; set defaults here
    _default_api_key = ""
    try:
        if "OPENAI_API_KEY" in st.secrets:
            _default_api_key = st.secrets["OPENAI_API_KEY"] or ""
        # Wire NCBI_API_KEY from secrets into environment for PubMed calls
        if "NCBI_API_KEY" in st.secrets and st.secrets["NCBI_API_KEY"]:
            os.environ["NCBI_API_KEY"] = st.secrets["NCBI_API_KEY"]
    except Exception:
        _default_api_key = ""
    model = "gpt-5-mini"
    num_rounds = 2
    web_search = True
    cache_outputs = True
    fast_path = False
    user_tag = ""

# Removed Load Previous Session UI per user request

## (captcha UI moved next to Run button)

# ---- Category-specific agenda placeholders, questions, and rules ----
CATEGORY_AGENDA_PLACEHOLDER: Dict[str, str] = {
    "Medical": "e.g., 58-year-old with chest pain and dyspnea; vitals; relevant history/meds/allergies; onset/timeline; key concerns",
}

CATEGORY_QUESTIONS: Dict[str, list[str]] = {
    "Medical": [
        "What are the most likely and must‑not‑miss diagnoses given the presentation?",
        "What additional history, exam findings, and risk factors are critical to narrow the differential?",
        "What immediate stabilization steps and precautions are needed, if any?",
        "What initial labs and imaging are recommended, with rationale?",
        "What evidence‑based initial management and disposition are appropriate?",
    ],
}

CATEGORY_RULES: Dict[str, list[str]] = {
    "Medical": [
        "Educational use only; not medical advice. Verify with local guidelines and supervising clinicians.",
        "Prioritize safety: identify red flags, contraindications, and required monitoring.",
        "State diagnostic uncertainty and outline alternatives and contingencies.",
        "Cite guideline‑aligned recommendations when possible; prefer least‑harm options.",
    ],
}

 

st.subheader("Case / Problem Description")
agenda = st.text_area(
    "Describe the case",
    placeholder=CATEGORY_AGENDA_PLACEHOLDER.get(selected_category, "Describe your case..."),
    height=160,
    key=f"agenda_text_{selected_category}",
)
st.markdown("</div>", unsafe_allow_html=True)

 


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def generate_clarifying_questions(case_text: str, max_questions: int, model_name: str, category: str) -> List[str]:
    client = OpenAI()
    system = (
        "You are a domain intake assistant for a multi‑agent advisor. Read the user's case description and draft "
        "concise clarifying questions to remove ambiguity and capture missing critical details for the specified "
        "domain. Do not answer the questions. Return exactly the requested number of questions, strictly as a "
        "numbered list (1., 2., 3., …) with no preamble or commentary."
    )
    user = (
        f"Domain/category: {category}\n\n"
        f"Case description:\n\n{case_text}\n\n"
        f"Return exactly {max_questions} clarifying questions, numbered 1..{max_questions}."
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content if resp.choices else ""
    # Parse lines that look like numbered items
    questions: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1. ", "- ", "• "
        if line[0].isdigit():
            # remove leading number and dot
            q = line.split(".", 1)
            line = q[1].strip() if len(q) > 1 else line
        if line.startswith(("- ", "• ")):
            line = line[2:].strip()
        questions.append(line)
    # Deduplicate and trim to max
    uniq: List[str] = []
    for q in questions:
        if q and q not in uniq:
            uniq.append(q)
    return uniq[:max_questions]

# Uncached variant for clarifying questions
def generate_clarifying_questions_nocache(case_text: str, max_questions: int, model_name: str, category: str) -> List[str]:
    client = OpenAI()
    system = (
        "You are a domain intake assistant for a multi‑agent advisor. Read the user's case description and draft "
        "concise clarifying questions to remove ambiguity and capture missing critical details for the specified "
        "domain. Do not answer the questions. Return exactly the requested number of questions, strictly as a "
        "numbered list (1., 2., 3., …) with no preamble or commentary."
    )
    user = (
        f"Domain/category: {category}\n\n"
        f"Case description:\n\n{case_text}\n\n"
        f"Return exactly {max_questions} clarifying questions, numbered 1..{max_questions}."
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content if resp.choices else ""
    questions: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit():
            q = line.split(".", 1)
            line = q[1].strip() if len(q) > 1 else line
        if line.startswith(("- ", "• ")):
            line = line[2:].strip()
        questions.append(line)
    uniq: List[str] = []
    for q in questions:
        if q and q not in uniq:
            uniq.append(q)
    return uniq[:max_questions]


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
    # One parallel round of member advice, then a lead synthesis
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

    # Run members in parallel
    member_outputs: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(member_specs) or 1)) as pool:
        futures = []
        for m in member_specs:
            sys, usr = member_prompt(m)
            futures.append(pool.submit(_chat, model_name, sys, usr))
        for fut in futures:
            try:
                member_outputs.append(fut.result() or "")
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
        + "Produce the final consensus in markdown."
    )
    summary_md = _chat(model_name, lead_system, lead_user)
    return summary_md or "(No summary generated)"

def build_web_context(category: str, agenda_text: str) -> str:
    """Fetch brief web highlights (free DuckDuckGo) for the agenda and category to prime advisors."""
    if DDGS is None:
        return ""
    try:
        query = f"{category} background for: {agenda_text[:500]}"
        bullets: List[str] = []
        with DDGS() as ddgs:  # free, no API key
            for r in ddgs.text(query, max_results=5):
                title = r.get("title") or r.get("href") or ""
                snippet = (r.get("body") or "").strip()[:300]
                url = r.get("href") or ""
                bullets.append(f"- {title}: {snippet} ({url})")
        return ("Web search highlights:\n" + "\n".join(bullets)) if bullets else ""
    except Exception:
        return ""

def build_pubmed_context(agenda_text: str, max_results: int = 5) -> tuple[str, str]:
    """Fetch brief PubMed highlights for the agenda and return (query, markdown)."""
    try:
        import os as _os
        import json as _json
        from urllib.parse import urlencode, quote_plus as _qp
        from urllib.request import urlopen as _urlopen

        # Simple query: agenda free text + filters
        user_q = (agenda_text or "").strip()
        if not user_q:
            return ("", "")
        term = f"{user_q} AND (english[la]) AND (" + " OR ".join([
            "last 5 years[dp]",
            "systematic[sb]",
        ]) + ")"
        params = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": str(max_results),
            "term": term,
        }
        api_key = _os.environ.get("NCBI_API_KEY")
        if api_key:
            params["api_key"] = api_key
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        esearch_url = f"{base}/esearch.fcgi?{urlencode(params, quote_via=_qp)}"
        with _urlopen(esearch_url, timeout=10) as r:
            es = _json.loads(r.read().decode("utf-8"))
        idlist = (es.get("esearchresult", {}).get("idlist") or [])[:max_results]
        if not idlist:
            return (term, "")
        esum_params = {
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(idlist),
        }
        if api_key:
            esum_params["api_key"] = api_key
        esum_url = f"{base}/esummary.fcgi?{urlencode(esum_params, quote_via=_qp)}"
        with _urlopen(esum_url, timeout=10) as r:
            summary = _json.loads(r.read().decode("utf-8"))
        result = summary.get("result", {})
        items = []
        for pmid in idlist:
            rec = result.get(pmid) or {}
            title = rec.get("title") or "(no title)"
            src = rec.get("source") or ""
            yr = rec.get("pubdate") or rec.get("sortpubdate") or ""
            items.append(f"- {title} — {src} {yr} (PMID: {pmid})")
        md = ("PubMed highlights:\n" + "\n".join(items)) if items else ""
        return (term, md)
    except Exception:
        return ("", "")

# ----- Full meeting caching -----
def _serialize_agent(agent: Agent) -> Dict[str, str]:
    return {
        "title": agent.title,
        "expertise": agent.expertise,
        "goal": agent.goal,
        "role": agent.role,
        "model": agent.model,
    }


def _deserialize_agent(data: Dict[str, str]) -> Agent:
    return Agent(
        title=data["title"],
        expertise=data["expertise"],
        goal=data["goal"],
        role=data["role"],
        model=data["model"],
    )


@st.cache_data(show_spinner=True, ttl=60 * 60 * 24)
def run_meeting_cached(
    agenda: str,
    agenda_questions: tuple[str, ...],
    agenda_rules: tuple[str, ...],
    contexts: tuple[str, ...],
    num_rounds: int,
    pubmed_search: bool,
    team_lead_data: Dict[str, str],
    team_members_data: tuple[Dict[str, str], ...],
    save_name: str,
) -> str:
    save_dir = BASE_DIR / "advisor_meetings"
    save_dir.mkdir(parents=True, exist_ok=True)
    team_lead = _deserialize_agent(team_lead_data)
    # Parallelize deserialization of members
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(team_members_data) or 1)) as pool:
        team_members = tuple(pool.map(_deserialize_agent, team_members_data))

    # Map gpt-5* selections to an Assistants-supported model
    def to_assistants_model(name: str) -> str:
        n = (name or "").lower()
        if n.startswith("gpt-5"):
            return "gpt-4.1-nano"
        return name

    team_lead.model = to_assistants_model(team_lead.model)
    team_members = tuple(
        Agent(m.title, m.expertise, m.goal, m.role, to_assistants_model(m.model)) for m in team_members
    )
    summary = run_meeting(
        meeting_type="team",
        agenda=agenda,
        save_dir=save_dir,
        save_name=save_name,
        team_lead=team_lead,
        team_members=team_members,
        agenda_questions=agenda_questions,
        agenda_rules=agenda_rules,
        contexts=contexts,
        num_rounds=num_rounds,
        temperature=1.0,
        pubmed_search=pubmed_search,
        return_summary=True,
    )
    return summary

if False:
    # Basic per-user rate limit: 3 req/min
    if "session_id" not in st.session_state:
        st.session_state.session_id = secrets.token_hex(8)
    _user_id = (user_tag.strip() or st.session_state.session_id) + ":clarify"
    if not _rate_limit_ok(_user_id, window_s=60, max_calls=3):
        st.error("Rate limit reached. Please wait a minute and try again.")
        st.stop()
    # Use sidebar key if provided
    if _default_api_key:
        os.environ["OPENAI_API_KEY"] = _default_api_key
    if not _default_api_key:
        st.error("Please set your OpenAI API key in the sidebar or .env before generating questions.")
    elif not agenda.strip():
        st.error("Please enter a case description first.")
    else:
        try:
            pbar = st.progress(0, text="Preparing to generate questions…")
            pbar.progress(50, text="Generating clarifying questions…")
            questions_list = generate_clarifying_questions(agenda, 5, model, selected_category)
            st.session_state.clarifying_questions = questions_list
            # Initialize answer slots for new questions
            for q in questions_list:
                st.session_state.clarifying_answers.setdefault(q, "")
            st.success("Clarifying questions generated.")
            pbar.progress(100, text="Clarifying questions ready")
        except Exception as e:
            st.exception(e)

# Advanced settings expander
# Removed Advanced settings expander

if st.session_state.clarifying_questions:
    with st.expander("Clarifying Questions (answer to improve precision)", expanded=True):
        for q in st.session_state.clarifying_questions:
            st.session_state.clarifying_answers[q] = st.text_area(q, value=st.session_state.clarifying_answers.get(q, ""), height=70)

## Removed Agenda Questions and Rules sections from UI; defaults applied per category when running.

st.subheader("Advisors — Team Setup")

# Load category preset
_preset = CATEGORY_PRESETS[selected_category]

# Role chips (compact)
st.markdown(" ".join([f"<span class='chip'>{m['title']}</span>" for m in _preset["members"]]), unsafe_allow_html=True)

# Editable team inside expander
with st.expander("Edit team", expanded=False):
    # Team lead inputs
    lead_title = st.text_input(
        "Team Lead Title",
        value=_preset["lead"]["title"],
        key=f"lead_title_{selected_category}",
    )
    lead_expertise = st.text_input(
        "Team Lead Expertise",
        value=_preset["lead"]["expertise"],
        key=f"lead_expertise_{selected_category}",
    )
    # Dynamic member inputs
    member_titles: List[str] = []
    member_expertises: List[str] = []
    for idx, m in enumerate(_preset["members"]):
        c1, c2 = st.columns(2)
        with c1:
            member_titles.append(
                st.text_input(
                    f"Member {idx + 1} Title",
                    value=m["title"],
                    key=f"m{idx}_title_{selected_category}",
                )
            )
        with c2:
            member_expertises.append(
                st.text_input(
                    f"Member {idx + 1} Expertise",
                    value=m["expertise"],
                    key=f"m{idx}_exp_{selected_category}",
                )
            )

# CAPTCHA UI removed
# Sticky run bar (always visible)
st.markdown("<div class='runbar'>", unsafe_allow_html=True)
st.caption("Consensus typically takes 2–5 minutes.")
run_btn = st.button(
    "Run Advisors",
    type="primary",
    use_container_width=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Keyboard shortcut (Cmd/Ctrl+Enter)
components.html(
    """
    <script>
    document.addEventListener('keydown', function(e){
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        const btns = parent.document.querySelectorAll('button[kind=\"primary\"]');
        if (btns && btns.length) btns[btns.length-1].click();
      }
    });
    </script>
    """,
    height=0,
)

output_container = st.container()

# Helper to render artifacts for a given session name
def render_session_artifacts(session_name: str):
    # Prefer new advisor_meetings; fallback to medical_meetings
    save_dir = BASE_DIR / "advisor_meetings"
    md_path = save_dir / f"{session_name}.md"
    json_path = save_dir / f"{session_name}.json"
    if not md_path.exists():
        old_md = BASE_DIR / "medical_meetings" / f"{session_name}.md"
        if old_md.exists():
            md_path = old_md
    if not json_path.exists():
        old_js = BASE_DIR / "medical_meetings" / f"{session_name}.json"
        if old_js.exists():
            json_path = old_js

    st.subheader("Consensus Summary (from transcript)")
    if md_path.exists():
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            # Heuristic: show the last "### Recommendation" + below when available
            summary_start = md_content.find("### Recommendation")
            if summary_start != -1:
                st.markdown(md_content[summary_start:])
            else:
                st.markdown(md_content)
        except Exception:
            st.info("Unable to parse summary from transcript; showing full transcript below.")
    else:
        st.info("Transcript (.md) not found.")

    st.divider()
    st.subheader("Discussion Transcript")
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        with st.expander("Show full markdown transcript", expanded=True):
            st.markdown(md_content)
        st.download_button(
            label="Download transcript (.md)",
            data=md_content,
            file_name=f"{session_name}.md",
            mime="text/markdown",
        )
    else:
        st.info("Transcript (.md) not found.")

    st.subheader("Raw Messages (JSON)")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            json_content = f.read()
        try:
            import json as _json
            messages = _json.loads(json_content)
            with st.expander("Show raw messages JSON", expanded=False):
                st.json(messages)
        except Exception:
            st.code(json_content, language="json")
        st.download_button(
            label="Download messages (.json)",
            data=json_content,
            file_name=f"{session_name}.json",
            mime="application/json",
        )
    else:
        st.info("Messages (.json) not found.")


# ----- Web session naming and pruning helpers -----
def _get_web_session_basenames(save_dir: Path) -> list[str]:
    names = set()
    for p in list(save_dir.glob("web_*.md")) + list(save_dir.glob("web_*.json")):
        names.add(p.stem)
    return sorted(names)


def _make_next_web_session_name(save_dir: Path) -> str:
    import re
    max_idx = 0
    for stem in _get_web_session_basenames(save_dir):
        m = re.match(r"web_(\d+)$", stem)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    return f"web_{max_idx + 1:05d}"


def _prune_web_sessions(save_dir: Path, max_sessions: int = 5) -> None:
    from typing import Tuple
    stems = _get_web_session_basenames(save_dir)
    if len(stems) <= max_sessions:
        return
    # Compute last modified time per session (max of md/json)
    def mtime_for(stem: str) -> float:
        md = save_dir / f"{stem}.md"
        js = save_dir / f"{stem}.json"
        times: list[float] = []
        if md.exists():
            times.append(md.stat().st_mtime)
        if js.exists():
            times.append(js.stat().st_mtime)
        return max(times) if times else 0.0

    stems_sorted = sorted(stems, key=mtime_for, reverse=True)
    to_delete = stems_sorted[max_sessions:]
    for stem in to_delete:
        for ext in (".md", ".json"):
            path = save_dir / f"{stem}{ext}"
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

if run_btn:
    # Basic per-user rate limit: 3 req/min
    if "session_id" not in st.session_state:
        st.session_state.session_id = secrets.token_hex(8)
    _user_id = (user_tag.strip() or st.session_state.session_id) + ":run"
    if not _rate_limit_ok(_user_id, window_s=60, max_calls=3):
        st.error("Rate limit reached. Please wait a minute and try again.")
        st.stop()
    if not _default_api_key:
        st.error("OpenAI API key missing. Set it in .streamlit/secrets.toml or the environment.")
    elif not agenda.strip():
        st.error("Please provide a case description (agenda).")
    else:
        if _default_api_key:
            os.environ["OPENAI_API_KEY"] = _default_api_key
        # Build clarifications context
        clarifications_text = ""
        if st.session_state.clarifying_questions:
            qa_lines = ["Clarifications provided by user:"]
            for cq in st.session_state.clarifying_questions:
                ans = st.session_state.clarifying_answers.get(cq, "").strip()
                if ans:
                    qa_lines.append(f"- {cq}\n  Answer: {ans}")
                else:
                    qa_lines.append(f"- {cq}\n  Answer: (not provided)")
            clarifications_text = "\n".join(qa_lines)
        # Optional web search context (DuckDuckGo)
        web_context_text = build_web_context(selected_category, agenda) if web_search else ""
        # Always fetch PubMed context and log query
        pm_query, pm_md = build_pubmed_context(agenda)
        if pm_query:
            with st.expander("PubMed query and highlights", expanded=False):
                st.code(f"Query: {pm_query}", language="text")
                if pm_md:
                    st.code(pm_md, language="markdown")
        if web_context_text:
            with st.expander("Web search highlights (DuckDuckGo)", expanded=False):
                st.code(web_context_text, language="markdown")
        team_lead = Agent(
            title=lead_title,
            expertise=lead_expertise,
            goal=_preset["lead"]["goal"] + PROMPT_GOAL_SUFFIX_LEAD,
            role=_preset["lead"]["role"],
            model=model,
        )
        # Build members from dynamic inputs while preserving preset goals/roles (parallelized)
        def _make_member(i: int, m: Dict[str, str]) -> Agent:
            title_i = member_titles[i] if i < len(member_titles) else m["title"]
            exp_i = member_expertises[i] if i < len(member_expertises) else m["expertise"]
            return Agent(
                title=title_i,
                expertise=exp_i,
                goal=m["goal"] + PROMPT_GOAL_SUFFIX_MEMBER,
                role=m["role"],
                model=model,
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(_preset["members"]) or 1)) as pool:
            team_members = tuple(pool.map(lambda idx_m: _make_member(idx_m[0], idx_m[1]), enumerate(_preset["members"])))
        agenda_qs = tuple(CATEGORY_QUESTIONS[selected_category])
        agenda_rules = tuple(list(CATEGORY_RULES[selected_category]) + [ACTIONABILITY_RULE, ADVICE_RULE])
        save_dir = BASE_DIR / "advisor_meetings"
        save_dir.mkdir(parents=True, exist_ok=True)
        with st.spinner("Running advisors… this usually takes 2–5 minutes"):
            try:
                # Auto-number session name and prune to keep the latest 5 web_* sessions
                auto_save_name = _make_next_web_session_name(save_dir)
                step = st.empty()
                bar = st.progress(0, text="Configuring advisors…")
                bar.progress(20, text="Assembling agenda and rules…")
                if fast_path:
                    bar.progress(40, text="Starting fast path (Completions)…")
                    # Build lead/member specs for fast path
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
                        contexts=tuple(x for x in (clarifications_text, web_context_text, pm_md) if x),
                        lead_spec=lead_spec,
                        member_specs=member_specs,
                        model_name=model,
                        num_rounds=int(num_rounds),
                    )
                    # Save artifacts for fast path so Transcript/Raw JSON tabs work
                    try:
                        md_path = save_dir / f"{auto_save_name}.md"
                        md_content = (
                            "# Medical Advisors — Transcript (Fast Path)\n\n"
                            f"## Agenda\n\n{agenda.strip()}\n\n"
                            + (f"## Clarifications\n\n{clarifications_text}\n\n" if clarifications_text else "")
                            + (f"## Web highlights\n\n{web_context_text}\n\n" if web_context_text else "")
                            + "## Consensus Summary\n\n"
                            + (summary or "(No summary generated)")
                        )
                        with open(md_path, "w", encoding="utf-8") as f:
                            f.write(md_content)

                        json_path = save_dir / f"{auto_save_name}.json"
                        messages_obj = {
                            "mode": "fast",
                            "agenda": agenda,
                            "clarifications": clarifications_text or "",
                            "web_context": web_context_text or "",
                            "team_lead": lead_spec,
                            "team_members": list(member_specs),
                            "summary_md": summary,
                        }
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(messages_obj, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                elif cache_outputs:
                    bar.progress(40, text="Starting cached team meeting…")
                    summary = run_meeting_cached(
                        agenda=agenda,
                        agenda_questions=agenda_qs,
                        agenda_rules=agenda_rules,
                        contexts=tuple(x for x in (clarifications_text, web_context_text, pm_md) if x),
                        num_rounds=st.session_state.get("num_rounds_override", None) or int(num_rounds),
                        pubmed_search=True,
                        team_lead_data=_serialize_agent(team_lead),
                        team_members_data=tuple(_serialize_agent(m) for m in team_members),
                        save_name=auto_save_name,
                    )
                else:
                    # Uncached path mirrors run_meeting_cached; apply same mapping
                    def to_assistants_model(name: str) -> str:
                        n = (name or "").lower()
                        if n.startswith("gpt-5"):
                            return "gpt-4.1-nano"
                        return name
                    team_lead.model = to_assistants_model(team_lead.model)
                    team_members_live = tuple(
                        Agent(m.title, m.expertise, m.goal, m.role, to_assistants_model(m.model)) for m in team_members
                    )
                    bar.progress(40, text="Starting live team meeting…")
                    summary = run_meeting(
                        meeting_type="team",
                        agenda=agenda,
                        save_dir=save_dir,
                        save_name=auto_save_name,
                        team_lead=team_lead,
                        team_members=team_members_live,
                        agenda_questions=agenda_qs,
                        agenda_rules=agenda_rules,
                        contexts=tuple(x for x in (clarifications_text, web_context_text, pm_md) if x),
                        num_rounds=st.session_state.get("num_rounds_override", None) or int(num_rounds),
                        temperature=1.0,
                        pubmed_search=True,
                        return_summary=True,
                    )
                bar.progress(80, text="Summarizing consensus…")
                _prune_web_sessions(save_dir, max_sessions=5)
                bar.progress(100, text="Done")
                tabs = st.tabs(["🧭 Consensus Summary", "🗒️ Transcript", "🧱 Raw JSON"]) 
                with tabs[0]:
                    st.markdown('<div id="consensus-summary-anchor"></div>', unsafe_allow_html=True)
                    output_container.subheader("Consensus Summary")
                    output_container.markdown(summary)
                    # Auto-scroll to summary when ready
                    components.html(
                        """
                        <script>
                        setTimeout(function(){
                          var el = parent.document.querySelector('#consensus-summary-anchor');
                          if (el && el.scrollIntoView) {
                            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                          }
                        }, 100);
                        </script>
                        """,
                        height=0,
                    )
                with tabs[1]:
                    # Transcript
                    render_session_artifacts(auto_save_name)
                with tabs[2]:
                    # Only render JSON section
                    save_dir = BASE_DIR / "advisor_meetings"
                    json_path = save_dir / f"{auto_save_name}.json"
                    if json_path.exists():
                        with open(json_path, "r", encoding="utf-8") as f:
                            json_content = f.read()
                        try:
                            messages = json.loads(json_content)
                            st.json(messages)
                        except Exception:
                            st.code(json_content, language="json")
                    else:
                        st.info("Messages (.json) not found.")
            except Exception as e:
                st.exception(e)
