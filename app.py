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
from advisors.prompts import (
    PROMPT_GOAL_SUFFIX_LEAD,
    PROMPT_GOAL_SUFFIX_MEMBER,
    ACTIONABILITY_RULE,
    ADVICE_RULE,
)
from advisors.services.meeting_fast import run_fast_completions

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
                "title": "Infectious Diseases",
                "expertise": "antimicrobials, sepsis, source control",
                "goal": "recommend diagnostic studies and targeted empiric therapy",
                "role": "ID",
            },
            {
                "title": "Clinical Pharmacist",
                "expertise": "dosing, interactions, renal/hepatic adjustments",
                "goal": "optimize medications, dosing, and monitoring parameters",
                "role": "pharmacy",
            },
        ],
    },
    "Entrepreneur ideas": {
        "lead": {
            "title": "Startup Mentor",
            "expertise": "idea maze, founder-market fit, early traction",
            "goal": "drive the team to a crisp problem statement, wedge, and MVP scope",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Product Strategy",
                "expertise": "discovery, prioritization, outcomes over output",
                "goal": "define MVP, success metrics, and ruthless prioritization",
                "role": "product strategist",
            },
            {
                "title": "Growth",
                "expertise": "acquisition loops, retention, network effects",
                "goal": "propose growth loops and testable hypotheses",
                "role": "growth lead",
            },
            {
                "title": "Fundraising",
                "expertise": "round sequencing, pitch, cap table, investor fit",
                "goal": "shape the narrative, milestones, and target investor list",
                "role": "fundraising advisor",
            },
            {
                "title": "Ops & Finance",
                "expertise": "unit economics, cash runway, ops scaling",
                "goal": "stress-test unit economics and operating model",
                "role": "CFO/ops",
            },
            {
                "title": "Legal (Startup Counsel)",
                "expertise": "company setup, IP, contracts, employment",
                "goal": "highlight legal risks and recommended defaults",
                "role": "legal counsel",
            },
        ],
    },
    "NFL fantasy player selection": {
        "lead": {
            "title": "Lead Fantasy Analyst",
            "expertise": "macro narratives, roster construction, game theory",
            "goal": "synthesize data and scouting to produce start/sit, draft, and waiver decisions",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Data & Modeling",
                "expertise": "air yards, target shares, projections, simulations",
                "goal": "produce projections and uncertainty bands; flag buy/sell signals",
                "role": "modeling lead",
            },
            {
                "title": "Injury Analyst",
                "expertise": "injury timelines, performance impact, re-injury risk",
                "goal": "quantify availability and performance deltas by injury",
                "role": "injury specialist",
            },
            {
                "title": "Matchups & Weather",
                "expertise": "defensive schemes, pace, Vegas lines, weather",
                "goal": "adjust projections using matchups and environmental factors",
                "role": "context analyst",
            },
            {
                "title": "DFS & Waivers",
                "expertise": "ownership, leverage, FAAB bidding, schedule exploitation",
                "goal": "recommend DFS stacks and weekly waiver priorities",
                "role": "DFS/waiver strategist",
            },
            {
                "title": "Risk & Portfolio",
                "expertise": "exposure caps, diversification, contest selection",
                "goal": "manage risk across lineups and season-long portfolios",
                "role": "portfolio manager",
            },
        ],
    },
    "Personal finance and investing": {
        "lead": {
            "title": "Portfolio Manager",
            "expertise": "asset allocation, index investing, rebalancing",
            "goal": "deliver a low-cost, diversified plan matched to horizon and risk",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Tax Planner (CPA)",
                "expertise": "tax-advantaged accounts, harvesting, entity selection",
                "goal": "minimize lifetime taxes; coordinate with investment plan",
                "role": "tax strategy",
            },
            {
                "title": "Retirement Planner",
                "expertise": "safe withdrawal, annuities, social security timing",
                "goal": "design sustainable retirement income with guardrails",
                "role": "retirement",
            },
            {
                "title": "Risk Actuary",
                "expertise": "insurance, tail risks, liability management",
                "goal": "size emergency fund and insurance; stress-test plan",
                "role": "risk management",
            },
            {
                "title": "Real Estate",
                "expertise": "buy/hold, cash flow, leverage, local markets",
                "goal": "evaluate real-estate as a complement to the portfolio",
                "role": "real estate",
            },
            {
                "title": "Behavioral Finance",
                "expertise": "biases, nudges, commitment devices",
                "goal": "reduce behavioral errors and improve adherence",
                "role": "behavioral advisor",
            },
        ],
    },
    "Legal strategy and contracts": {
        "lead": {
            "title": "General Counsel",
            "expertise": "corporate law, negotiation, dispute resolution",
            "goal": "balance risk and business goals; define legal strategy",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Contracts Negotiator",
                "expertise": "commercial terms, risk allocation, remedies",
                "goal": "craft enforceable, business-aligned terms",
                "role": "contracts",
            },
            {
                "title": "IP/Patent Counsel",
                "expertise": "patentability, prior art, licensing",
                "goal": "protect and leverage IP while avoiding pitfalls",
                "role": "intellectual property",
            },
            {
                "title": "Employment Law",
                "expertise": "hiring, termination, equity, compliance",
                "goal": "minimize labor risk and ensure compliant practices",
                "role": "employment",
            },
            {
                "title": "Privacy & Compliance",
                "expertise": "GDPR, CCPA, data governance",
                "goal": "define lawful bases and implement safeguards",
                "role": "privacy/compliance",
            },
            {
                "title": "Litigation Strategist",
                "expertise": "venue, motions, settlement strategy",
                "goal": "optimize litigation posture and alternatives",
                "role": "litigation",
            },
        ],
    },
    "Software architecture and DevOps": {
        "lead": {
            "title": "Chief Architect",
            "expertise": "domain modeling, evolutionary architecture, governance",
            "goal": "align architecture with business goals; reduce complexity and risk",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Cloud & Infrastructure",
                "expertise": "AWS/GCP/Azure, cost, multi-tenancy",
                "goal": "produce a scalable, cost-aware infra plan",
                "role": "cloud/infra",
            },
            {
                "title": "Security",
                "expertise": "threat modeling, AppSec, incident response",
                "goal": "embed security by design with actionable controls",
                "role": "security lead",
            },
            {
                "title": "Data & ML Platform",
                "expertise": "data contracts, feature stores, model ops",
                "goal": "design reliable data/ML pipelines and SLAs",
                "role": "data/ML",
            },
            {
                "title": "SRE & Reliability",
                "expertise": "SLIs/SLOs, capacity, chaos engineering",
                "goal": "set reliability targets and runbooks",
                "role": "SRE",
            },
            {
                "title": "QA & Automation",
                "expertise": "test strategy, CI/CD, quality gates",
                "goal": "ensure fast feedback and defect prevention",
                "role": "QA",
            },
        ],
    },
    "Product design and UX": {
        "lead": {
            "title": "Head of Design",
            "expertise": "human-centered design, usability, systems thinking",
            "goal": "balance desirability, feasibility, and viability",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "UX Research",
                "expertise": "generative/evaluative research, JTBD, synthesis",
                "goal": "identify user needs and opportunity areas",
                "role": "research",
            },
            {
                "title": "Interaction Design",
                "expertise": "flows, IA, patterns, affordances",
                "goal": "define interaction models and prototypes",
                "role": "interaction design",
            },
            {
                "title": "Content Strategy",
                "expertise": "voice/tone, IA, microcopy",
                "goal": "craft clear, consistent, accessible content",
                "role": "content",
            },
            {
                "title": "Accessibility",
                "expertise": "WCAG, inclusive UX, audits",
                "goal": "ensure compliance and inclusive experiences",
                "role": "accessibility",
            },
            {
                "title": "Design Systems",
                "expertise": "tokens, components, governance",
                "goal": "create scalable, consistent UI system",
                "role": "design systems",
            },
        ],
    },
    "Marketing and growth": {
        "lead": {
            "title": "Chief Marketing Officer",
            "expertise": "positioning, brand, growth strategy",
            "goal": "craft a coherent strategy from brand to performance",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Performance & Growth",
                "expertise": "paid media, CRO, growth loops",
                "goal": "design scalable acquisition and retention loops",
                "role": "growth",
            },
            {
                "title": "SEO",
                "expertise": "technical SEO, content strategy, international",
                "goal": "build durable organic growth program",
                "role": "SEO lead",
            },
            {
                "title": "Brand",
                "expertise": "category design, distinctiveness, memory structures",
                "goal": "create distinctive brand assets and memory cues",
                "role": "brand",
            },
            {
                "title": "Lifecycle & CRM",
                "expertise": "segmentation, messaging, automation",
                "goal": "increase activation, retention, and LTV",
                "role": "CRM",
            },
            {
                "title": "Marketing Analytics",
                "expertise": "attribution, incrementality, MMM",
                "goal": "measure causality and guide budget allocation",
                "role": "analytics",
            },
        ],
    },
    "Academic research and writing": {
        "lead": {
            "title": "Principal Investigator",
            "expertise": "causal inference, research design, ethics",
            "goal": "ensure methodological rigor and meaningful contribution",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Literature Synthesis",
                "expertise": "systematic reviews, PRISMA, search strategies",
                "goal": "map prior work and extract evidence consistently",
                "role": "literature review",
            },
            {
                "title": "Methods & Study Design",
                "expertise": "bias control, sampling, preregistration",
                "goal": "design valid, reproducible studies",
                "role": "methods",
            },
            {
                "title": "Statistics",
                "expertise": "Bayesian and frequentist inference, power, modeling",
                "goal": "analyze data and quantify uncertainty",
                "role": "statistics",
            },
            {
                "title": "Writing & Editing",
                "expertise": "structure, clarity, argumentation",
                "goal": "craft clear manuscripts and rebuttals",
                "role": "editor",
            },
            {
                "title": "Citation Management",
                "expertise": "Zotero, LaTeX, styles, reproducible builds",
                "goal": "maintain clean references and templates",
                "role": "tooling",
            },
        ],
    },
    "Career coaching and hiring": {
        "lead": {
            "title": "Head of Talent",
            "expertise": "hiring strategy, org design, performance systems",
            "goal": "optimize candidate-market fit and offer outcomes",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Resume & LinkedIn",
                "expertise": "positioning, achievements, ATS optimization",
                "goal": "build a compelling, search-friendly profile",
                "role": "resume/branding",
            },
            {
                "title": "Interview Coach (Tech)",
                "expertise": "DS&A, system design, behavioral",
                "goal": "plan study path, drills, and mock interviews",
                "role": "interview prep",
            },
            {
                "title": "Technical Evaluator",
                "expertise": "code review, career ladders, hiring rubrics",
                "goal": "assess technical fit and level accurately",
                "role": "technical evaluation",
            },
            {
                "title": "Compensation & Offers",
                "expertise": "levels, equity, negotiation strategy",
                "goal": "optimize offer structure and negotiation plan",
                "role": "compensation",
            },
            {
                "title": "Networking & Referrals",
                "expertise": "direct outreach, value-first networking",
                "goal": "build an effective warm-intro pipeline",
                "role": "networking coach",
            },
        ],
    },
    "Fashion": {
        "lead": {
            "title": "Creative Director",
            "expertise": "brand vision, collection curation, storytelling",
            "goal": "orchestrate a cohesive, on‑brand collection that balances desirability, feasibility, and margin",
            "role": "team lead and final arbiter",
        },
        "members": [
            {
                "title": "Trend & Market Research",
                "expertise": "trend forecasting, cultural analysis, consumer insights",
                "goal": "surface relevant trends, references, and whitespace in the market",
                "role": "trend research",
            },
            {
                "title": "Design & Materials",
                "expertise": "garment construction, textiles, fit, tech packs",
                "goal": "translate concepts into feasible designs with appropriate materials and fits",
                "role": "design lead",
            },
            {
                "title": "Merchandising & Pricing",
                "expertise": "assortment planning, price architecture, margin targets",
                "goal": "define assortment, price points, and margin structure across SKUs",
                "role": "merchandising",
            },
            {
                "title": "Supply Chain & Sustainability",
                "expertise": "sourcing, MOQs, lead times, ethical manufacturing",
                "goal": "propose sourcing plan, timelines, and sustainability tradeoffs",
                "role": "operations & sustainability",
            },
            {
                "title": "Brand & Visual Identity",
                "expertise": "creative direction, lookbooks, content, campaigns",
                "goal": "craft visual language and go‑to‑market assets to launch the collection",
                "role": "brand/visuals",
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
    "Medical": "🩺",
    "Entrepreneur ideas": "🚀",
    "NFL fantasy player selection": "🏈",
    "Personal finance and investing": "💰",
    "Legal strategy and contracts": "⚖️",
    "Software architecture and DevOps": "🛠️",
    "Product design and UX": "🎨",
    "Marketing and growth": "📈",
    "Academic research and writing": "🎓",
    "Career coaching and hiring": "👔",
    "Fashion": "👗",
}

CATEGORY_SUBTITLE = {
    "Medical": "Attending physician leading a multidisciplinary discussion to form a safe, guideline‑aware diagnostic and treatment plan.",
    "Entrepreneur ideas": "Mentor‑led panel aligning product, growth, financing, operations, and legal to ship a compelling MVP and plan milestones.",
    "NFL fantasy player selection": "Lead analyst synthesizing projections, injuries, matchups, and risk to drive weekly start/sit, waivers, and DFS strategy.",
    "Personal finance and investing": "Portfolio manager coordinating tax, retirement, risk, real estate, and behavior for a durable, low‑cost plan.",
    "Legal strategy and contracts": "General counsel orchestrating contracts, IP, employment, privacy, and litigation strategy to balance risk and outcomes.",
    "Software architecture and DevOps": "Chief architect aligning cloud, security, data/ML, reliability, and QA for evolvable, efficient systems.",
    "Product design and UX": "Head of design integrating research, interaction, content, accessibility, and systems for measurable UX outcomes.",
    "Marketing and growth": "CMO unifying brand and performance with SEO, lifecycle, and analytics to build durable growth.",
    "Academic research and writing": "PI guiding literature synthesis, methods, statistics, writing, and citations for rigorous, reproducible work.",
    "Career coaching and hiring": "Head of talent coordinating resume, interviews, evaluation, compensation, and networking to land offers.",
    "Fashion": "Creative director leading trend, design, merchandising, ops, and brand to deliver a cohesive collection.",
}

st.set_page_config(page_title="Advisors", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stHeader"] { display: none; }
    [data-testid="stToolbar"] { display: none !important; }
    .block-container { padding-top: 0.5rem !important; }
    .card { padding: 1rem 1.2rem; border-radius: 8px; border: 1px solid #e8ebf3; margin-bottom: 1rem; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background:#eef3ff; color:#2952ff; font-size: 12px; margin-right:8px; }
    .hero-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
    .hero-subtitle { color: #6b7280; margin-top: 0; }
    .chip { display:inline-block; padding:4px 10px; margin:2px 6px 2px 0; border-radius:999px; background:#1f2937; color:#e5e7eb; font-size:12px; }
    .runbar { position:sticky; bottom:0; z-index:50; background:rgba(17,24,39,.85); backdrop-filter:saturate(180%) blur(8px); padding:10px 12px; border-top:1px solid #2b2f36; }
    .skel { background:linear-gradient(90deg, #1f2937 25%, #374151 37%, #1f2937 63%); background-size:400% 100%; animation:sh 1.2s ease-in-out infinite; border-radius:8px; }
    @keyframes sh { 0%{background-position:100% 0} 100%{background-position:0 0} }
    </style>
    """,
    unsafe_allow_html=True,
)

left_h, right_h = st.columns([3, 1])
with left_h:
    # Advisor Category at top
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    selected_category = "Medical"
    st.markdown("</div>", unsafe_allow_html=True)
    _emoji = CATEGORY_EMOJI.get(selected_category, "🧠")
    _subtitle = CATEGORY_SUBTITLE.get(selected_category, "Leader‑led expert panel tailored to the domain to deliver a clear, actionable plan.")
    st.markdown(f"<div class='hero-title'>{_emoji} Advisors — {selected_category}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-subtitle'>{_subtitle}</div>", unsafe_allow_html=True)
with right_h:
    pass

# Safety notice
st.warning(
    "Educational prototype. Not professional advice (medical, legal, financial, etc.). "
    "For information only; consult qualified professionals for decisions. "
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
    except Exception:
        _default_api_key = ""
    model = "gpt-5-nano"
    num_rounds = 1
    web_search = True
    cache_outputs = True
    fast_path = True
    user_tag = ""

# Removed Load Previous Session UI per user request

## (captcha UI moved next to Run button)

# ---- Category-specific agenda placeholders, questions, and rules ----
CATEGORY_AGENDA_PLACEHOLDER: Dict[str, str] = {
    "Medical": "e.g., 58-year-old with chest pain and dyspnea; vitals; relevant history/meds/allergies; onset/timeline; key concerns",
    "Entrepreneur ideas": "e.g., Idea: AI copilot for X market; target users are...",
    "NFL fantasy player selection": "e.g., 12-team PPR, Week 5 start/sit + waivers.\nStart/Sit: DK Metcalf vs DeVonta Smith; RB2: Rhamondre Stevenson vs James Conner\nWaivers: Nico Collins, Tank Dell\nInjuries: Cooper Kupp (hamstring)",
    "Personal finance and investing": "e.g., Age 35, saving for retirement and down payment; risk tolerance...",
    "Legal strategy and contracts": "e.g., SaaS MSA negotiation with enterprise; key terms to optimize...",
    "Software architecture and DevOps": "e.g., Designing a multi-tenant SaaS on AWS with strict cost targets...",
    "Product design and UX": "e.g., Improving onboarding for a B2B app; current drop-off points...",
    "Marketing and growth": "e.g., SaaS with flat growth; goal: increase activation and retention...",
    "Academic research and writing": "e.g., Plan a study on X; prior literature suggests...",
    "Career coaching and hiring": "e.g., Transitioning to senior engineer role in 6 months; gaps are...",
    "Fashion": "e.g., Concept for Spring/Summer collection; target customer, price band, and inspiration...",
}

CATEGORY_QUESTIONS: Dict[str, list[str]] = {
    "Medical": [
        "What are the most likely and must‑not‑miss diagnoses given the presentation?",
        "What additional history, exam findings, and risk factors are critical to narrow the differential?",
        "What immediate stabilization steps and precautions are needed, if any?",
        "What initial labs and imaging are recommended, with rationale?",
        "What evidence‑based initial management and disposition are appropriate?",
    ],
    "Entrepreneur ideas": [
        "What specific user problem is painful and frequent enough to solve now?",
        "What is the narrow wedge/MVP and how will we validate it quickly?",
        "What are the primary growth loops and activation metrics?",
        "What milestones de-risk fundraising and sequencing of capital?",
        "What legal/compliance or operational risks must be mitigated?",
    ],
    "NFL fantasy player selection": [
        "Start/Sit example: DK Metcalf vs DeVonta Smith — who and why?",
        "RB2 choice: Rhamondre Stevenson vs James Conner — projections, matchup, usage?",
        "Waiver priorities: Nico Collins, Tank Dell — expected value and FAAB bids?",
        "DFS stacks/leverage: which QB-WR/TE pairings and bring-backs look optimal?",
        "News/weather adjustments (e.g., Cooper Kupp hamstring) that materially move projections?",
    ],
    "Personal finance and investing": [
        "What is the appropriate asset allocation given horizon and risk?",
        "How can we minimize taxes across accounts this year and long-term?",
        "What insurance and emergency fund levels are prudent?",
        "What real-estate considerations complement the portfolio?",
        "What behavioral risks may derail the plan and how to mitigate them?",
    ],
    "Legal strategy and contracts": [
        "What is the desired business outcome and acceptable risk envelope?",
        "Which contract terms most impact risk allocation and remedies?",
        "What IP, privacy, and employment issues require special handling?",
        "What negotiation strategy, concessions, and fallbacks should we use?",
        "What litigation or dispute strategies should be pre-planned?",
    ],
    "Software architecture and DevOps": [
        "What architecture best aligns with domain and future evolution?",
        "What SLIs/SLOs, capacity, and reliability targets are appropriate?",
        "What data/ML platform choices support current and near-term needs?",
        "What security controls and threat model are required by design?",
        "How do we ensure testability, CI/CD, and cost efficiency?",
    ],
    "Product design and UX": [
        "Who are the target users and unmet needs (JTBD)?",
        "What onboarding and interaction changes will move key metrics?",
        "What content and IA adjustments improve clarity and success?",
        "How do we ensure accessibility (WCAG) and inclusivity?",
        "What design system components/tokens are required?",
    ],
    "Marketing and growth": [
        "What positioning and category narrative create distinctiveness?",
        "Which growth loops (acquisition/activation/retention) should we build?",
        "What SEO and content strategies drive durable demand?",
        "How will we measure incrementality and allocate budget?",
        "What lifecycle/CRM programs increase LTV and reduce churn?",
    ],
    "Academic research and writing": [
        "What is the precise research question and causal claim?",
        "What prior literature and evidence map inform design?",
        "What study design and methods minimize bias and maximize power?",
        "What analysis plan and uncertainty quantification will we use?",
        "What writing and submission strategy fits the target venue?",
    ],
    "Career coaching and hiring": [
        "What target role and level are realistic in the chosen timeframe?",
        "What skill gaps and projects best demonstrate readiness?",
        "What resume/LinkedIn changes improve signal and discovery?",
        "What interview preparation plan yields the highest ROI?",
        "What compensation, negotiation, and networking strategy should we use?",
    ],
    "Fashion": [
        "What is the concept, muse, and story that unify the collection?",
        "What assortment, color/material palette, and price architecture are appropriate?",
        "What sourcing plan, MOQs, and timelines fit the launch window?",
        "What fit, construction, and QA considerations are critical?",
        "What brand/visual strategy and channel plan will launch effectively?",
    ],
}

CATEGORY_RULES: Dict[str, list[str]] = {
    "Medical": [
        "Educational use only; not medical advice. Verify with local guidelines and supervising clinicians.",
        "Prioritize safety: identify red flags, contraindications, and required monitoring.",
        "State diagnostic uncertainty and outline alternatives and contingencies.",
        "Cite guideline‑aligned recommendations when possible; prefer least‑harm options.",
    ],
    "Entrepreneur ideas": [
        "Prefer falsifiable hypotheses and smallest viable tests.",
        "Quantify unit economics and concentrate on retention over vanity metrics.",
        "State risks and regulatory constraints explicitly.",
        "Avoid generic platitudes; propose concrete next steps.",
    ],
    "NFL fantasy player selection": [
        "Tie recommendations to projections, injury reports, and matchup data.",
        "State uncertainty ranges and alternatives (floor/ceiling).",
        "Avoid hindsight bias; document assumptions and news dependencies.",
        "Respect league format and scoring (e.g., PPR vs standard).",
    ],
    "Personal finance and investing": [
        "Not financial advice; provide educational guidance only.",
        "Favor low-cost, diversified strategies; disclose assumptions and risks.",
        "Consider taxes, account types, and time horizon explicitly.",
        "Quantify ranges rather than false precision.",
    ],
    "Legal strategy and contracts": [
        "Not legal advice; educational discussion only.",
        "Highlight jurisdictional differences and compliance requirements.",
        "Clarify tradeoffs between risk and business outcomes.",
        "Avoid conclusory statements without stating assumptions.",
    ],
    "Software architecture and DevOps": [
        "Prefer simple, evolvable designs; manage complexity explicitly.",
        "State reliability/cost/security tradeoffs and SLO impacts.",
        "Document assumptions about scale and failure modes.",
        "Provide concrete runbooks and testing strategies.",
    ],
    "Product design and UX": [
        "Ground decisions in user research and usability heuristics.",
        "Ensure accessibility and inclusive design.",
        "Favor prototypes and measurable experiments.",
        "Keep copy clear and consistent with voice/tone.",
    ],
    "Marketing and growth": [
        "Avoid vanity metrics; focus on causal attribution and LTV.",
        "Propose concrete experiments with measurement plans.",
        "Consider brand consistency alongside performance goals.",
        "State data sources and quality caveats.",
    ],
    "Academic research and writing": [
        "Follow ethical standards and preregistration when applicable.",
        "Report uncertainty, not just point estimates.",
        "Cite sources precisely; avoid overgeneralization.",
        "Ensure reproducibility of analyses and references.",
    ],
    "Career coaching and hiring": [
        "Avoid confidential employer information; respect NDAs.",
        "Base advice on public rubrics and role expectations.",
        "Encourage evidence-backed preparation and practice.",
        "Tailor plans to constraints and timeline.",
    ],
    "Fashion": [
        "Not professional advice; validate with sampling and fit tests.",
        "State assumptions on costs, MOQs, and lead times explicitly.",
        "Consider sustainability, compliance, and ethical sourcing.",
        "Propose concrete next steps with calendar checkpoints.",
    ],
}

 

st.markdown("<div class='card'>", unsafe_allow_html=True)
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

# Inline CAPTCHA (above button)
if "captcha_sum" not in st.session_state:
    a, b = random.randint(1, 9), random.randint(1, 9)
    st.session_state.captcha_q = f"What is {a} + {b}?"
    st.session_state.captcha_sum = a + b
    st.session_state.captcha_ok = False
st.caption("Quick check")
cap_ans = st.text_input(st.session_state.captcha_q, key="captcha_ans_main")
if st.button("Verify", key="captcha_verify_btn_main"):
    try:
        st.session_state.captcha_ok = int(cap_ans) == st.session_state.captcha_sum
        if st.session_state.captcha_ok:
            st.success("Verified")
        else:
            st.error("Try again")
    except Exception:
        st.error("Enter a number")

# Sticky run bar
st.markdown("<div class='runbar'>", unsafe_allow_html=True)
run_btn = st.button("Run Advisors", type="primary", use_container_width=True,
                    disabled=not bool(st.session_state.get("captcha_ok", False)))
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
    # Enforce simple CAPTCHA before running
    if not st.session_state.get("captcha_ok"):
        st.error("Please verify the CAPTCHA in the sidebar before running.")
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
        with st.spinner("Running advisors... this may take a few minutes"):
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
                        contexts=tuple(x for x in (clarifications_text, web_context_text) if x),
                        lead_spec=lead_spec,
                        member_specs=member_specs,
                        model_name=model,
                        num_rounds=int(num_rounds),
                    )
                elif cache_outputs:
                    bar.progress(40, text="Starting cached team meeting…")
                    summary = run_meeting_cached(
                        agenda=agenda,
                        agenda_questions=agenda_qs,
                        agenda_rules=agenda_rules,
                        contexts=tuple(x for x in (clarifications_text, web_context_text) if x),
                        num_rounds=st.session_state.get("num_rounds_override", None) or int(num_rounds),
                        pubmed_search=False,
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
                        contexts=tuple(x for x in (clarifications_text, web_context_text) if x),
                        num_rounds=st.session_state.get("num_rounds_override", None) or int(num_rounds),
                        temperature=1.0,
                        pubmed_search=False,
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
