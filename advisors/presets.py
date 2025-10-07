"""Domain-specific advisor presets shared across the app and CLI entry points."""

from typing import Dict, List

CATEGORY_PRESETS: Dict[str, Dict] = {
    "Medical": {
        "lead": {
            "title": "Internal Medicine",
            "expertise": "differential diagnosis, inpatient management",
            "goal": "construct prioritized differential and inpatient plan",
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

CATEGORY_AGENDA_PLACEHOLDER: Dict[str, str] = {
    "Medical": "e.g., 58-year-old with chest pain and dyspnea; vitals; relevant history/meds/allergies; onset/timeline; key concerns",
}

CATEGORY_QUESTIONS: Dict[str, List[str]] = {
    "Medical": [
        "What are the most likely and must-not-miss diagnoses given the presentation?",
        "What additional history, exam findings, and risk factors are critical to narrow the differential?",
        "What immediate stabilization steps and precautions are needed, if any?",
        "What initial labs and imaging are recommended, with rationale?",
        "What evidence-based initial management and disposition are appropriate?",
    ],
}

CATEGORY_RULES: Dict[str, List[str]] = {
    "Medical": [
        "Educational use only; not medical advice. Verify with local guidelines and supervising clinicians.",
        "Prioritize safety: identify red flags, contraindications, and required monitoring.",
        "State diagnostic uncertainty and outline alternatives and contingencies.",
        "Cite guideline-aligned recommendations when possible; prefer least-harm options.",
    ],
}

CATEGORY_EMOJI: Dict[str, str] = {
    "Medical": "🩺",
}

CATEGORY_SUBTITLE: Dict[str, str] = {
    "Medical": "Attending physician leading a multidisciplinary discussion to form a safe, guideline-aware diagnostic and treatment plan.",
}
