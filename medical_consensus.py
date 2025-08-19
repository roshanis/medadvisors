import os
from pathlib import Path
from virtual_lab.agent import Agent
from virtual_lab.run_meeting import run_meeting
from dotenv import load_dotenv

# Load environment variables from .env next to this script
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Configuration: ensure OPENAI_API_KEY is set in environment
if not os.environ.get('OPENAI_API_KEY'):
    raise SystemExit('Please set OPENAI_API_KEY in your environment before running.')


SAVE_DIR = Path("./medical_meetings")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    consensus = run_meeting(
        meeting_type="team",
        agenda=AGENDA,
        save_dir=SAVE_DIR,
        save_name="chest_pain_case",
        team_lead=TEAM_LEAD,
        team_members=TEAM_MEMBERS,
        agenda_questions=AGENDA_QUESTIONS,
        agenda_rules=AGENDA_RULES,
        num_rounds=2,
        temperature=0.2,
        pubmed_search=False,
        return_summary=True,
    )
    print("\n===== CONSENSUS SUMMARY =====\n")
    print(consensus)
