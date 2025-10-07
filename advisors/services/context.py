"""Helpers for fetching lightweight external context (web highlights, PubMed)."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

try:
    from duckduckgo_search import DDGS  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DDGS = None


def build_web_context(category: str, agenda_text: str) -> str:
    """Fetch brief web highlights (DuckDuckGo text search).

    The DuckDuckGo client is free but optional; if unavailable we fall back to
    returning an empty string so the caller can gracefully skip the context.
    """

    if DDGS is None:
        return ""
    try:
        query = f"{category} background for: {agenda_text[:500]}"
        bullets: List[str] = []
        with DDGS() as ddgs:  # free, no API key
            for result in ddgs.text(query, max_results=5):
                title = result.get("title") or result.get("href") or ""
                snippet = (result.get("body") or "").strip()[:300]
                url = result.get("href") or ""
                if not (title or snippet):
                    continue
                bullets.append(f"- {title}: {snippet} ({url})")
        return ("Web search highlights:\n" + "\n".join(bullets)) if bullets else ""
    except Exception:
        return ""


def build_pubmed_context(agenda_text: str, max_results: int = 5) -> Tuple[str, str, Dict, Dict]:
    """Fetch PubMed highlights for the agenda.

    Returns a tuple of (query, markdown summary, esearch JSON, esummary JSON).
    Network failures fall back to empty payloads so callers can continue.
    """

    try:
        import json as _json
        from urllib.parse import urlencode, quote_plus as _qp
        from urllib.request import urlopen as _urlopen

        user_q = (agenda_text or "").strip()
        if not user_q:
            return ("", "", {}, {})

        def _esearch(_term: str, _retmax: int) -> Dict:
            params = {
                "db": "pubmed",
                "retmode": "json",
                "retmax": str(_retmax),
                "term": _term,
                "sort": "relevance",
            }
            api_key = os.environ.get("NCBI_API_KEY")
            if api_key:
                params["api_key"] = api_key
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            url = f"{base}/esearch.fcgi?{urlencode(params, quote_via=_qp)}"
            with _urlopen(url, timeout=20) as response:
                return _json.loads(response.read().decode("utf-8"))

        def _esummary(_ids: List[str]) -> Dict:
            esum_params = {
                "db": "pubmed",
                "retmode": "json",
                "id": ",".join(_ids),
            }
            api_key = os.environ.get("NCBI_API_KEY")
            if api_key:
                esum_params["api_key"] = api_key
            base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
            url = f"{base}/esummary.fcgi?{urlencode(esum_params, quote_via=_qp)}"
            with _urlopen(url, timeout=20) as response:
                return _json.loads(response.read().decode("utf-8"))

        # Pass 1: English + recent/systematic filter for precision
        term = f"{user_q} AND (english[la]) AND ((last 5 years[dp]) OR (systematic[sb]))"
        es = _esearch(term, max_results)
        idlist = (es.get("esearchresult", {}).get("idlist") or [])[:max_results]

        # Pass 2: relax filter if nothing returned
        if not idlist:
            term_alt = f"{user_q} AND english[la]"
            es = _esearch(term_alt, max_results * 2)
            idlist = (es.get("esearchresult", {}).get("idlist") or [])[:max_results]

        # Pass 3: bias toward guidelines if still empty
        if not idlist:
            term_guidelines = f"({user_q}) AND clinical guidelines"
            es = _esearch(term_guidelines, max_results * 2)
            idlist = (es.get("esearchresult", {}).get("idlist") or [])[:max_results]

        if not idlist:
            return (term, "", es, {})

        esummary = _esummary(idlist)
        docs = esummary.get("result", {})
        md_lines = ["PubMed highlights:"]
        for pmid in idlist:
            doc = docs.get(pmid)
            if not doc:
                continue
            title = doc.get("title", "(title unavailable)")
            journal = doc.get("fulljournalname") or doc.get("source") or ""
            authors = doc.get("authors") or []
            first_author = authors[0]["name"] if authors else ""
            year = (doc.get("pubdate") or "").split(" ")[0]
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            md_lines.append(f"- {title} - {first_author} et al., {journal} ({year}). {url}")
        return (term, "\n".join(md_lines), es, esummary)
    except Exception:
        return ("", "", {}, {})
