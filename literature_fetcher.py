"""
literature_fetcher.py
─────────────────────
SilverstoneLab Echem Suite — Literatür otomatik çekme modülü

Scopus (birincil, view=COMPLETE ile abstract dahil)
  → per-paper abstract fallback (arama sonucu boş gelirse)
  → OpenAlex (ücretsiz fallback)

Abstractlardan LOD/LOQ, lineer aralık, duyarlılık, elektrot, metot çıkarır.
"""

from __future__ import annotations
import re
import time
import requests
from typing import Optional

# ── Regex desenleri ────────────────────────────────────────────────────────────
_RE_LOD = re.compile(
    r"(?:LOD|limit\s+of\s+detection|detection\s+limit)[^\d]{0,20}"
    r"([\d]+\.?[\d]*(?:\s*[×xX]\s*10[-−]?\d+)?)"
    r"\s*(n[mM]|µ[mM]|[mM]M|ng[/\s]?mL|µg[/\s]?mL|ppb|ppt|nA|µA|fM|pM)?",
    re.IGNORECASE,
)
_RE_LR = re.compile(
    r"(?:linear\s+(?:dynamic\s+)?range?|working\s+range?|linear\s+response)[^\d]{0,30}"
    r"([\d]+\.?[\d]*(?:[eE][+-]?\d+)?)"
    r"\s*(?:to|–|-|~|and)\s*"
    r"([\d]+\.?[\d]*(?:[eE][+-]?\d+)?)"
    r"\s*(n[mM]|µ[mM]|[mM]M|ng[/\s]?mL|µg[/\s]?mL|ppb|ppt|fM|pM)?",
    re.IGNORECASE,
)
_RE_SENS = re.compile(
    r"(?:sensitivity|slope)[^\d]{0,20}"
    r"([\d]+\.?[\d]*)\s*"
    r"(µA\s*/\s*(?:µ[mM]|[mM]M|n[mM]|mM)|mA\s*/\s*mM|nA\s*/\s*µM"
    r"|µA\s*µM[-−]1|µA\s*mM[-−]1|nA\s*nM[-−]1)?",
    re.IGNORECASE,
)
_RE_METH = re.compile(
    r"\b(DPV|differential\s+pulse\s+voltammetry"
    r"|SWV|square.?wave\s+voltammetry"
    r"|CV|cyclic\s+voltammetry"
    r"|CA|chronoamperometry"
    r"|EIS|amperometry"
    r"|DPSV|AdSV|LSASV|SWASV|DPASV|LSV|NPV)\b",
    re.IGNORECASE,
)
_RE_ELECTRODE = re.compile(
    r"\b(GCE|glassy\s+carbon(?:\s+electrode)?|SPCE|screen.?printed(?:\s+carbon)?(?:\s+electrode)?"
    r"|CPE|carbon\s+paste(?:\s+electrode)?"
    r"|ITO|BDD|BDDE|pencil\s+graphite(?:\s+electrode)?|PGE"
    r"|AuE|gold(?:\s+electrode)?|carbon\s+fiber|graphene"
    r"|MXene|MOF|MWCNT|SWCNT|CNT|rGO|graphene\s+oxide)\b",
    re.IGNORECASE,
)


def _extract_from_abstract(abstract: str, default_unit: str = "µM") -> dict:
    """Abstracttan metrik çıkar. Eksik alanlar boş string döner."""
    result = {"LOD": "", "Linear range": "", "Sensitivity": "", "Method": "", "Electrode": ""}
    if not abstract:
        return result

    m = _RE_LOD.search(abstract)
    if m:
        val  = m.group(1).strip()
        unit = m.group(2).strip() if m.group(2) else default_unit
        result["LOD"] = f"{val} {unit}"

    m = _RE_LR.search(abstract)
    if m:
        unit = m.group(3).strip() if m.group(3) else default_unit
        result["Linear range"] = f"{m.group(1)}–{m.group(2)} {unit}"

    m = _RE_SENS.search(abstract)
    if m:
        unit = f" {m.group(2).strip()}" if m.group(2) else ""
        result["Sensitivity"] = f"{m.group(1)}{unit}"

    m = _RE_METH.search(abstract)
    if m:
        result["Method"] = m.group(1).upper()

    m = _RE_ELECTRODE.search(abstract)
    if m:
        result["Electrode"] = m.group(1)

    return result


# ── Scopus ─────────────────────────────────────────────────────────────────────
_SCOPUS_SEARCH  = "https://api.elsevier.com/content/search/scopus"
_SCOPUS_ABSTRACT= "https://api.elsevier.com/content/abstract/scopus_id/{eid}"


def _build_scopus_query(analyte: str, electrode: str, year_from: int) -> str:
    base = (
        f'TITLE-ABS-KEY("{analyte}" AND '
        f'("electrochemical sensor" OR "modified electrode" OR '
        f'"voltammetric" OR "amperometric" OR "biosensor"))'
        f" AND PUBYEAR > {year_from - 1}"
    )
    if electrode:
        base += f' AND TITLE-ABS-KEY("{electrode}")'
    return base


def _scopus_fetch_abstract(eid: str, api_key: str) -> str:
    """Tek bir makale için abstract çek (search sonucu boş gelirse fallback)."""
    url = _SCOPUS_ABSTRACT.format(eid=eid)
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            core = (data.get("abstracts-retrieval-response", {})
                       .get("coredata", {}))
            return core.get("dc:description", "") or ""
    except Exception:
        pass
    return ""


def _scopus_search(analyte: str, electrode: str, year_from: int,
                   max_results: int, api_key: str) -> list[dict]:
    """Scopus REST API — view=COMPLETE ile abstract dahil."""
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    params = {
        "query": _build_scopus_query(analyte, electrode, year_from),
        "count": min(max_results, 25),
        "view":  "COMPLETE",          # abstract dahil
        "sort":  "coverDate",
    }
    try:
        r = requests.get(_SCOPUS_SEARCH, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        entries = r.json().get("search-results", {}).get("entry", [])
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def _parse_scopus_entries(entries: list[dict], analyte: str,
                          api_key: str = "") -> list[dict]:
    rows = []
    for e in entries:
        title    = e.get("dc:title", "")
        authors  = e.get("dc:creator", "")
        date     = e.get("prism:coverDate", "")
        year     = date[:4] if date else ""
        doi      = e.get("prism:doi", "")
        eid      = e.get("eid", "")

        # Abstract: view=COMPLETE'te dc:description gelir; yoksa ayrı çek
        abstract = (e.get("dc:description", "")
                    or e.get("abstract", "")
                    or "")
        if not abstract and eid and api_key:
            abstract = _scopus_fetch_abstract(eid, api_key)
            time.sleep(0.3)  # rate limit

        ref_str = ""
        if authors:
            ref_str = authors.split(",")[0].strip() + f" et al., {year}"
        else:
            ref_str = year

        metrics = _extract_from_abstract(abstract)
        rows.append({
            "Electrode/Modifier": metrics["Electrode"] or "—",
            "Method":             metrics["Method"]    or "—",
            "LOD":                metrics["LOD"]       or "—",
            "Linear range":       metrics["Linear range"] or "—",
            "Sensitivity":        metrics["Sensitivity"]  or "—",
            "Reference":          ref_str,
            "_title":             title,
            "_doi":               doi,
            "_abstract_len":      len(abstract),
            "_source":            "Scopus",
        })
    return rows


# ── OpenAlex (ücretsiz fallback) ───────────────────────────────────────────────
_OA_SEARCH = "https://api.openalex.org/works"


def _openalex_search(analyte: str, electrode: str, year_from: int,
                     max_results: int) -> list[dict]:
    kw = f"{analyte} electrochemical sensor"
    if electrode:
        kw += f" {electrode}"
    params = {
        "search":  kw,
        "filter":  f"from_publication_date:{year_from}-01-01,type:journal-article",
        "per-page": min(max_results, 25),
        "select":  "title,authorships,publication_year,doi,abstract_inverted_index",
        "sort":    "publication_year:desc",
        "mailto":  "echem@silverstonesci.com",
    }
    try:
        r = requests.get(_OA_SEARCH, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inv_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)


def _parse_oa_entries(entries: list[dict], analyte: str) -> list[dict]:
    rows = []
    for e in entries:
        title  = e.get("title", "")
        year   = str(e.get("publication_year", ""))
        doi    = (e.get("doi") or "").replace("https://doi.org/", "")
        auths  = e.get("authorships", [])
        first  = ""
        if auths:
            a0 = auths[0].get("author", {})
            name = a0.get("display_name", "")
            first = name.split()[-1] if name else ""
        ref_str = f"{first} et al., {year}" if first else year

        abstract = _reconstruct_abstract(e.get("abstract_inverted_index"))
        metrics  = _extract_from_abstract(abstract)
        rows.append({
            "Electrode/Modifier": metrics["Electrode"] or "—",
            "Method":             metrics["Method"]    or "—",
            "LOD":                metrics["LOD"]       or "—",
            "Linear range":       metrics["Linear range"] or "—",
            "Sensitivity":        metrics["Sensitivity"]  or "—",
            "Reference":          ref_str,
            "_title":             title,
            "_doi":               doi,
            "_abstract_len":      len(abstract),
            "_source":            "OpenAlex",
        })
    return rows


# ── Ana fonksiyon ──────────────────────────────────────────────────────────────
def fetch_literature(
    analyte:     str,
    electrode:   str = "",
    year_from:   int = 2018,
    max_results: int = 20,
    scopus_key:  str = "",
) -> list[dict]:
    """
    Scopus (view=COMPLETE) → abstract yoksa per-paper fallback → OpenAlex.
    Dönen liste doğrudan Literature Comparison tablosuna aktarılabilir.
    """
    rows: list[dict] = []

    if scopus_key:
        entries = _scopus_search(analyte, electrode, year_from, max_results, scopus_key)
        if entries:
            rows = _parse_scopus_entries(entries, analyte, api_key=scopus_key)

    if not rows:
        oa_entries = _openalex_search(analyte, electrode, year_from, max_results)
        rows = _parse_oa_entries(oa_entries, analyte)

    # Boş reference olanları filtrele
    rows = [r for r in rows if r.get("Reference") and r.get("Reference") != "—"]
    return rows[:max_results]
