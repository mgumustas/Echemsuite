"""
literature_fetcher.py
─────────────────────
SilverstoneLab Echem Suite — Literatür otomatik çekme modülü

Scopus (birincil) → OpenAlex (ücretsiz fallback)
Abstractlardan LOD/LOQ, lineer aralık, duyarlılık, elektrot, metot çıkarır.

Kullanım:
    from literature_fetcher import fetch_literature
    rows = fetch_literature(
        analyte="dopamine",
        electrode="GCE",
        year_from=2018,
        max_results=20,
        scopus_key="YOUR_KEY"   # boş → sadece OpenAlex
    )
"""

from __future__ import annotations
import re
import time
import requests
from typing import Optional

# ── Regex desenleri ────────────────────────────────────────────────────────────
_RE_LOD = re.compile(
    r"(?:LOD|limit\s+of\s+detection)[^\d]*"
    r"([\d]+\.?[\d]*(?:\s*[×x]\s*10[⁻\-]?\d+)?)"
    r"\s*(n[mM]|µ[mM]|[mM]M|ng[/\s]mL|µg[/\s]mL|ppb|ppt)?",
    re.IGNORECASE,
)
_RE_LR = re.compile(
    r"(?:linear\s+range?|working\s+range?)[^\d]*"
    r"([\d]+\.?[\d]*)\s*(?:to|–|-|~)\s*([\d]+\.?[\d]*)"
    r"\s*(n[mM]|µ[mM]|[mM]M|ng[/\s]mL)?",
    re.IGNORECASE,
)
_RE_SENS = re.compile(
    r"(?:sensitivity|slope)[^\d]*"
    r"([\d]+\.?[\d]*)\s*"
    r"(?:µA\s*/\s*(?:µ[mM]|[mM]M|n[mM]|mM)|mA\s*/\s*mM|nA\s*/\s*µM)?",
    re.IGNORECASE,
)
_RE_METH = re.compile(
    r"\b(DPV|differential\s+pulse\s+voltammetry|SWV|square\s+wave\s+voltammetry"
    r"|CV|cyclic\s+voltammetry|CA|chronoamperometry|EIS|amperometry|DPSV"
    r"|AdSV|LSASV|SWASV|DPASV)\b",
    re.IGNORECASE,
)
_RE_ELECTRODE = re.compile(
    r"\b(GCE|glassy\s+carbon|SPCE|screen.?printed|CPE|carbon\s+paste"
    r"|ITO|BDD|BDDE|pencil\s+graphite|PGE|AuE|gold\s+electrode"
    r"|carbon\s+fiber|graphene|MXene|MOF)\b",
    re.IGNORECASE,
)


def _extract_from_abstract(abstract: str, default_unit: str = "µM") -> dict:
    """Abstracttan metrik çıkar. Eksik alanlar boş string döner."""
    result = {
        "LOD":           "",
        "Linear range":  "",
        "Sensitivity":   "",
        "Method":        "",
        "Electrode":     "",
    }
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
        result["Sensitivity"] = m.group(1)

    m = _RE_METH.search(abstract)
    if m:
        result["Method"] = m.group(1).upper()

    m = _RE_ELECTRODE.search(abstract)
    if m:
        result["Electrode"] = m.group(1)

    return result


# ── Scopus ─────────────────────────────────────────────────────────────────────
_SCOPUS_SEARCH = "https://api.elsevier.com/content/search/scopus"
_SCOPUS_ABS    = "https://api.elsevier.com/content/abstract/scopus_id/{eid}"


def _build_scopus_query(analyte: str, electrode: str, year_from: int) -> str:
    base = (
        f'TITLE-ABS-KEY("{analyte}" AND '
        f'("electrochemical sensor" OR "modified electrode" OR '
        f'"voltammetric" OR "amperometric"))'
        f" AND PUBYEAR > {year_from - 1}"
    )
    if electrode:
        base += f' AND TITLE-ABS-KEY("{electrode}")'
    return base


def _scopus_search(analyte: str, electrode: str, year_from: int,
                   max_results: int, api_key: str) -> list[dict]:
    """Scopus REST API ile arama — institutional IP veya key gerektirir."""
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept":        "application/json",
    }
    params = {
        "query":  _build_scopus_query(analyte, electrode, year_from),
        "count":  min(max_results, 25),
        "field":  "dc:title,dc:creator,prism:publicationName,prism:coverDate,"
                  "prism:doi,eid,dc:description",
        "sort":   "coverDate",
    }
    try:
        r = requests.get(_SCOPUS_SEARCH, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        entries = (r.json()
                   .get("search-results", {})
                   .get("entry", []))
        return entries
    except Exception:
        return []


def _parse_scopus_entries(entries: list[dict], analyte: str) -> list[dict]:
    rows = []
    for e in entries:
        title    = e.get("dc:title", "")
        authors  = e.get("dc:creator", "")
        journal  = e.get("prism:publicationName", "")
        date     = e.get("prism:coverDate", "")
        year     = date[:4] if date else ""
        doi      = e.get("prism:doi", "")
        abstract = e.get("dc:description", "")
        ref_str  = f"{authors.split(',')[0]} et al., {year}" if authors else year

        metrics = _extract_from_abstract(abstract)
        rows.append({
            "Electrode/Modifier": metrics["Electrode"] or "—",
            "Method":             metrics["Method"]    or "—",
            f"LOD":               metrics["LOD"]       or "—",
            "Linear range":       metrics["Linear range"] or "—",
            "Sensitivity":        metrics["Sensitivity"]  or "—",
            "Reference":          ref_str,
            "_title":             title,
            "_doi":               doi,
            "_source":            "Scopus",
        })
    return rows


# ── OpenAlex (ücretsiz fallback) ───────────────────────────────────────────────
_OA_SEARCH = "https://api.openalex.org/works"


def _build_openalex_query(analyte: str, electrode: str, year_from: int) -> str:
    kw = f"{analyte} electrochemical sensor"
    if electrode:
        kw += f" {electrode}"
    return kw


def _openalex_search(analyte: str, electrode: str, year_from: int,
                     max_results: int) -> list[dict]:
    params = {
        "search":           _build_openalex_query(analyte, electrode, year_from),
        "filter":           f"from_publication_date:{year_from}-01-01,"
                            "type:journal-article",
        "per-page":         min(max_results, 25),
        "select":           "title,authorships,primary_location,publication_year,"
                            "doi,abstract_inverted_index",
        "sort":             "publication_year:desc",
        "mailto":           "echem@silverstonesci.com",
    }
    try:
        r = requests.get(_OA_SEARCH, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


def _reconstruct_abstract(inv_index: dict | None) -> str:
    """OpenAlex inverted index → düz metin."""
    if not inv_index:
        return ""
    # {word: [pos, ...]} → sıralı
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
        doi    = e.get("doi", "")
        auths  = e.get("authorships", [])
        first  = ""
        if auths:
            a0 = auths[0].get("author", {})
            first = a0.get("display_name", "")
            first = first.split()[-1] if first else ""
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
            "_doi":               doi.replace("https://doi.org/", "") if doi else "",
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
    Önce Scopus'u dene (key varsa), başarısız veya key yoksa OpenAlex'e düş.
    Dönen liste doğrudan Literature Comparison tablosuna aktarılabilir.
    """
    rows: list[dict] = []

    if scopus_key:
        entries = _scopus_search(analyte, electrode, year_from, max_results, scopus_key)
        if entries:
            rows = _parse_scopus_entries(entries, analyte)

    if not rows:
        # OpenAlex fallback
        oa_entries = _openalex_search(analyte, electrode, year_from, max_results)
        rows = _parse_oa_entries(oa_entries, analyte)

    # Boş satırları filtrele (sadece "—" olanları çıkar)
    rows = [r for r in rows if r.get("Reference") and r.get("Reference") != "—"]

    return rows[:max_results]
