"""
literature_fetcher.py
─────────────────────
SilverstoneLab Echem Suite — Literatür otomatik çekme modülü

Scopus (view=COMPLETE → STANDARD fallback) + OpenAlex (ücretsiz)
fetch_literature() → (rows: list[dict], status: dict)
  status = {"source": str, "count": int, "error": str, "abstract_found": int}
"""

from __future__ import annotations
import re
import time
import requests

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
    r"|CA|chronoamperometry|EIS|amperometry"
    r"|DPSV|AdSV|LSASV|SWASV|DPASV|LSV|NPV)\b",
    re.IGNORECASE,
)
_RE_ELECTRODE = re.compile(
    r"\b(GCE|glassy\s+carbon(?:\s+electrode)?"
    r"|SPCE|screen.?printed(?:\s+carbon)?(?:\s+electrode)?"
    r"|CPE|carbon\s+paste(?:\s+electrode)?"
    r"|ITO|BDD|BDDE|pencil\s+graphite(?:\s+electrode)?|PGE"
    r"|AuE|gold(?:\s+electrode)?|carbon\s+fiber|graphene"
    r"|MXene|MOF|MWCNT|SWCNT|CNT|rGO|graphene\s+oxide)\b",
    re.IGNORECASE,
)


def _extract_from_abstract(abstract: str, default_unit: str = "µM") -> dict:
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
_SCOPUS_SEARCH   = "https://api.elsevier.com/content/search/scopus"
_SCOPUS_ABSTRACT = "https://api.elsevier.com/content/abstract/scopus_id/{eid}"


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
    """Tek makale için abstract çek — Scopus Abstract Retrieval API."""
    url = _SCOPUS_ABSTRACT.format(eid=eid.replace("2-s2.0-", ""))
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            core = (r.json()
                    .get("abstracts-retrieval-response", {})
                    .get("coredata", {}))
            return core.get("dc:description", "") or ""
    except Exception:
        pass
    return ""


def _openalex_abstract_by_doi(doi: str) -> str:
    """DOI kullanarak OpenAlex'ten abstract çek (ücretsiz, kurumsal token gerektirmez)."""
    if not doi:
        return ""
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    try:
        r = requests.get(url, params={"select": "abstract_inverted_index"}, timeout=10)
        if r.status_code == 200:
            inv = r.json().get("abstract_inverted_index")
            return _reconstruct_abstract(inv)
    except Exception:
        pass
    return ""


def _scopus_search_with_view(analyte: str, electrode: str, year_from: int,
                              max_results: int, api_key: str,
                              view: str = "COMPLETE") -> tuple[list, str]:
    """
    Scopus araması. view=COMPLETE abstract içerir ama bazı key'lerde yasak.
    Hata → ([], hata mesajı) döner.
    """
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    params = {
        "query": _build_scopus_query(analyte, electrode, year_from),
        "count": min(max_results, 25),
        "view":  view,
        "sort":  "coverDate",
    }
    try:
        r = requests.get(_SCOPUS_SEARCH, headers=headers, params=params, timeout=20)
        if r.status_code == 200:
            entries = r.json().get("search-results", {}).get("entry", [])
            if isinstance(entries, list) and entries:
                # Bazen tek sonuç dict olarak gelir
                if isinstance(entries[0], dict) and "error" in entries[0]:
                    return [], f"Scopus query error: {entries[0].get('error')}"
            return (entries if isinstance(entries, list) else []), ""
        else:
            # HTTP hata — mesajı çıkar
            try:
                err_body = r.json()
                err_msg = (err_body.get("service-error", {})
                                   .get("status", {})
                                   .get("statusText", r.text[:200]))
            except Exception:
                err_msg = r.text[:200]
            return [], f"Scopus HTTP {r.status_code}: {err_msg}"
    except requests.exceptions.Timeout:
        return [], "Scopus timeout — VPN/kampüs ağı aktif mi?"
    except requests.exceptions.ConnectionError as e:
        return [], f"Scopus bağlantı hatası: {e}"
    except Exception as e:
        return [], f"Scopus beklenmeyen hata: {e}"


def _parse_scopus_entries(entries: list, api_key: str = "",
                           fetch_missing: bool = True) -> tuple[list, int]:
    """Girişleri satıra çevirir. (rows, abstract_found_count) döner."""
    rows = []
    abstract_found = 0
    for e in entries:
        title   = e.get("dc:title", "")
        authors = e.get("dc:creator", "")
        date    = e.get("prism:coverDate", "")
        year    = date[:4] if date else ""
        doi     = e.get("prism:doi", "")
        eid     = e.get("eid", "")

        abstract = e.get("dc:description", "") or e.get("abstract", "") or ""

        # Scopus'tan abstract gelmemişse: önce Scopus Abstract Retrieval API, sonra OpenAlex by DOI
        if not abstract and fetch_missing:
            if eid and api_key:
                abstract = _scopus_fetch_abstract(eid, api_key)
                time.sleep(0.3)
            # Hâlâ boşsa → DOI varsa OpenAlex'ten ücretsiz çek
            if not abstract and doi:
                abstract = _openalex_abstract_by_doi(doi)
                time.sleep(0.2)

        if abstract:
            abstract_found += 1

        ref_str = (authors.split(",")[0].strip() + f" et al., {year}"
                   if authors else year)
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
            "_source":            "Scopus",
        })
    return rows, abstract_found


# ── OpenAlex ───────────────────────────────────────────────────────────────────
_OA_SEARCH = "https://api.openalex.org/works"


def _openalex_search(analyte: str, electrode: str, year_from: int,
                     max_results: int) -> tuple[list, str]:
    kw = f"{analyte} electrochemical sensor"
    if electrode:
        kw += f" {electrode}"
    params = {
        "search":   kw,
        "filter":   f"from_publication_date:{year_from}-01-01,type:journal-article",
        "per-page": min(max_results, 25),
        "select":   "title,authorships,publication_year,doi,abstract_inverted_index",
        "sort":     "publication_year:desc",
        "mailto":   "echem@silverstonesci.com",
    }
    try:
        r = requests.get(_OA_SEARCH, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("results", []), ""
    except requests.exceptions.Timeout:
        return [], "OpenAlex timeout"
    except Exception as e:
        return [], f"OpenAlex hata: {e}"


def _reconstruct_abstract(inv_index: dict | None) -> str:
    if not inv_index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in inv_index.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)


def _parse_oa_entries(entries: list) -> tuple[list, int]:
    rows = []
    abstract_found = 0
    for e in entries:
        title  = e.get("title", "")
        year   = str(e.get("publication_year", ""))
        doi    = (e.get("doi") or "").replace("https://doi.org/", "")
        auths  = e.get("authorships", [])
        first  = ""
        if auths:
            name  = auths[0].get("author", {}).get("display_name", "")
            first = name.split()[-1] if name else ""
        ref_str  = f"{first} et al., {year}" if first else year
        abstract = _reconstruct_abstract(e.get("abstract_inverted_index"))
        if abstract:
            abstract_found += 1
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
            "_source":            "OpenAlex",
        })
    return rows, abstract_found


# ── Ana fonksiyon ──────────────────────────────────────────────────────────────
def fetch_literature(
    analyte:     str,
    electrode:   str = "",
    year_from:   int = 2018,
    max_results: int = 20,
    scopus_key:  str = "",
    prefer_openalex: bool = False,
) -> tuple[list[dict], dict]:
    """
    Döner: (rows, status)
    status = {
      "source":         "Scopus" | "OpenAlex" | "none",
      "count":          int,
      "abstract_found": int,
      "error":          str,   # boş → başarılı
      "warnings":       list[str],
    }
    """
    status: dict = {"source": "none", "count": 0,
                    "abstract_found": 0, "error": "", "warnings": []}

    # OpenAlex tercih edilmişse direkt oraya git
    if prefer_openalex:
        oa_entries, oa_err = _openalex_search(analyte, electrode, year_from, max_results)
        if oa_err:
            status["error"] = oa_err
            return [], status
        rows, abs_found = _parse_oa_entries(oa_entries)
        status.update({"source": "OpenAlex", "count": len(rows),
                       "abstract_found": abs_found})
        return rows[:max_results], status

    # Scopus dene
    if scopus_key:
        # Önce view=COMPLETE
        entries, err = _scopus_search_with_view(
            analyte, electrode, year_from, max_results, scopus_key, view="COMPLETE")

        if err and entries == []:
            # COMPLETE başarısız → STANDARD ile tekrar dene
            status["warnings"].append(f"COMPLETE view başarısız ({err}), STANDARD deneniyor…")
            entries, err = _scopus_search_with_view(
                analyte, electrode, year_from, max_results, scopus_key, view="STANDARD")

        if err and entries == []:
            status["error"] = err
            status["warnings"].append("Scopus başarısız — OpenAlex'e düşülüyor.")
        elif entries:
            rows, abs_found = _parse_scopus_entries(entries, scopus_key, fetch_missing=True)
            rows = [r for r in rows if r.get("Reference") and r.get("Reference") != "—"]
            status.update({"source": "Scopus", "count": len(rows),
                           "abstract_found": abs_found})
            return rows[:max_results], status

    # OpenAlex fallback
    oa_entries, oa_err = _openalex_search(analyte, electrode, year_from, max_results)
    if oa_err:
        if status["error"]:
            status["error"] += f" | OpenAlex da başarısız: {oa_err}"
        else:
            status["error"] = oa_err
        return [], status

    rows, abs_found = _parse_oa_entries(oa_entries)
    rows = [r for r in rows if r.get("Reference") and r.get("Reference") != "—"]
    status.update({"source": "OpenAlex", "count": len(rows),
                   "abstract_found": abs_found})
    return rows[:max_results], status
