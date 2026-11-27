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
import json
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


_LLM_PROMPT = """\
You are a world-class expert in electrochemical sensors, voltammetry, and analytical chemistry.

Your task: extract EXACTLY these 5 analytical parameters from the abstract below.
Return ONLY a valid JSON object — no explanation, no markdown, no extra text.

Keys and rules:
- "LOD": The limit of detection. Copy the EXACT number and unit as written.
  Examples: "0.05 µM", "12 nM", "3.2 ng/mL", "0.43 pM", "2.1 × 10⁻⁹ M"
  If stated as "X µg/L" keep it as "X µg/L". Do NOT convert. If absent → "".

- "Linear range": The linear dynamic range. Format: "low–high unit".
  Examples: "0.1–100 µM", "5–500 nM", "0.01–10 µg/mL", "1 nM–10 µM"
  Take both endpoints and unit directly from abstract. If absent → "".

- "Sensitivity": Analytical sensitivity or slope of calibration curve.
  Examples: "12.5 µA µM⁻¹ cm⁻²", "0.84 µA/µM", "4.37 mA M⁻¹", "3.21 nA nM⁻¹"
  If absent → "".

- "Method": The primary electrochemical technique used. Use the SHORT acronym.
  DPV=Differential pulse voltammetry, SWV=Square-wave voltammetry, CV=Cyclic voltammetry,
  CA=Chronoamperometry, SWASV=Square-wave anodic stripping, DPASV=Differential pulse anodic stripping,
  EIS=Electrochemical impedance spectroscopy, LSV=Linear sweep voltammetry, NPV=Normal pulse voltammetry.
  If the full name is given (e.g. "differential pulse voltammetry"), return "DPV".
  If absent → "".

- "Electrode": The working electrode, including ALL surface modifiers in full.
  Examples: "MWCNTs/β-CD/GCE", "SPCE", "ZnSNP@Gr/GCE", "MIP/ITO", "BiF/CPGE", "rGO-AuNPs/GCE"
  Include every modifier layer. If the abstract says "glassy carbon electrode modified with X",
  write "X/GCE". If absent → "".

IMPORTANT: If the paper is a REVIEW or OVERVIEW (no original experimental data), return all values as "".

Abstract:
{abstract}

JSON:"""

_LLM_KEYS = {"LOD", "Linear range", "Sensitivity", "Method", "Electrode"}

# ── Review makale filtresi ─────────────────────────────────────────────────────
_REVIEW_WORDS = {
    "review", "overview", "survey", "perspective", "perspectives",
    "mini-review", "mini review", "progress", "advances", "recent advances",
    "emerging", "state of the art",
}

def _is_review_paper(title: str) -> bool:
    """Başlıkta review kelimesi varsa True döner — LLM çağrısı atlanır."""
    t = title.lower()
    return any(kw in t for kw in _REVIEW_WORDS)


# ── Başlıktan çıkarım promptu (abstract yoksa) ────────────────────────────────
_TITLE_PROMPT = """\
You are an expert in electrochemical sensors. You have ONLY the paper title below — no abstract.
Extract what is EXPLICITLY stated in the title. Do NOT infer or guess.

Return ONLY a valid JSON with these exact keys:
- "Electrode": full electrode/modifier stack if mentioned (e.g. "ZnSNP@Gr/GCE", "MIP/rGO/SPE").
  If the title says "ZnSNP@Gr nanocomposite" but no base electrode → write "ZnSNP@Gr".
  If nothing about electrode → "".
- "Method": electrochemical acronym if mentioned (DPV, SWV, CV, CA, EIS...). If not → "".

Title: {title}

JSON:"""


def _call_llm(prompt: str, ai_key: str, ai_provider: str,
              gemini_model: str, max_tokens: int = 300) -> tuple[str, str]:
    """Ham LLM çağrısı — (raw_text, error) döner."""
    raw = ""
    try:
        if ai_provider == "Claude":
            import anthropic
            client = anthropic.Anthropic(api_key=ai_key)
            msg = client.messages.create(
                model="claude-haiku-4-5", max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
        elif ai_provider == "OpenAI":
            import openai
            client = openai.OpenAI(api_key=ai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
        elif ai_provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=ai_key)
            _gm = genai.GenerativeModel(
                gemini_model,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", temperature=0.0,
                ),
            )
            raw = _gm.generate_content(prompt).text.strip()
        else:
            return "", f"bilinmeyen provider: {ai_provider}"
        return raw, ""
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def _parse_llm_json(raw: str, allowed_keys: set) -> tuple[dict, str]:
    """LLM çıktısını JSON'a çevirir — (dict, error) döner."""
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}, f"JSON bulunamadı (raw: {raw[:80]})"
    try:
        data = json.loads(m.group())
        result = {k: str(v).strip() for k, v in data.items()
                  if k in allowed_keys and str(v).strip() not in ("", "null", "None")}
        return result, ""
    except json.JSONDecodeError as e:
        return {}, f"JSON parse: {e} | raw: {raw[:80]}"


def _extract_with_llm(abstract: str, ai_key: str, ai_provider: str = "Claude",
                       gemini_model: str = "gemini-2.0-flash") -> tuple[dict, str]:
    """LLM ile abstract'tan elektrokimyasal parametreleri çıkar.
    Döner: (çıkarılan_dict, hata_mesajı)
    Başarısızsa ({}, hata) döner → regex fallback devreye girer."""
    if not abstract or not ai_key:
        return {}, "abstract veya key yok"
    prompt = _LLM_PROMPT.format(abstract=abstract[:3000])
    raw, err = _call_llm(prompt, ai_key, ai_provider, gemini_model)
    if err:
        return {}, err
    return _parse_llm_json(raw, _LLM_KEYS)


def _extract_from_title_llm(title: str, ai_key: str, ai_provider: str = "Claude",
                             gemini_model: str = "gemini-2.0-flash") -> tuple[dict, str]:
    """Sadece başlıktan Electrode ve Method çıkar (abstract yoksa fallback)."""
    if not title or not ai_key:
        return {}, "title veya key yok"
    prompt = _TITLE_PROMPT.format(title=title)
    raw, err = _call_llm(prompt, ai_key, ai_provider, gemini_model, max_tokens=100)
    if err:
        return {}, err
    return _parse_llm_json(raw, {"Electrode", "Method"})


def _extract_from_abstract(abstract: str, default_unit: str = "µM",
                            ai_key: str = "", ai_provider: str = "Claude",
                            gemini_model: str = "gemini-2.0-flash") -> dict:
    """
    Önce LLM (ai_key varsa), sonra regex ile tamamlar.
    Döner: metrics dict + "_llm_err" key'i (boş string → başarılı)
    """
    result = {"LOD": "", "Linear range": "", "Sensitivity": "",
              "Method": "", "Electrode": "", "_llm_err": ""}
    if not abstract:
        return result

    # ── 1. LLM çıkarım ────────────────────────────────────────────────────────
    if ai_key:
        llm, _llm_err = _extract_with_llm(abstract, ai_key, ai_provider,
                                           gemini_model=gemini_model)
        result["_llm_err"] = _llm_err
        for k in list(result.keys()):
            if llm.get(k):
                result[k] = llm[k]

    # ── 2. Regex fallback — LLM'in boş bıraktıklarını tamamla ────────────────
    if not result["LOD"]:
        m = _RE_LOD.search(abstract)
        if m:
            val  = m.group(1).strip()
            # Birim bulunamazsa "?" koy — varsayılan µM yanlış yönlendiriyordu
            unit = m.group(2).strip() if m.group(2) else "?"
            result["LOD"] = f"{val} {unit}"

    if not result["Linear range"]:
        m = _RE_LR.search(abstract)
        if m:
            unit = m.group(3).strip() if m.group(3) else "?"
            result["Linear range"] = f"{m.group(1)}–{m.group(2)} {unit}"

    if not result["Sensitivity"]:
        m = _RE_SENS.search(abstract)
        if m:
            unit = f" {m.group(2).strip()}" if m.group(2) else ""
            result["Sensitivity"] = f"{m.group(1)}{unit}"

    if not result["Method"]:
        m = _RE_METH.search(abstract)
        if m:
            result["Method"] = m.group(1).upper()

    if not result["Electrode"]:
        m = _RE_ELECTRODE.search(abstract)
        if m:
            result["Electrode"] = m.group(1)

    return result


# ── Scopus ─────────────────────────────────────────────────────────────────────
_SCOPUS_SEARCH   = "https://api.elsevier.com/content/search/scopus"
_SCOPUS_ABSTRACT = "https://api.elsevier.com/content/abstract/scopus_id/{eid}"


def _build_scopus_query(analyte: str, electrode: str, year_from: int) -> str:
    # Analiti ayrı tut, elektrokimyasal terimler phrase olarak eşleşsin
    # Review/overview/survey başlıklarını hariç tut
    base = (
        f'TITLE-ABS-KEY("{analyte}") AND '
        f'TITLE-ABS-KEY("electrochemical sensor" OR "modified electrode" OR '
        f'"voltammetric sensor" OR "amperometric sensor" OR '
        f'"electrochemical detection" OR "electrochemical determination") AND '
        f'NOT TITLE("review" OR "overview" OR "survey" OR "perspective" OR '
        f'"mini-review" OR "progress report" OR "emerging") AND '
        f'PUBYEAR > {year_from - 1} AND DOCTYPE(ar)'
    )
    # DOCTYPE(ar) → sadece araştırma makaleleri, review/book chapter değil
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
                           fetch_missing: bool = True,
                           ai_key: str = "", ai_provider: str = "Claude",
                           gemini_model: str = "gemini-2.0-flash") -> tuple[list, int]:
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

        _has_abs = bool(abstract)
        if _has_abs:
            abstract_found += 1

        ref_str = (authors.split(",")[0].strip() + f" et al., {year}"
                   if authors else year)

        # Review makale → LLM/regex'i atla, hepsi "—"
        if _is_review_paper(title):
            metrics = {"LOD": "", "Linear range": "", "Sensitivity": "",
                       "Method": "", "Electrode": "", "_llm_err": "review"}
        elif _has_abs:
            metrics = _extract_from_abstract(abstract, ai_key=ai_key,
                                             ai_provider=ai_provider,
                                             gemini_model=gemini_model)
        else:
            # Abstract yok → başlıktan Electrode/Method çıkar
            metrics = {"LOD": "", "Linear range": "", "Sensitivity": "",
                       "Method": "", "Electrode": "", "_llm_err": ""}
            if ai_key and title:
                _tm, _te = _extract_from_title_llm(title, ai_key, ai_provider, gemini_model)
                metrics["Electrode"] = _tm.get("Electrode", "")
                metrics["Method"]    = _tm.get("Method", "")
                metrics["_llm_err"]  = _te
            # Regex ile de dene (title kısa olsa da çalışır)
            if not metrics["Electrode"]:
                _rm = _RE_ELECTRODE.search(title)
                if _rm:
                    metrics["Electrode"] = _rm.group(1)
            if not metrics["Method"]:
                _rm = _RE_METH.search(title)
                if _rm:
                    metrics["Method"] = _rm.group(1).upper()

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
            "_has_abstract":      _has_abs,
            "_llm_err":           metrics.get("_llm_err", ""),
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


def _parse_oa_entries(entries: list, ai_key: str = "",
                      ai_provider: str = "Claude",
                      gemini_model: str = "gemini-2.0-flash") -> tuple[list, int]:
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
        _has_abs = bool(abstract)
        if _has_abs:
            abstract_found += 1

        if _is_review_paper(title):
            metrics = {"LOD": "", "Linear range": "", "Sensitivity": "",
                       "Method": "", "Electrode": "", "_llm_err": "review"}
        elif _has_abs:
            metrics = _extract_from_abstract(abstract, ai_key=ai_key,
                                             ai_provider=ai_provider,
                                             gemini_model=gemini_model)
        else:
            metrics = {"LOD": "", "Linear range": "", "Sensitivity": "",
                       "Method": "", "Electrode": "", "_llm_err": ""}
            if ai_key and title:
                _tm, _te = _extract_from_title_llm(title, ai_key, ai_provider, gemini_model)
                metrics["Electrode"] = _tm.get("Electrode", "")
                metrics["Method"]    = _tm.get("Method", "")
                metrics["_llm_err"]  = _te
            if not metrics["Electrode"]:
                _rm = _RE_ELECTRODE.search(title)
                if _rm:
                    metrics["Electrode"] = _rm.group(1)
            if not metrics["Method"]:
                _rm = _RE_METH.search(title)
                if _rm:
                    metrics["Method"] = _rm.group(1).upper()

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
            "_has_abstract":      _has_abs,
            "_llm_err":           metrics.get("_llm_err", ""),
        })
    return rows, abstract_found


# ── Ana fonksiyon ──────────────────────────────────────────────────────────────
def fetch_literature(
    analyte:      str,
    electrode:    str = "",
    year_from:    int = 2018,
    max_results:  int = 20,
    scopus_key:   str = "",
    prefer_openalex: bool = False,
    ai_key:       str = "",
    ai_provider:  str = "Claude",
    gemini_model: str = "gemini-2.0-flash",
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

    _use_ai = bool(ai_key)
    if _use_ai:
        status["warnings"].append(
            f"🤖 AI analiz aktif ({ai_provider}) — abstract'lar LLM ile yorumlanıyor."
        )

    # OpenAlex tercih edilmişse direkt oraya git
    if prefer_openalex:
        oa_entries, oa_err = _openalex_search(analyte, electrode, year_from, max_results)
        if oa_err:
            status["error"] = oa_err
            return [], status
        rows, abs_found = _parse_oa_entries(oa_entries, ai_key=ai_key,
                                            ai_provider=ai_provider, gemini_model=gemini_model)
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
            rows, abs_found = _parse_scopus_entries(
                entries, scopus_key, fetch_missing=True,
                ai_key=ai_key, ai_provider=ai_provider,
                gemini_model=gemini_model)
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

    rows, abs_found = _parse_oa_entries(oa_entries, ai_key=ai_key,
                                        ai_provider=ai_provider, gemini_model=gemini_model)
    rows = [r for r in rows if r.get("Reference") and r.get("Reference") != "—"]
    status.update({"source": "OpenAlex", "count": len(rows),
                   "abstract_found": abs_found})
    return rows[:max_results], status
