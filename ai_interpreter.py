"""
SilverstoneLab — AI Interpreter
Calls the Anthropic Claude API to generate electrochemistry-expert
commentary on analysis results from each module.
"""

from __future__ import annotations
import json


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (shared across all modules)
# ─────────────────────────────────────────────────────────────
_SYSTEM = """You are a senior electrochemist with expertise in
modified electrode design, electrocatalysis, and electroanalytical
chemistry. You interpret experimental results concisely, in the
style of a peer-reviewed analytical chemistry paper (Analytical
Chemistry, Biosensors & Bioelectronics, Electrochimica Acta).

Rules:
- Write in fluent English, 2–5 short paragraphs.
- Quote numeric values exactly as given; use correct SI units.
- Comment on data quality, compare to literature benchmarks
  where relevant, and flag any anomalies worth investigating.
- End with one sentence summarising the key take-away.
- Do NOT use bullet points or headers inside your commentary.
- Do NOT start with "Certainly" or any affirmation phrase."""


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────
def _call(api_key: str, user_msg: str, model: str = "claude-opus-5-5") -> str:
    """Call Claude via the Anthropic SDK; returns plain text."""
    try:
        import anthropic
    except ImportError:
        return "⚠️ 'anthropic' package not installed. Add it to requirements.txt."

    _FALLBACK_MODELS = [
        model,
        "claude-opus-5-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-3-5-sonnet-20241022",
    ]

    # Let the SDK manage its own http client (avoids httpx vs httpx2 version conflicts)
    client = anthropic.Anthropic(api_key=api_key)

    last_err = None
    for _m in _FALLBACK_MODELS:
        try:
            msg = client.messages.create(
                model      = _m,
                max_tokens = 4096,   # large enough for thinking + text
                system     = _SYSTEM,
                messages   = [{"role": "user", "content": user_msg}],
            )

            # Iterate blocks — TextBlock has .text str, ThinkingBlock does not
            for block in msg.content:
                try:
                    t = block.text
                    if isinstance(t, str) and t.strip():
                        return t.strip()
                except AttributeError:
                    continue
            return "⚠️ No text content in response."
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "not_found" in err_str or "404" in err_str:
                continue   # try next model
            # For connection errors, return descriptive message instead of raising
            if any(x in err_str for x in ["connection", "timeout", "network", "ssl", "connect"]):
                return (f"⚠️ Connection error reaching Anthropic API: {e}\n\n"
                        "Check: (1) API key is valid, (2) Streamlit Cloud network allows outbound HTTPS.")
            raise
    return f"⚠️ No available model found. Last error: {last_err}"


# ─────────────────────────────────────────────────────────────
# MODULE-SPECIFIC INTERPRETERS
# ─────────────────────────────────────────────────────────────

def interpret_cv(api_key: str, results: dict, electrode_name: str = "electrode", model: str = "claude-opus-5-5") -> str:
    ecsa   = results.get("ecsa", {})
    mech   = results.get("mechanism", {})
    k0     = results.get("k0", {})

    ecsa_v  = ecsa.get("ECSA_cm2",  "N/A")
    r2_ecsa = ecsa.get("R2",        "N/A")
    b       = mech.get("slope_b",   "N/A")
    mech_lbl= mech.get("mechanism", "N/A")
    k0_mean = k0.get("mean",        None)
    k0_str  = f"{k0_mean:.3e} cm s⁻¹" if k0_mean else "not calculated"

    prompt = f"""Interpret the following cyclic voltammetry scan-rate study results for
the electrode: {electrode_name}.

ECSA (Randles-Ševčík): {ecsa_v} cm² (R² = {r2_ecsa})
Current–scan-rate mechanism (log ip vs log ν slope b): {b}  → classified as: {mech_lbl}
Apparent heterogeneous rate constant k° (Nicholson method): {k0_str}

Discuss:
1. The quality of the ECSA estimate and what it implies about active surface area.
2. What the b-value reveals about the rate-limiting step (diffusion vs. surface-confined process).
3. The significance of the k° value and what it says about electron-transfer kinetics.
4. Any recommendations for optimising the electrode or the measurement protocol."""

    return _call(api_key, prompt, model)


def interpret_eis(api_key: str, param_df, electrode_labels: list[str] | None = None, model: str = "claude-opus-5-5") -> str:
    if param_df is None or (hasattr(param_df, 'empty') and param_df.empty):
        return "No EIS fit parameters available for interpretation."

    try:
        tbl = param_df.to_string(index=False)
    except Exception:
        tbl = str(param_df)

    labels_str = ", ".join(electrode_labels) if electrode_labels else "not specified"

    prompt = f"""You are interpreting electrochemical impedance spectroscopy (EIS) data
for a manuscript targeting Analytical Chemistry, Biosensors & Bioelectronics,
or Electrochimica Acta.

Electrode series (in order): {labels_str}
Randles-CPE circuit fitting results:

{tbl}

Write 3–4 paragraphs of expert commentary covering:

1. Rs (solution resistance, Ω): confirm stability across electrodes —
   any significant change indicates measurement artefacts or ionic-strength effects.

2. Rct (charge-transfer resistance, Ω): describe the TREND across the series
   with exact values. Explain mechanistically whether each modification step
   increased or decreased Rct and why (surface blocking, electrocatalysis,
   faster electron-transfer kinetics, analyte adsorption, etc.).
   Compare Rct values to typical literature ranges for similar systems.

3. CPE parameters (Q and n): interpret n — if n < 0.85 this suggests significant
   surface roughness or inhomogeneity; if n ≈ 1 the interface is near-ideal
   capacitive. Discuss C_eff (effective capacitance) and what changes in
   double-layer capacitance reveal about surface area modification.

4. Fit quality (χ² or residuals if present): state whether the Randles-CPE
   model is adequate or whether a more complex circuit (Warburg, nested RC)
   should be considered.

Quote every numeric value. Use correct SI notation. Do NOT use bullet points."""

    return _call(api_key, prompt, model)


def interpret_ph(api_key: str, results: dict, analyte: str = "analyte", model: str = "claude-opus-5-5") -> str:
    slope    = results.get("slope_mV_per_pH", "N/A")
    r2       = results.get("R2",              "N/A")
    mn       = results.get("ratio_m_n",       "N/A")
    opt_pH   = results.get("optimal_pH",      "N/A")

    prompt = f"""Interpret the following pH-dependent electrochemical study for {analyte}.

Peak potential (Ep) vs pH slope: {slope} mV pH⁻¹  (R² = {r2})
Nernst-predicted slope for m/n = {mn}: {-59.2 * float(mn) if isinstance(mn, (int,float)) else 'see ratio'}  mV pH⁻¹
Optimal working pH identified: {opt_pH}

Discuss:
1. Whether the observed slope is consistent with Nernstian behaviour
   and what the m/n ratio implies about the proton–electron stoichiometry.
2. The mechanistic significance of the pH dependence
   (e.g. PCET, pH-sensitive equilibrium prior to electron transfer).
3. The choice of optimal pH and its relevance for practical sensing
   (physiological fluids, real-sample compatibility).
4. Any buffer or ionic-strength considerations the analyst should bear in mind."""

    return _call(api_key, prompt, model)


def interpret_calibration(api_key: str, results: dict,
                           analyte: str = "analyte", unit: str = "µM",
                           model: str = "claude-opus-5-5",
                           precision_results: dict = None,
                           accuracy_rows: list = None,
                           sa_results: dict = None,
                           journal: str = "",
                           language: str = "English") -> str:
    """
    Calibration-only interpretation OR full ICH Q2(R2) validation report
    when precision_results / accuracy_rows are provided.
    """
    cr        = results.get("calibration", results)
    lod       = cr.get("LOD",        "N/A")
    loq       = cr.get("LOQ",        "N/A")
    slope     = cr.get("slope",      "N/A")
    intercept = cr.get("intercept",  "N/A")
    r         = cr.get("r", cr.get("R2", "N/A"))
    lr        = cr.get("linear_range", ["N/A", "N/A"])

    # ── Full validation report (bulk mode) ────────────────────────────────
    if precision_results or accuracy_rows:
        journal_note = (f" The manuscript targets {journal}." if journal else "")
        lang_note    = (" Write the entire response in Turkish." if language == "Türkçe" else "")

        # Build precision block
        prec_block = ""
        if precision_results:
            intra_r = precision_results.get("intraday_results", {})
            inter_r = precision_results.get("interday_results", {})
            if intra_r:
                prec_block += "\nRepeatability (intraday):\n"
                for lbl, v in intra_r.items():
                    prec_block += (f"  {lbl}: mean={v['mean']:.4f} µA, "
                                   f"%RSD={v['rsd']:.2f}%, n={v['n']}, "
                                   f"{'PASS' if v['pass'] else 'FAIL'} (≤2%)\n")
            if inter_r:
                prec_block += "\nIntermediate precision (interday):\n"
                for lbl, v in inter_r.items():
                    prec_block += (f"  {lbl}: mean={v['mean']:.4f} µA, "
                                   f"%RSD={v['rsd']:.2f}%, n={v['n']}, "
                                   f"{'PASS' if v['pass'] else 'FAIL'} (≤5%)\n")

        # Build accuracy block
        acc_block = ""
        if accuracy_rows:
            acc_block = "\nAccuracy / Recovery:\n"
            for row in accuracy_rows:
                acc_block += (f"  {row['QC Level']}: nominal={list(row.values())[1]} {unit}, "
                              f"recovery={row['Recovery (%)']:.2f}%, "
                              f"%RSD={row['%RSD']:.2f}%, "
                              f"{'PASS' if 98<=row['Recovery (%)']<=102 else 'FAIL'}\n")

        # Build standard addition block
        sa_block = ""
        if sa_results:
            sa_block = (f"\nStandard addition: slope={sa_results.get('slope','N/A')}, "
                        f"R²={sa_results.get('R2','N/A')}, "
                        f"C_sample={sa_results.get('sample_concentration','N/A')} {unit}\n")

        prompt = f"""You are an expert analytical chemist and scientific writer.
Write a complete, publication-ready validation section for the following electrochemical sensor study.{journal_note}{lang_note}

ANALYTE: {analyte}
METHOD: Electrochemical sensor (voltammetry)

=== CALIBRATION ===
Linear equation: ip (µA) = {slope} × C + {intercept}   [C in {unit}]
Correlation coefficient (r): {r}
LOD (3σ/m, IUPAC): {lod} {unit}
LOQ (10σ/m, IUPAC): {loq} {unit}
Linear dynamic range: {lr[0]} – {lr[1]} {unit}
{prec_block}{acc_block}{sa_block}

INSTRUCTIONS:
Write the following sections exactly as they would appear in a high-impact journal manuscript (Analytical Chemistry, Biosensors & Bioelectronics, ACS Sensors, Talanta level):

**1. Methods – Analytical Validation (≈150 words)**
Describe the validation approach referencing ICH Q2(R2) guideline. Mention calibration, repeatability, intermediate precision and accuracy protocols.

**2. Results – Calibration Performance (≈200 words)**
Report and interpret linearity, slope, r, LOD, LOQ. Compare LOD/LOQ to relevant literature values for {analyte} detection if known. Comment on electrocatalytic sensitivity.

**3. Results – Precision and Accuracy (≈200 words)**
Report %RSD values for repeatability and intermediate precision with pass/fail vs ICH criteria. Report recovery percentages for each QC level and assess ICH Q2(R2) compliance (98–102%).

**4. Results – Standard Addition (≈100 words, skip if no data)**
Interpret the standard addition result and discuss matrix effects compensation.

**5. Table caption suggestion**
Suggest one concise table title that covers the validation summary (calibration + precision + accuracy combined).

Use precise numerical values from the data above. Use passive voice and present tense. Do not use bullet points — write flowing scientific prose."""

        return _call(api_key, prompt, model)

    # ── Calibration-only interpretation (original behaviour) ─────────────
    prompt = f"""Interpret the following analytical calibration results for {analyte}
obtained with an electrochemical sensor.

Linear equation:  ip (µA) = {slope} · C + {intercept}   (C in {unit})
Correlation coefficient r: {r}
LOD (3σ/m, IUPAC): {lod} {unit}
LOQ (10σ/m, IUPAC): {loq} {unit}
Linear dynamic range: {lr[0]} – {lr[1]} {unit}

Discuss:
1. Linearity quality and whether r meets the typical ≥ 0.9990 criterion
   for analytical sensor publications.
2. The LOD/LOQ values in the context of physiological or environmental
   relevance (compare to known literature values if possible).
3. The sensitivity (slope) and what it implies about electrocatalytic
   activity toward {analyte}.
4. Recommendations for real-sample applicability and any matrix effects
   to anticipate."""

    return _call(api_key, prompt, model)


def interpret_standard_addition(api_key: str, results: dict,
                                 analyte: str = "analyte", unit: str = "µM", model: str = "claude-opus-5-5") -> str:
    c_samp = results.get("C_sample_uM", "N/A")
    r2     = results.get("R2",          "N/A")
    slope  = results.get("slope",       "N/A")

    prompt = f"""Interpret the following standard addition result for {analyte}
in a real sample, measured by electrochemical detection.

Calculated sample concentration: {c_samp} {unit}
Regression slope: {slope} µA {unit}⁻¹  (R² = {r2})

Discuss:
1. The reliability of the standard addition result, judging from R²
   and whether the slope is consistent with the calibration sensitivity.
2. How standard addition compensates for matrix effects compared to
   external calibration, and whether the R² achieved here is sufficient.
3. Any concern about spike recovery, dilution linearity, or species
   speciation that should be verified before reporting the result.
4. Confidence in the reported concentration and suggested follow-up validation."""

    return _call(api_key, prompt, model)


def interpret_full_project(api_key: str, project: dict, model: str = "claude-opus-5-5") -> str:
    """Generate an executive summary for the whole project."""
    analyses = project.get("analyses", {})
    if not analyses:
        return "No analyses recorded in this project yet."

    sections = []

    # CV
    if "cv" in analyses:
        m = analyses["cv"].get("metrics", {})
        pds = analyses["cv"].get("param_df_str", "")
        ecsa = m.get("ecsa", m.get("ECSA_cm2", ""))
        b    = m.get("slope_b", m.get("b_value", ""))
        k0   = m.get("k0_mean", m.get("k0", ""))
        r2   = m.get("R2", m.get("r2", ""))
        s = f"CV (Scan-Rate Study): ECSA={ecsa} cm², b={b} (log ip vs log ν), k°={k0} cm s⁻¹, R²={r2}"
        if pds: s += f"\nFit table:\n{pds[:400]}"
        sections.append(s)

    # EIS
    if "eis" in analyses:
        m = analyses["eis"].get("metrics", {})
        pds = analyses["eis"].get("param_df_str", "")
        electrodes = m.get("electrodes", [])
        # Try to pull Rct values
        rct_vals = m.get("Rct_ohm", m.get("Rct", m.get("rct", [])))
        rs_vals  = m.get("Rs_ohm",  m.get("Rs",  m.get("rs",  [])))
        n_vals   = m.get("n_cpe",   m.get("n",   []))
        s = f"EIS (Randles-CPE): electrodes={electrodes}, Rs={rs_vals}, Rct={rct_vals}, n(CPE)={n_vals}"
        if pds: s += f"\nFit table:\n{pds[:600]}"
        sections.append(s)

    # pH
    if "ph" in analyses:
        m = analyses["ph"].get("metrics", {})
        pds = analyses["ph"].get("param_df_str", "")
        slope  = m.get("slope_mV_per_pH", m.get("slope", ""))
        r2     = m.get("R2", m.get("r2", ""))
        opt_ph = m.get("optimal_pH", "")
        s = f"pH Study: Ep/pH slope={slope} mV pH⁻¹, R²={r2}, optimal pH={opt_ph}"
        if pds: s += f"\n{pds[:300]}"
        sections.append(s)

    # Calibration
    if "cal" in analyses:
        m = analyses["cal"].get("metrics", {})
        lod  = m.get("LOD", m.get("lod", ""))
        loq  = m.get("LOQ", m.get("loq", ""))
        sens = m.get("slope", m.get("sensitivity", ""))
        r    = m.get("r", m.get("R2", ""))
        lr   = m.get("linear_range", ["", ""])
        sections.append(
            f"Calibration: sensitivity={sens} µA µM⁻¹, LOD={lod} µM, LOQ={loq} µM, "
            f"r={r}, linear range={lr[0]}–{lr[1]} µM"
        )

    # Standard addition
    if "sa" in analyses:
        m = analyses["sa"].get("metrics", {})
        c   = m.get("C_sample_uM", m.get("concentration", ""))
        r2  = m.get("R2", "")
        sections.append(f"Standard Addition: [sample]={c} µM, R²={r2}")

    summary_block = "\n\n".join(sections)
    proj_name = project.get("name", "Untitled Project")

    # Kayıtlı AI yorumlarını da ekle (varsa)
    interp_block = []
    for key in ("cv", "eis", "ph", "cal", "sa"):
        if key in analyses:
            saved_interp = analyses[key].get("interpretation", "").strip()
            if saved_interp:
                label = {"cv":"CV","eis":"EIS","ph":"pH","cal":"Calibration","sa":"Standard Addition"}[key]
                interp_block.append(f"[{label} AI interpretation]\n{saved_interp[:600]}")
    interp_section = ("\n\n".join(interp_block)
                      if interp_block else "")

    prompt = f"""You are writing the Abstract and Discussion Summary for a manuscript
about the electrochemical sensor study titled "{proj_name}".

Quantitative experimental results:
{summary_block}
{"" if not interp_section else chr(10) + "Per-module AI interpretations (summaries):" + chr(10) + interp_section}

Write exactly 3 paragraphs in the style required by Biosensors & Bioelectronics
or Analytical Chemistry:

Paragraph 1 — Motivation and electrode design strategy (2–3 sentences):
  State the analytical problem addressed, the electrode material/modification
  approach, and why this system was chosen.

Paragraph 2 — Key electrochemical characterisation findings (3–4 sentences):
  Report the most important numbers: Rct changes from EIS (cite exact Ω values),
  ECSA and b-value from CV, pH dependence slope vs Nernst theory.
  Use precise SI units throughout.{"" if not interp_section else " Incorporate insights from the per-module interpretations where they add nuance."}

Paragraph 3 — Analytical performance and practical relevance (2–3 sentences):
  State LOD, LOQ, linear dynamic range, and sensitivity.
  Compare to at least one literature benchmark if the data suggest it.
  End with the key take-away sentence about applicability (real samples, clinical or
  environmental significance).

RULES:
- Do NOT use bullet points or headers.
- Quote every numeric value present in the data above.
- If a value is missing, omit that parameter rather than saying "data not provided".
- Total length: {"220–280" if interp_section else "180–220"} words."""

    return _call(api_key, prompt, model)
