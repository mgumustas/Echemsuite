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
- Do NOT start with "Certainly" or any affirmation phrase.
- CRITICAL — formatting: Do NOT use LaTeX math notation ($...$, \\mu, \\text, _{}, ^{}).
  Instead use plain Unicode symbols directly: Ω, µA, µM, cm², s⁻¹, °C, etc.
  Write variable names in plain text: Rct, Ipa, k0, ECSA, E1/2, ΔEp."""


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


def interpret_selectivity(api_key: str, results: list, analyte: str = "analyte",
                          threshold_pct: float = 5.0, model: str = "claude-opus-5-5") -> str:
    """AI interpretation for selectivity / interferent study."""
    if not results:
        return "No selectivity data provided."
    n_total = len(results)
    n_pass  = sum(1 for r in results if "✅" in str(r.get("Status", "")))
    fail_list = [r.get("Interferent", r.get("interferent", "?")) for r in results
                 if "❌" in str(r.get("Status", ""))]
    rows_txt = "\n".join(
        f"  {r.get('Interferent', r.get('interferent','?'))}: "
        f"signal change = {r.get('Signal Change (%)', r.get('change_pct','?'))}%, "
        f"status = {r.get('Status','?')}"
        for r in results
    )
    prompt = f"""Interpret the following selectivity study for the electrochemical detection of {analyte}.
Interference threshold: ±{threshold_pct:.0f}% signal change.

Results ({n_pass}/{n_total} passed):
{rows_txt}

Discuss:
1. The overall selectivity profile — which interferents posed the greatest challenge and why,
   based on expected redox potentials or non-specific binding mechanisms.
2. Whether the {threshold_pct:.0f}% threshold is appropriate for the intended application
   (clinical, environmental, or food analysis) and how it compares to ICH or AOAC criteria.
3. If any interferents failed, propose practical mitigation strategies (potential selection,
   masking agents, sample pre-treatment, or modified electrode chemistry).
4. The practical significance: can the sensor be deployed in real matrices such as serum,
   urine, or food extracts without prior removal of the failing interferents?"""
    return _call(api_key, prompt, model)


def interpret_stability(api_key: str, results: list, analyte: str = "analyte",
                        threshold_pct: float = 90.0, model: str = "claude-opus-5-5") -> str:
    """AI interpretation for storage/operational stability study."""
    if not results:
        return "No stability data provided."
    days  = [r.get("Day", r.get("day", "?")) for r in results]
    recs  = [r.get("% Recovery", r.get("recovery_pct", None)) for r in results]
    valid = [(d, r) for d, r in zip(days, recs) if r is not None]
    last_day = valid[-1][0] if valid else "?"
    last_rec = valid[-1][1] if valid else "?"
    n_fail   = sum(1 for r in results if "❌" in str(r.get("Status", "")))
    rows_txt = "\n".join(
        f"  Day {r.get('Day','?')}: {r.get('% Recovery','?')}% recovery — {r.get('Status','?')}"
        for r in results
    )
    prompt = f"""Interpret the following storage stability study for an electrochemical sensor
detecting {analyte}. The minimum acceptable recovery threshold is {threshold_pct:.0f}%.

Day-by-day recovery data:
{rows_txt}

Summary: {len(valid)} time points, final recovery = {last_rec}% at day {last_day}, {n_fail} time point(s) below threshold.

Discuss:
1. The stability profile — characterise the degradation trend (linear, exponential, plateau)
   and identify the inflection point where performance begins to degrade meaningfully.
2. The estimated shelf-life based on the threshold, and how this compares to practical
   requirements for electrochemical sensors reported in the literature (typically 2–8 weeks).
3. Likely degradation mechanisms for the electrode modification (leaching of nanomaterials,
   oxidation of mediators, biofouling, electrolyte ingress) based on the observed trend.
4. Recommended storage conditions or electrode stabilisation strategies to extend shelf-life."""
    return _call(api_key, prompt, model)


def interpret_surface(api_key: str, metrics: dict, electrode_name: str = "modified electrode",
                      model: str = "claude-opus-5-5") -> str:
    """AI interpretation for surface characterization (EIS + CV bare vs modified)."""
    if not metrics:
        return "No surface characterization data provided."
    rct_b = metrics.get("Rct Bare (Ω)", metrics.get("rct_bare", "N/A"))
    rct_m = metrics.get("Rct Modified (Ω)", metrics.get("rct_mod", "N/A"))
    ratio = metrics.get("Rct Ratio (Bare/Mod)", "N/A")
    ipa_b = metrics.get("Ipa Bare (µA)", metrics.get("ipa_bare", "N/A"))
    ipa_m = metrics.get("Ipa Modified (µA)", metrics.get("ipa_mod", "N/A"))
    enhance = metrics.get("Signal Enhancement", "N/A")
    sem_note = metrics.get("SEM observation", "")

    _has_sem = bool(metrics.get("has_sem_images", False))
    _sem_preamble = ""
    if _has_sem:
        _sem_preamble = (
            "\nSEM/FESEM images of the electrode surface are attached. "
            "Carefully examine the morphology of each image provided.\n"
        )

    prompt = f"""Interpret the following surface characterization data for {electrode_name}
compared to bare GCE, obtained by EIS (Nyquist), CV in a redox probe solution, and SEM imaging.
{_sem_preamble}
EIS results:
  Rct (bare GCE) = {rct_b} Ω
  Rct (modified) = {rct_m} Ω
  Rct ratio (bare/modified) = {ratio}

CV results (peak anodic current, ip,a):
  Bare GCE: {ipa_b} µA
  Modified:  {ipa_m} µA
  Signal enhancement: {enhance}
{f"SEM observation: {sem_note}" if sem_note else ""}

Discuss the following points thoroughly:
1. What the change in Rct reveals about the electron-transfer kinetics after surface modification —
   whether the modification facilitates or hinders charge transfer and why.
2. The correlation between the EIS Rct change and the CV current enhancement, and whether
   they are consistent with the proposed modification mechanism.
3. How these surface characterization results support the electrode's suitability for
   electrocatalytic sensing of the target analyte.
4. Any anomalies (e.g. Rct increase combined with current enhancement, or vice versa)
   that warrant further investigation.{'''
5. SEM/FESEM morphological analysis: Describe in detail the surface morphology visible in the
   attached images. Comment on particle size, distribution, porosity, aggregation, surface
   roughness, and any structural features (nanosheets, nanoparticles, flower-like structures, etc.).
   Compare the bare and modified electrode morphologies if both images are present.
   Explain how the observed morphology explains or supports the electrochemical findings (Rct
   reduction, current enhancement). Discuss how the nanostructure may contribute to the
   increased electroactive surface area and improved electrocatalytic performance.''' if _has_sem else ''}"""
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

    # Selectivity
    if "sel" in analyses:
        m = analyses["sel"].get("metrics", {})
        n_tested = m.get("Interferents tested", "?")
        pass_rate = m.get("Pass rate", "?")
        threshold = m.get("Threshold", "?")
        sections.append(f"Selectivity: {n_tested} interferents tested, pass rate={pass_rate}, threshold={threshold}")

    # Stability
    if "stab" in analyses:
        m = analyses["stab"].get("metrics", {})
        days = m.get("Days monitored", "?")
        final_rec = m.get("Final recovery", "?")
        threshold = m.get("Min. threshold", "?")
        status = m.get("Final status", "?")
        sections.append(f"Stability: {days} time points, final recovery={final_rec} (threshold {threshold}), final status={status}")

    # Surface characterization
    if "surf" in analyses:
        m = analyses["surf"].get("metrics", {})
        rct_b = m.get("Rct Bare (Ω)", "?")
        rct_m = m.get("Rct Modified (Ω)", "?")
        ratio = m.get("Rct Ratio (Bare/Mod)", "?")
        enhance = m.get("Signal Enhancement", "?")
        sections.append(f"Surface Characterization: Rct(bare)={rct_b} Ω, Rct(modified)={rct_m} Ω, ratio={ratio}, signal enhancement={enhance}")

    summary_block = "\n\n".join(sections)
    proj_name = project.get("name", "Untitled Project")

    # Include saved AI interpretations from all modules
    _all_labels = {"cv":"CV","eis":"EIS","ph":"pH","cal":"Calibration","sa":"Standard Addition",
                   "sel":"Selectivity","stab":"Stability","surf":"Surface Characterization"}
    interp_block = []
    for key in ("cv", "eis", "ph", "cal", "sa", "sel", "stab", "surf"):
        if key in analyses:
            saved_interp = analyses[key].get("interpretation", "").strip()
            if saved_interp:
                label = _all_labels.get(key, key)
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


# ─────────────────────────────────────────────────────────────
# DPV CALIBRATION INTERPRETATION
# ─────────────────────────────────────────────────────────────
def interpret_dpv_calibration(api_key: str, metrics: dict,
                               model: str = "claude-opus-5-5") -> str:
    """AI interpretation for DPV sensor calibration figures of merit."""
    if not metrics:
        return "No calibration data provided."

    analyte   = metrics.get("analyte", "analyte")
    unit      = metrics.get("unit", "µM")
    slope     = metrics.get("slope", "N/A")
    intercept = metrics.get("intercept", "N/A")
    r2        = metrics.get("R2", "N/A")
    lod       = metrics.get("LOD", "N/A")
    loq       = metrics.get("LOQ", "N/A")
    lr        = metrics.get("linear_range", "N/A")
    sens      = metrics.get("sensitivity", "N/A")
    sens_norm = metrics.get("sensitivity_norm")
    n         = metrics.get("n_points", "N/A")
    window    = metrics.get("peak_window_V", "N/A")

    prompt = f"""Interpret the following DPV (differential pulse voltammetry) calibration data
for the electrochemical detection of {analyte} at a modified electrode.

Calibration parameters:
  Number of calibration points: {n}
  Peak potential window used: {window} V
  Linear range: {lr} {unit}
  Slope (sensitivity): {slope} µA/{unit}
{f"  Normalised sensitivity: {sens_norm} µA/{unit}·cm²" if sens_norm else ""}
  Intercept: {intercept} µA
  Coefficient of determination (R²): {r2}
  Limit of Detection (3σ/m): {lod} {unit}
  Limit of Quantification (10σ/m): {loq} {unit}

Discuss the following:
1. The linearity and quality of the calibration curve: comment on R², the slope, and
   whether the linear range covers practically relevant concentration levels for the target analyte.
2. The analytical figures of merit: evaluate LOD and LOQ in the context of real-sample applications
   (clinical, environmental, food safety) and compare to typical literature benchmarks for this class
   of analyte and electrode.
3. The sensitivity: interpret the slope in terms of electrode responsiveness, and comment on
   the normalised sensitivity if provided.
4. Any aspects that could be improved (e.g. extending the linear range, reducing the LOD,
   increasing the number of calibration points) or that indicate a high-performance sensor.

Use plain text — no LaTeX. Use µA, {unit}, Ω, cm² directly. Total: 2–4 paragraphs."""

    return _call(api_key, prompt, model)


# ─────────────────────────────────────────────────────────────
# REAL SAMPLE ANALYSIS INTERPRETATION
# ─────────────────────────────────────────────────────────────
def interpret_real_sample(api_key: str, metrics: dict,
                           model: str = "claude-opus-5-5") -> str:
    """AI interpretation for spike-recovery real sample analysis."""
    if not metrics:
        return "No real sample data provided."

    analyte   = metrics.get("analyte", "analyte")
    matrix    = metrics.get("matrix", "real sample")
    unit      = metrics.get("unit", "µM")
    mean_rec  = metrics.get("mean_recovery_%", "N/A")
    rsd       = metrics.get("RSD_%", "N/A")
    n         = metrics.get("n_samples", "N/A")
    samples   = metrics.get("samples", [])
    recs      = metrics.get("recoveries", [])

    _sample_lines = "\n".join(
        f"  {s}: {r}%" for s, r in zip(samples, recs)
    ) if samples else "  (no individual sample data)"

    prompt = f"""Interpret the following real sample spike-recovery data for the DPV determination
of {analyte} in {matrix}.

Recovery data:
  Number of samples: {n}
  Mean recovery: {mean_rec}%
  RSD: {rsd}%
  Individual recoveries:
{_sample_lines}

Discuss:
1. Whether the mean recovery and RSD meet the standard acceptability criteria
   (typically 95–105 % recovery and RSD ≤ 5 % for electroanalytical methods; 90–110 % and
   RSD ≤ 10 % for complex matrices) and what this means for practical applicability.
2. The significance of these results for validating the method in {matrix}, including
   any matrix effects or interferences that may explain deviations if present.
3. The suitability of this sensor for routine determination of {analyte} in {matrix}
   based on these recovery data.

Use plain text — no LaTeX. Use µA, {unit} directly. Total: 2–3 paragraphs."""

    return _call(api_key, prompt, model)


# ─────────────────────────────────────────────────────────────
# LITERATURE COMPARISON INTERPRETATION
# ─────────────────────────────────────────────────────────────
def interpret_literature_comparison(api_key: str, metrics: dict,
                                     model: str = "claude-opus-5-5") -> str:
    """AI interpretation summarising the sensor vs literature."""
    if not metrics:
        return "No comparison data provided."

    analyte   = metrics.get("analyte", "analyte")
    unit      = metrics.get("unit", "µM")
    own_lod   = metrics.get("own_LOD")
    own_lr    = metrics.get("own_linear_range")
    own_sens  = metrics.get("own_sensitivity")
    lit       = metrics.get("literature", [])

    _lit_lines = ""
    for _e in lit:
        _lit_lines += (
            f"  {_e.get('Electrode/Modifier','?')} | {_e.get('Method','?')} | "
            f"LOD={_e.get('LOD ('+unit+')', _e.get('LOD','?'))} {unit} | "
            f"LR={_e.get('Linear range ('+unit+')', _e.get('Linear range','?'))} | "
            f"Ref: {_e.get('Reference','?')}\n"
        )

    prompt = f"""Critically compare the following electrochemical sensor for {analyte}
detection with previously reported sensors in the literature.

This work:
  LOD: {own_lod:.4g if isinstance(own_lod, float) else own_lod} {unit}
  Linear range: {own_lr} {unit}
  Sensitivity: {own_sens:.4f if isinstance(own_sens, float) else own_sens} µA/{unit}

Published sensors:
{_lit_lines if _lit_lines else "  (no literature entries provided)"}

Provide a concise comparative analysis (2–4 paragraphs):
1. How does this sensor's LOD compare to the literature? Is it superior, comparable, or inferior?
   Name specific references where the comparison is most relevant.
2. How does the linear range cover the practically relevant concentration window for
   {analyte} in real samples?
3. Overall, what advances does this sensor offer over existing reported sensors
   (e.g. simpler fabrication, greener modifier, lower LOD, wider range, faster response)?
4. Any weaknesses or gaps that should be acknowledged or addressed in future work.

Use plain text — no LaTeX. Write in the style of a Biosensors & Bioelectronics discussion section.
Total: 3–4 paragraphs."""

    return _call(api_key, prompt, model)


# ─────────────────────────────────────────────────────────────
# DPV OPTIMIZATION INTERPRETATION
# ─────────────────────────────────────────────────────────────
def interpret_dpv_optimization(api_key: str, metrics: dict,
                                model: str = "claude-opus-5-5") -> str:
    """AI interpretation of OFAT DPV parameter optimization results."""
    if not metrics:
        return "No optimization data provided."

    analyte = metrics.get("analyte", "analyte")
    params  = metrics.get("optimized_parameters", {})
    if not params:
        return "No optimized parameters recorded yet."

    _param_lines = "\n".join(
        f"  {k}: optimal value = {v}" for k, v in params.items()
    )

    prompt = f"""Interpret the following one-factor-at-a-time (OFAT) optimization results
for a DPV electrochemical sensor for {analyte} detection.

Optimized DPV and experimental parameters:
{_param_lines}

Write a concise Methods/Results-style paragraph (2–3 paragraphs) suitable for a journal paper
(Biosensors & Bioelectronics, Sensors and Actuators B, Analytica Chimica Acta) covering:

1. For each instrumental DPV parameter (pulse amplitude, pulse width, step potential):
   explain the electrochemical rationale behind the optimal value — why too low or too high
   a value reduces sensitivity or peak resolution.
2. For accumulation parameters (time and potential, if present):
   explain the preconcentration mechanism and how the optimum balances sensitivity
   with measurement throughput.
3. For material/loading parameters (modifier concentration, volume, incubation time):
   interpret in terms of surface coverage, diffusion limitations, and electrode fouling.
4. End with one sentence listing all final optimized values as they would appear in a
   Methods section ("The optimal conditions were determined to be: pulse amplitude X mV,
   step potential Y mV, ...").
5. Close with one brief sentence acknowledging the inherent limitation of OFAT optimization:
   that it evaluates each factor independently and therefore cannot capture interaction
   effects between parameters; note that response surface methodology (e.g. Box-Behnken or
   central composite design) would be required to fully characterize factor interactions.

Use plain text — no LaTeX. Use µA, mV, ms, s, V, µL directly."""

    return _call(api_key, prompt, model)
