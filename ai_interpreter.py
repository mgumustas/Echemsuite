"""
SilverstoneLab — AI Interpreter
Calls the Anthropic Claude API to generate electrochemistry-expert
commentary on analysis results from each module.
"""

from __future__ import annotations
import json
import random


# ─────────────────────────────────────────────────────────────
# WRITING STYLE VARIANTS  (picked randomly each call)
# ─────────────────────────────────────────────────────────────
_STYLE_VARIANTS = [
    {
        "name": "terse-quantitative",
        "addon": (
            "Write in a terse, data-dense style. Sentences are short and declarative. "
            "Lead with the most striking numerical result. Compress every argument. "
            "Do NOT start with 'The' or 'In this study'. Start with a data point or a direct claim."
        ),
        "frame": (
            "Report the most critical values first, then explain their significance. "
            "Skip preamble. Each paragraph should open with a number."
        ),
    },
    {
        "name": "narrative-mechanistic",
        "addon": (
            "Weave the data into a mechanistic story. Explain WHY each result occurs, "
            "not just what it is. Connect electrode design logic to analytical outcome. "
            "Let cause and effect drive the paragraph structure."
        ),
        "frame": (
            "Tell the story of what these data reveal about the electrode interface. "
            "Lead with mechanism, follow with numbers, conclude with what this enables."
        ),
    },
    {
        "name": "comparative-critical",
        "addon": (
            "Ground every claim in comparison — Nernst theory, literature benchmarks, or prior art. "
            "Use precise comparative language: 'superior to', 'consistent with', 'falls short of', "
            "'notably lower than'. Be willing to flag weaknesses alongside strengths."
        ),
        "frame": (
            "Frame each result against an external benchmark. The final paragraph must "
            "explicitly situate this sensor's performance relative to the state of the art."
        ),
    },
    {
        "name": "concise-nature-letter",
        "addon": (
            "Write in compressed Nature or Angewandte Chemie letter style. "
            "Every sentence carries new information. No hedging, no throat-clearing, no repetition. "
            "Pack more data into fewer words. Sentences may be long if they carry multiple claims."
        ),
        "frame": (
            "Open mid-argument, as if continuing a discussion already in progress. "
            "Do not restate the method — interpret it."
        ),
    },
    {
        "name": "contextual-applied",
        "addon": (
            "Emphasise real-world significance throughout. Connect every analytical figure-of-merit "
            "to a practical context — clinical threshold, environmental regulation, pharmacopoeia limit, "
            "or typical in-vivo concentration. The reader should understand not just what the number is "
            "but why it matters."
        ),
        "frame": (
            "Open with the clinical or environmental problem, develop through the data that solve it, "
            "close with what this sensor can realistically be deployed for."
        ),
    },
]

# ─────────────────────────────────────────────────────────────
# FULL-PROJECT PARAGRAPH STRUCTURES  (rotated randomly)
# ─────────────────────────────────────────────────────────────
_FULL_PROJECT_STRUCTURES = [
    # A — Classic: rationale → characterisation → analytical
    """\
Paragraph 1 — Electrode design rationale (2–3 sentences):
  State the analytical challenge, the modification strategy, and the mechanistic reasoning behind it.
Paragraph 2 — Electrochemical characterisation (3–4 sentences):
  Report Rct changes from EIS (exact Ω values), ECSA and b-value from CV, pH slope vs Nernst theory.
  Interpret what each number reveals about the electrode interface.
Paragraph 3 — Analytical performance and applicability (2–3 sentences):
  State LOD, LOQ, linear range, sensitivity. Compare to a literature benchmark.
  Close with a one-sentence take-away on where this sensor can be deployed.""",

    # B — Lead with the headline number
    """\
Paragraph 1 — Headline analytical achievement (2–3 sentences):
  Open with LOD and linear range as the top result. Put the sensor's performance in immediate context —
  compare to a clinical threshold or typical literature value.
Paragraph 2 — Why it works: electrode design and interface characterisation (3–4 sentences):
  Explain what makes the LOD possible. Report Rct reduction (exact Ω), ECSA increase, b-value,
  and pH slope — each tied mechanistically to the design choice that produced it.
Paragraph 3 — Validation and outlook (2 sentences):
  Address real-sample recovery, selectivity margin, and the practical scenario where this sensor
  provides analytical value.""",

    # C — Problem-to-solution narrative
    """\
Paragraph 1 — The sensing problem (2 sentences):
  State why detecting the target analyte is analytically difficult and what conventional methods
  fail to deliver at this concentration or in this matrix.
Paragraph 2 — The engineered solution and its electrochemical validation (3–4 sentences):
  Describe the electrode architecture. Show that EIS (Rct), CV (b, ECSA), and pH data confirm
  the design works as intended — each result as evidence for the design hypothesis.
Paragraph 3 — Performance, comparison, and deployment (3 sentences):
  Report the figures of merit, compare to recent literature sensors for the same analyte,
  and state what this sensor can realistically be used for.""",
]


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (base — style addon appended per call)
# ─────────────────────────────────────────────────────────────
_SYSTEM_BASE = """You are a senior electrochemist with expertise in
modified electrode design, electrocatalysis, and electroanalytical
chemistry. You interpret experimental results in the style of a
peer-reviewed analytical chemistry paper (Analytical Chemistry,
Biosensors & Bioelectronics, Electrochimica Acta).

Rules (always apply):
- Quote numeric values exactly as given; use correct SI units.
- Do NOT use bullet points or headers inside your commentary.
- Do NOT start with "Certainly", "Great", or any affirmation phrase.
- CRITICAL — formatting: Do NOT use LaTeX math notation ($...$, \\mu, \\text, _{}, ^{}).
  Instead use plain Unicode symbols directly: Ω, µA, µM, cm², s⁻¹, °C, etc.
  Write variable names in plain text: Rct, Ipa, k0, ECSA, E1/2, ΔEp."""

# Keep the old name as alias for backward compat
_SYSTEM = _SYSTEM_BASE


def _pick_style() -> dict:
    """Her çağrıda rastgele bir yazım stili seç."""
    return random.choice(_STYLE_VARIANTS)


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPER
# ─────────────────────────────────────────────────────────────
def _call(api_key: str, user_msg: str, model: str = "claude-opus-5-5",
          extra_system: str = "") -> str:
    """Call Claude via the Anthropic SDK; returns plain text.
    extra_system: style/tone addon appended to _SYSTEM_BASE per call.
    """
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

    system_prompt = _SYSTEM_BASE
    if extra_system:
        system_prompt = _SYSTEM_BASE + "\n\nWriting style for THIS response:\n" + extra_system

    client = anthropic.Anthropic(api_key=api_key)

    last_err = None
    for _m in _FALLBACK_MODELS:
        try:
            msg = client.messages.create(
                model      = _m,
                max_tokens = 4096,
                system     = system_prompt,
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

    _style = _pick_style()
    prompt = f"""Interpret the following cyclic voltammetry scan-rate study results for
the electrode: {electrode_name}.

ECSA (Randles-Ševčík): {ecsa_v} cm² (R² = {r2_ecsa})
Current–scan-rate mechanism (log ip vs log ν slope b): {b}  → classified as: {mech_lbl}
Apparent heterogeneous rate constant k° (Nicholson method): {k0_str}

Framing guidance: {_style['frame']}
Cover: active surface area (ECSA), rate-limiting step (b-value), electron-transfer kinetics (k°),
and any anomaly or optimisation recommendation implied by these numbers.
Write 2–4 paragraphs."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


def interpret_eis(api_key: str, param_df, electrode_labels: list[str] | None = None, model: str = "claude-opus-5-5") -> str:
    if param_df is None or (hasattr(param_df, 'empty') and param_df.empty):
        return "No EIS fit parameters available for interpretation."

    try:
        tbl = param_df.to_string(index=False)
    except Exception:
        tbl = str(param_df)

    labels_str = ", ".join(electrode_labels) if electrode_labels else "not specified"

    _style = _pick_style()
    prompt = f"""Interpret the following EIS data for a manuscript targeting
Analytical Chemistry, Biosensors & Bioelectronics, or Electrochimica Acta.

Electrode series (in order): {labels_str}
Randles-CPE circuit fitting results:

{tbl}

Framing guidance: {_style['frame']}
Ensure you address: Rs stability, Rct trend across the series with exact Ω values and
mechanistic explanation for each step, CPE parameters (n interpretation, C_eff changes),
and circuit-model adequacy. Quote every numeric value. Write 3–4 paragraphs."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


def interpret_ph(api_key: str, results: dict, analyte: str = "analyte", model: str = "claude-opus-5-5") -> str:
    slope    = results.get("slope_mV_per_pH", "N/A")
    r2       = results.get("R2",              "N/A")
    mn       = results.get("ratio_m_n",       "N/A")
    opt_pH   = results.get("optimal_pH",      "N/A")

    _style = _pick_style()
    prompt = f"""Interpret the following pH-dependent electrochemical study for {analyte}.

Peak potential (Ep) vs pH slope: {slope} mV pH⁻¹  (R² = {r2})
Nernst-predicted slope for m/n = {mn}: {-59.2 * float(mn) if isinstance(mn, (int,float)) else 'see ratio'}  mV pH⁻¹
Optimal working pH identified: {opt_pH}

Framing guidance: {_style['frame']}
Cover: Nernstian consistency and what m/n implies, the mechanistic picture
(PCET vs prior equilibrium), practical relevance of the optimal pH, and
any ionic-strength or buffer considerations. Write 2–3 paragraphs."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


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
    _style = _pick_style()
    prompt = f"""Interpret the following analytical calibration results for {analyte}
obtained with an electrochemical sensor.

Linear equation:  ip (µA) = {slope} · C + {intercept}   (C in {unit})
Correlation coefficient r: {r}
LOD (3σ/m, IUPAC): {lod} {unit}
LOQ (10σ/m, IUPAC): {loq} {unit}
Linear dynamic range: {lr[0]} – {lr[1]} {unit}

Framing guidance: {_style['frame']}
Discuss in 2–3 paragraphs:
- Linearity quality and whether r meets the typical ≥ 0.9990 criterion for analytical sensor publications.
- The LOD/LOQ values in the context of physiological or environmental relevance (compare to known
  literature values if possible).
- The sensitivity (slope) and what it implies about electrocatalytic activity toward {analyte}.
- Recommendations for real-sample applicability and any matrix effects to anticipate."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


def interpret_standard_addition(api_key: str, results: dict,
                                 analyte: str = "analyte", unit: str = "µM", model: str = "claude-opus-5-5") -> str:
    c_samp = results.get("C_sample_uM", "N/A")
    r2     = results.get("R2",          "N/A")
    slope  = results.get("slope",       "N/A")

    _style = _pick_style()
    prompt = f"""Interpret the following standard addition result for {analyte}
in a real sample, measured by electrochemical detection.

Calculated sample concentration: {c_samp} {unit}
Regression slope: {slope} µA {unit}⁻¹  (R² = {r2})

Framing guidance: {_style['frame']}
Discuss in 2–3 paragraphs:
- The reliability of the standard addition result, judging from R² and whether the slope is
  consistent with the calibration sensitivity.
- How standard addition compensates for matrix effects compared to external calibration, and
  whether the R² achieved here is sufficient.
- Any concern about spike recovery, dilution linearity, or species speciation that should be
  verified before reporting the result.
- Confidence in the reported concentration and suggested follow-up validation."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


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
    _style = _pick_style()
    prompt = f"""Interpret the following selectivity study for the electrochemical detection of {analyte}.
Interference threshold: ±{threshold_pct:.0f}% signal change.

Results ({n_pass}/{n_total} passed):
{rows_txt}

Framing guidance: {_style['frame']}
Discuss in 2–3 paragraphs:
- The overall selectivity profile — which interferents posed the greatest challenge and why,
  based on expected redox potentials or non-specific binding mechanisms.
- Whether the {threshold_pct:.0f}% threshold is appropriate for the intended application
  (clinical, environmental, or food analysis) and how it compares to ICH or AOAC criteria.
- If any interferents failed, propose practical mitigation strategies (potential selection,
  masking agents, sample pre-treatment, or modified electrode chemistry).
- The practical significance: can the sensor be deployed in real matrices such as serum,
  urine, or food extracts without prior removal of the failing interferents?"""
    return _call(api_key, prompt, model, extra_system=_style["addon"])


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
    _style = _pick_style()
    prompt = f"""Interpret the following storage stability study for an electrochemical sensor
detecting {analyte}. The minimum acceptable recovery threshold is {threshold_pct:.0f}%.

Day-by-day recovery data:
{rows_txt}

Summary: {len(valid)} time points, final recovery = {last_rec}% at day {last_day}, {n_fail} time point(s) below threshold.

Framing guidance: {_style['frame']}
Discuss in 2–3 paragraphs:
- The stability profile — characterise the degradation trend (linear, exponential, plateau)
  and identify the inflection point where performance begins to degrade meaningfully.
- The estimated shelf-life based on the threshold, and how this compares to practical
  requirements for electrochemical sensors reported in the literature (typically 2–8 weeks).
- Likely degradation mechanisms for the electrode modification (leaching of nanomaterials,
  oxidation of mediators, biofouling, electrolyte ingress) based on the observed trend.
- Recommended storage conditions or electrode stabilisation strategies to extend shelf-life."""
    return _call(api_key, prompt, model, extra_system=_style["addon"])


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

    _style = _pick_style()
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

Framing guidance: {_style['frame']}
Discuss thoroughly in 3–4 paragraphs:
- What the change in Rct reveals about electron-transfer kinetics after surface modification —
  whether the modification facilitates or hinders charge transfer and why.
- The correlation between the EIS Rct change and the CV current enhancement, and whether
  they are consistent with the proposed modification mechanism.
- How these surface characterization results support the electrode's suitability for
  electrocatalytic sensing of the target analyte.
- Any anomalies (e.g. Rct increase combined with current enhancement, or vice versa)
  that warrant further investigation.{'''
- SEM/FESEM morphological analysis: Describe the surface morphology in detail — particle size,
  distribution, porosity, aggregation, roughness, and structural features (nanosheets,
  nanoparticles, flower-like structures). Compare bare and modified morphologies if both images
  are present. Explain how the morphology supports the electrochemical findings and how the
  nanostructure may contribute to increased ECSA and improved electrocatalytic performance.''' if _has_sem else ''}"""
    return _call(api_key, prompt, model, extra_system=_style["addon"])


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

    _style  = _pick_style()
    _struct = random.choice(_FULL_PROJECT_STRUCTURES)
    _wcount = random.choice(["190–230", "220–260", "240–290"])

    prompt = f"""You are writing the Abstract and Discussion Summary for a manuscript
about the electrochemical sensor study titled "{proj_name}".

Quantitative experimental results:
{summary_block}
{"" if not interp_section else chr(10) + "Per-module AI interpretations (summaries):" + chr(10) + interp_section}

Write exactly 3 paragraphs following this structure:

{_struct}

RULES:
- Do NOT use bullet points or headers.
- Quote every numeric value present in the data above.
- If a value is missing, omit that parameter rather than saying "data not provided".
{"- Incorporate insights from the per-module interpretations where they add nuance." if interp_section else ""}
- Total length: {_wcount} words.

Framing guidance: {_style['frame']}"""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


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

    _style = _pick_style()
    prompt = f"""Interpret the following DPV calibration data for the electrochemical
detection of {analyte} at a modified electrode.

Calibration parameters:
  Number of calibration points: {n}
  Peak potential window used: {window} V
  Linear range: {lr} {unit}
  Slope (sensitivity): {slope} µA/{unit}
{f"  Normalised sensitivity: {sens_norm} µA/{unit}·cm²" if sens_norm else ""}
  Intercept: {intercept} µA
  R²: {r2}
  LOD (3σ/m): {lod} {unit}
  LOQ (10σ/m): {loq} {unit}

Framing guidance: {_style['frame']}
Cover: linearity and R² quality, LOD/LOQ in real-sample context (clinical/environmental benchmarks),
sensitivity interpretation, normalised sensitivity if available, and any performance limitation
or strength worth highlighting. Use µA, {unit}, cm² directly. Write 2–4 paragraphs."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


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

    _style = _pick_style()
    prompt = f"""Interpret the following spike-recovery data for the DPV determination
of {analyte} in {matrix}.

Recovery data:
  Number of samples: {n}
  Mean recovery: {mean_rec}%
  RSD: {rsd}%
  Individual recoveries:
{_sample_lines}

Framing guidance: {_style['frame']}
Cover: whether recovery and RSD meet standard criteria (95–105 % / RSD ≤5 % for clean matrices;
90–110 % / RSD ≤10 % for complex matrices), matrix effects or interference explanation if
deviations exist, and overall suitability for routine {analyte} determination in {matrix}.
Use µA, {unit} directly. Write 2–3 paragraphs."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


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

    _style = _pick_style()
    prompt = f"""Critically compare the following electrochemical sensor for {analyte}
detection with previously reported sensors in the literature.

This work:
  LOD: {f"{own_lod:.4g}" if isinstance(own_lod, float) else (own_lod if own_lod is not None else "—")} {unit}
  Linear range: {own_lr if own_lr is not None else "—"} {unit}
  Sensitivity: {f"{own_sens:.4f}" if isinstance(own_sens, float) else (own_sens if own_sens is not None else "—")} µA/{unit}

Published sensors:
{_lit_lines if _lit_lines else "  (no literature entries provided)"}

Framing guidance: {_style['frame']}
Provide a concise comparative analysis (3–4 paragraphs) covering:
- How this sensor's LOD compares to the literature — superior, comparable, or inferior — naming
  specific references where the comparison is most relevant.
- How the linear range covers the practically relevant concentration window for {analyte}
  in real samples.
- What advances this sensor offers over existing reported sensors (simpler fabrication, greener
  modifier, lower LOD, wider range, faster response).
- Any weaknesses or gaps that should be acknowledged or addressed in future work.

Use plain text — no LaTeX. Write in the style of a Biosensors & Bioelectronics discussion section."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])


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

    _style = _pick_style()
    prompt = f"""Interpret the following one-factor-at-a-time (OFAT) optimization results
for a DPV electrochemical sensor for {analyte} detection.

Optimized DPV and experimental parameters:
{_param_lines}

Framing guidance: {_style['frame']}
Write 2–3 paragraphs in Methods/Results style (Biosensors & Bioelectronics, Sensors and Actuators B,
Analytica Chimica Acta level) covering:
- For each instrumental DPV parameter (pulse amplitude, pulse width, step potential): explain the
  electrochemical rationale — why too low or too high a value reduces sensitivity or peak resolution.
- For accumulation parameters (time and potential, if present): explain the preconcentration mechanism
  and how the optimum balances sensitivity with measurement throughput.
- For material/loading parameters (modifier concentration, volume, incubation time): interpret in
  terms of surface coverage, diffusion limitations, and electrode fouling.
- One sentence listing all final optimized values as they would appear in a Methods section.
- One sentence acknowledging OFAT's inherent limitation (factor interactions not captured);
  note that response surface methodology (Box-Behnken or central composite design) would be needed.

Use plain text — no LaTeX. Use µA, mV, ms, s, V, µL directly."""

    return _call(api_key, prompt, model, extra_system=_style["addon"])
