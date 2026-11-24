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
def _call(api_key: str, user_msg: str, model: str = "claude-opus-4-0") -> str:
    """Call Claude via the Anthropic SDK; returns plain text."""
    try:
        import anthropic
    except ImportError:
        return "⚠️ 'anthropic' package not installed. Add it to requirements.txt."

    _FALLBACK_MODELS = [
        model,
        "claude-opus-4-0",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ]

    # Let the SDK manage its own http client (avoids httpx vs httpx2 version conflicts)
    client = anthropic.Anthropic(api_key=api_key)

    last_err = None
    for _m in _FALLBACK_MODELS:
        try:
            msg = client.messages.create(
                model      = _m,
                max_tokens = 800,
                system     = _SYSTEM,
                messages   = [{"role": "user", "content": user_msg}],
            )
            return msg.content[0].text.strip()
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

def interpret_cv(api_key: str, results: dict, electrode_name: str = "electrode", model: str = "claude-opus-4-0") -> str:
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


def interpret_eis(api_key: str, param_df, electrode_labels: list[str] | None = None, model: str = "claude-opus-4-0") -> str:
    if param_df is None or (hasattr(param_df, 'empty') and param_df.empty):
        return "No EIS fit parameters available for interpretation."

    try:
        tbl = param_df.to_string(index=False)
    except Exception:
        tbl = str(param_df)

    prompt = f"""Interpret the following EIS Randles-CPE fitting results.
Each row is a different electrode (bare → increasingly modified).

{tbl}

Columns typically contain: Rs (solution resistance, Ω),
Rct (charge-transfer resistance, Ω), Q (CPE coefficient, F sⁿ),
n (CPE exponent, dimensionless), C_eff (effective capacitance, F),
Roughness factor, χ² goodness of fit.

Discuss:
1. The trend in Rct across the electrode series and what it implies
   about each modification step (barrier vs. facilitation of electron transfer).
2. CPE exponent n and what deviation from n=1 means for surface homogeneity.
3. The effective double-layer capacitance and how it reflects surface area changes.
4. Overall quality of the impedance model fit and any anomalies."""

    return _call(api_key, prompt, model)


def interpret_ph(api_key: str, results: dict, analyte: str = "analyte", model: str = "claude-opus-4-0") -> str:
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
                           analyte: str = "analyte", unit: str = "µM", model: str = "claude-opus-4-0") -> str:
    cr       = results.get("calibration", results)
    lod      = cr.get("LOD",        "N/A")
    loq      = cr.get("LOQ",        "N/A")
    slope    = cr.get("slope",      "N/A")
    intercept= cr.get("intercept",  "N/A")
    r        = cr.get("r", cr.get("R2", "N/A"))
    lr       = cr.get("linear_range", ["N/A", "N/A"])

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
                                 analyte: str = "analyte", unit: str = "µM", model: str = "claude-opus-4-0") -> str:
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


def interpret_full_project(api_key: str, project: dict, model: str = "claude-opus-4-0") -> str:
    """Generate an executive summary for the whole project."""
    analyses = project.get("analyses", {})
    parts = []
    for key, label in [("cv","CV"),("eis","EIS"),("ph","pH"),
                       ("cal","Calibration"),("sa","Standard Addition")]:
        if key in analyses:
            m = analyses[key].get("metrics", {})
            parts.append(f"{label}: {json.dumps(m, default=str)[:300]}")

    if not parts:
        return "No analyses recorded in this project yet."

    summary = "\n".join(parts)
    prompt = f"""Write a 3-paragraph executive summary for the electrochemical
study titled "{project.get('name','Untitled Project')}".

Available analysis results (abbreviated):
{summary}

The summary should read like the Abstract of an electroanalytical
paper: motivation → key results → analytical performance significance.
Keep it under 200 words."""

    return _call(api_key, prompt, model)
