"""
SilverstoneLab — Electrochemical Analysis Suite
Streamlit web application  |  v2.0
"""

import streamlit as st
import tempfile, os, sys, io, zipfile, re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for Streamlit
from pathlib import Path

# ── make sure sibling modules are importable ─────────────────
sys.path.insert(0, str(Path(__file__).parent))

from cv_module          import run_cv_analysis
from eis_module         import run_eis_analysis
from ph_module          import run_ph_analysis
from analytical_module  import run_analytical_performance, standard_addition

# ═════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "SilverstoneLab | Echem Suite",
    page_icon   = "⚗️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ═════════════════════════════════════════════════════════════
# CSS — dark sidebar + clean metric cards
# ═════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 60%, #0d1b2a 100%);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { color: #c8d8e8 !important; }
[data-testid="stSidebar"] h2 { color: #7ec8e3 !important; }

.block-container { padding-top: 1.4rem; }

.sec-title {
    font-size: 1rem; font-weight: 600;
    color: #1f4e79;
    border-bottom: 2px solid #1f4e79;
    padding-bottom: 3px; margin-top: 1rem;
}
.wm {
    position: fixed; bottom: 10px; right: 14px;
    font-size: 11px; color: #1f4e79;
    opacity: 0.28; font-style: italic;
}
</style>
<div class="wm">© SilverstoneLab</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚗️ SilverstoneLab")
    st.markdown("**Electrochemical Analysis Suite**")
    st.caption("v2.0 — Surface-modified electrode studies")
    st.divider()

    page = st.radio("navigate", label_visibility="collapsed", options=[
        "🏠  Home",
        "📈  CV — Scan Rate Study",
        "🔵  EIS — Impedance",
        "🧪  pH — Working pH",
        "📊  Calibration & LOD/LOQ",
        "➕  Standard Addition",
    ])

    st.divider()
    with st.expander("📋 CSV format guide"):
        st.markdown("""
**CV files** (one per scan rate):
`Potential` (V), `Current` (A)

**EIS files** (one per electrode):
`Freq` (Hz), `Zreal` (Ω), `Zimag` (Ω)

**pH table** (single file):
`pH`, `Ep_V`, `ip_A`

**Calibration table**:
`Concentration`, `ip_1`, `ip_2`, …

**Standard addition**:
`Addition` (µM added), `ip` (µA)
        """)
    st.caption("Plots auto-stamped © SilverstoneLab")


# ═════════════════════════════════════════════════════════════
# UTILITY helpers
# ═════════════════════════════════════════════════════════════

def _dir_to_zip(dirpath):
    """Package all files in a directory into an in-memory ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(Path(dirpath).iterdir()):
            if f.is_file():
                zf.write(f, f.name)
    buf.seek(0)
    return buf.read()


def _show_images(dirpath):
    """Display all PNG images from a directory in a 2-column grid."""
    imgs = sorted([f for f in Path(dirpath).iterdir()
                   if f.suffix.lower() == '.png'])
    if not imgs:
        st.info("No plots were generated.")
        return
    cols = st.columns(min(len(imgs), 2))
    for i, img in enumerate(imgs):
        cols[i % 2].image(str(img), use_container_width=True)


def _rate_from_name(name):
    """Auto-detect scan rate from filename: cv_50mVs.csv → 50."""
    m = re.search(r'(\d+)\s*m[Vv]', name)
    return int(m.group(1)) if m else None


def _download_zip(dirpath, zip_name):
    st.download_button(
        label   = "⬇ Download results (ZIP)",
        data    = _dir_to_zip(dirpath),
        file_name = zip_name,
        mime    = "application/zip",
        use_container_width = True,
    )


# ═════════════════════════════════════════════════════════════
# 🏠  HOME
# ═════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("⚗️ SilverstoneLab — Electrochemical Analysis Suite")
    st.markdown("""
Integrated analysis platform for **surface-modified electrode** studies.
Upload your raw CSV data, set parameters, and get publication-ready
plots + Excel summary in seconds.
""")

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.markdown("""
**Available modules**

| Module | Key outputs |
|--------|-------------|
| 📈 CV — Scan Rate | ECSA, b-value (mechanism), k° (Nicholson), E½ |
| 🔵 EIS — Impedance | Randles-CPE fit, Rct, C_eff, roughness factor |
| 🧪 pH — Working pH | Ep vs pH slope, Nernst fit, optimal pH |
| 📊 Calibration | LOD, LOQ, linear range, %RSD, recovery |
| ➕ Std Addition | C_sample, 95 % prediction interval |
""")

    c2.markdown("""
**Quick-start tips**

- Name CV files `cv_25mVs.csv`, `cv_100mVs.csv` — scan rate is auto-detected.
- EIS files need columns `Freq`, `Zreal`, `Zimag`; sign convention selectable.
- pH table: provide `pH`, `Ep_V`, `ip_A` in one CSV.
- Calibration CSV: first column = concentration, remaining columns = replicates.
- All plots are saved with the **© SilverstoneLab** watermark.
""")

    st.info("Select a module from the left sidebar to begin.")


# ═════════════════════════════════════════════════════════════
# 📈  CV — SCAN RATE STUDY
# ═════════════════════════════════════════════════════════════
elif page == "📈  CV — Scan Rate Study":
    st.title("📈 CV — Scan Rate Study")
    st.caption("Randles-Ševčík ECSA · mechanism (b-value) · Nicholson k°")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="sec-title">Parameters</div>', unsafe_allow_html=True)
        conc_mM    = st.number_input("Probe concentration (mM)", value=1.0, min_value=0.0)
        diff_coeff = st.number_input("Diffusion coefficient D (cm²/s)", value=7.6e-6, format="%.2e")
        n_el       = st.number_input("Number of electrons n", value=1, min_value=1, max_value=6)

    with right:
        st.markdown('<div class="sec-title">Upload CV files</div>', unsafe_allow_html=True)
        st.caption("One CSV per scan rate — columns: `Potential` (V), `Current` (A)")
        cv_uploads = st.file_uploader("CV files", type="csv",
                                       accept_multiple_files=True, key="cv_up")

    # Scan-rate assignment table
    rate_map = {}
    if cv_uploads:
        st.markdown('<div class="sec-title">Scan-rate assignment (mV/s)</div>',
                    unsafe_allow_html=True)
        for f in cv_uploads:
            auto = _rate_from_name(f.name)
            hint = f" (auto-detected)" if auto else ""
            r = st.number_input(
                f"`{f.name}`{hint}",
                value=float(auto) if auto else 100.0,
                min_value=1.0, step=1.0, key=f"cv_rate_{f.name}"
            )
            rate_map[r] = f

    if cv_uploads and st.button("▶ Run CV Analysis", type="primary",
                                 use_container_width=True):
        with st.spinner("Analysing…"):
            tmpdir  = tempfile.mkdtemp()
            out_dir = os.path.join(tmpdir, "cv_out")
            os.makedirs(out_dir)

            disk_map = {}
            for rate, fobj in rate_map.items():
                p = os.path.join(tmpdir, f"cv_{int(rate)}mVs.csv")
                Path(p).write_bytes(fobj.getvalue())
                disk_map[rate] = p

            try:
                res = run_cv_analysis(
                    disk_map,
                    concentration_mM = conc_mM,
                    diffusion_coeff  = diff_coeff,
                    n_electrons      = int(n_el),
                    output_dir       = out_dir,
                )
                st.success("✅ Analysis complete")

                # Key metrics
                ecsa = res.get('ecsa', {})
                mech = res.get('mechanism', {})
                k0   = res.get('k0', {})

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ECSA (cm²)",       f"{ecsa.get('ECSA_cm2', 0):.4f}")
                m2.metric("R² (Randles-Šev.)", f"{ecsa.get('R2', 0):.5f}")
                b = mech.get('slope_b', 0.75)
                label = ("Diffusion" if b < 0.6
                         else "Adsorption" if b > 0.9 else "Mixed")
                m3.metric("b value",           f"{b:.4f}  ({label})")
                k0_val = k0.get('mean', None)
                m4.metric("k° (cm/s)",
                           f"{k0_val:.4e}" if k0_val else "—")

                st.markdown('<div class="sec-title">Plots</div>',
                            unsafe_allow_html=True)
                _show_images(out_dir)
                _download_zip(out_dir, "cv_analysis.zip")

            except Exception as exc:
                st.error(f"Error: {exc}")
                import traceback; st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════
# 🔵  EIS — IMPEDANCE
# ═════════════════════════════════════════════════════════════
elif page == "🔵  EIS — Impedance":
    st.title("🔵 EIS — Impedance Spectroscopy")
    st.caption("Randles-CPE circuit fitting · Nyquist · Bode · Rct comparison")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="sec-title">Parameters</div>', unsafe_allow_html=True)
        sign_z = st.selectbox(
            "Z_imag sign convention",
            options=[-1, 1],
            format_func=lambda x: (
                "−1  (stored as negative — most potentiostats)"
                if x == -1 else
                "+1  (stored as positive)"
            ),
        )

    with right:
        st.markdown('<div class="sec-title">Upload EIS files</div>', unsafe_allow_html=True)
        st.caption("One CSV per electrode — columns: `Freq` (Hz), `Zreal` (Ω), `Zimag` (Ω)")
        eis_uploads = st.file_uploader("EIS files", type="csv",
                                        accept_multiple_files=True, key="eis_up")

    label_map = {}
    if eis_uploads:
        st.markdown('<div class="sec-title">Electrode labels</div>',
                    unsafe_allow_html=True)
        for i, f in enumerate(eis_uploads):
            default_lbl = f.name.replace('.csv', '').replace('_', ' ')
            lbl = st.text_input(f"`{f.name}`", value=default_lbl,
                                 key=f"eis_lbl_{i}")
            label_map[lbl] = f

    if eis_uploads and st.button("▶ Run EIS Analysis", type="primary",
                                  use_container_width=True):
        with st.spinner("Fitting impedance spectra…"):
            tmpdir  = tempfile.mkdtemp()
            out_dir = os.path.join(tmpdir, "eis_out")
            os.makedirs(out_dir)

            file_dict = {}
            for lbl, fobj in label_map.items():
                safe = re.sub(r'[^\w]', '_', lbl)
                p = os.path.join(tmpdir, f"eis_{safe}.csv")
                Path(p).write_bytes(fobj.getvalue())
                file_dict[lbl] = p

            try:
                param_df = run_eis_analysis(
                    file_dict,
                    output_dir = out_dir,
                    sign_zimag = sign_z,
                )
                st.success("✅ Analysis complete")

                if param_df is not None and not param_df.empty:
                    st.markdown('<div class="sec-title">Fit parameters</div>',
                                unsafe_allow_html=True)
                    st.dataframe(param_df, use_container_width=True,
                                 hide_index=True)

                st.markdown('<div class="sec-title">Plots</div>',
                            unsafe_allow_html=True)
                _show_images(out_dir)
                _download_zip(out_dir, "eis_analysis.zip")

            except Exception as exc:
                st.error(f"Error: {exc}")
                import traceback; st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════
# 🧪  pH — WORKING pH
# ═════════════════════════════════════════════════════════════
elif page == "🧪  pH — Working pH":
    st.title("🧪 pH — Working pH Optimisation")
    st.caption("Ep vs pH · Nernst slope (−59.2 mV/pH reference) · optimal pH")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="sec-title">Parameters</div>', unsafe_allow_html=True)
        ph_n_el   = st.number_input("n (electrons)", value=2, min_value=1, max_value=6)
        pk_type   = st.selectbox("Peak type", ["anodic", "cathodic"])

    with right:
        st.markdown('<div class="sec-title">Upload pH table</div>', unsafe_allow_html=True)
        st.caption("Columns: `pH`, `Ep_V`, `ip_A`")
        ph_upload = st.file_uploader("pH table CSV", type="csv", key="ph_up")

        if ph_upload:
            st.dataframe(pd.read_csv(ph_upload).head(10),
                         use_container_width=True, hide_index=True)
            ph_upload.seek(0)

    if ph_upload and st.button("▶ Run pH Analysis", type="primary",
                                use_container_width=True):
        with st.spinner("Analysing pH dependence…"):
            tmpdir  = tempfile.mkdtemp()
            out_dir = os.path.join(tmpdir, "ph_out")
            os.makedirs(out_dir)

            ph_path = os.path.join(tmpdir, "ph_data.csv")
            Path(ph_path).write_bytes(ph_upload.getvalue())

            try:
                res = run_ph_analysis(
                    source      = ph_path,       # table CSV path
                    n_electrons = int(ph_n_el),
                    peak_type   = pk_type,
                    output_dir  = out_dir,
                )
                st.success("✅ Analysis complete")

                theory = -59.2 * res.get('m_proton', ph_n_el) / int(ph_n_el)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Slope (mV/pH)",
                           f"{res.get('slope_mV_per_pH', 0):.2f}",
                           delta=f"Theory: {theory:.1f}")
                m2.metric("R²",        f"{res.get('R2', 0):.5f}")
                m3.metric("m/n ratio", f"{res.get('ratio_m_n', 0):.3f}")
                m4.metric("Optimal pH",f"{res.get('optimal_pH', 0):.1f}")

                st.markdown('<div class="sec-title">Plots</div>',
                            unsafe_allow_html=True)
                _show_images(out_dir)
                _download_zip(out_dir, "ph_analysis.zip")

            except Exception as exc:
                st.error(f"Error: {exc}")
                import traceback; st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════
# 📊  CALIBRATION & LOD/LOQ
# ═════════════════════════════════════════════════════════════
elif page == "📊  Calibration & LOD/LOQ":
    st.title("📊 Calibration Curve, LOD & LOQ")
    st.caption("IUPAC LOD = 3 s/m · LOQ = 10 s/m · 95 % prediction interval")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="sec-title">Parameters</div>', unsafe_allow_html=True)
        analyte   = st.text_input("Analyte name", value="Analyte")
        conc_unit = st.text_input("Concentration unit", value="μM")
        cal_n_el  = st.number_input("n (electrons)", value=2, min_value=1)
        area_cal  = st.number_input("Electrode area (cm²) — optional, for normalised sensitivity",
                                     value=0.0, min_value=0.0, format="%.4f")

    with right:
        st.markdown('<div class="sec-title">Upload calibration table</div>',
                    unsafe_allow_html=True)
        st.caption("""
First column: concentration values.
Remaining columns: replicate current values (µA).
        """)
        cal_upload = st.file_uploader("Calibration CSV", type="csv", key="cal_up")

        if cal_upload:
            df_prev = pd.read_csv(cal_upload)
            st.dataframe(df_prev, use_container_width=True, hide_index=True)
            cal_upload.seek(0)

    if cal_upload and st.button("▶ Run Calibration Analysis", type="primary",
                                 use_container_width=True):
        with st.spinner("Computing calibration…"):
            tmpdir  = tempfile.mkdtemp()
            out_dir = os.path.join(tmpdir, "cal_out")
            os.makedirs(out_dir)

            try:
                cal_df = pd.read_csv(cal_upload)
                conc_col = cal_df.columns[0]
                ip_cols  = cal_df.columns[1:]

                # Build {conc: [ip_values]} dict
                cal_data = {}
                for _, row in cal_df.iterrows():
                    vals = [float(row[c]) for c in ip_cols
                            if pd.notna(row[c])]
                    if vals:
                        cal_data[float(row[conc_col])] = vals

                res = run_analytical_performance(
                    calibration_source = cal_data,
                    analyte_name       = analyte,
                    electrode_area_cm2 = float(area_cal) if area_cal > 0 else None,
                    unit               = conc_unit,
                    output_dir         = out_dir,
                )
                st.success("✅ Analysis complete")

                cr = res.get('calibration', res)   # handle both dict shapes
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("LOD",  f"{cr.get('LOD', 0):.4g} {conc_unit}")
                m2.metric("LOQ",  f"{cr.get('LOQ', 0):.4g} {conc_unit}")
                m3.metric("R",    f"{cr.get('r', cr.get('R2', 0)):.5f}")
                m4.metric("Slope",f"{cr.get('slope', 0):.4f} µA/{conc_unit}")

                st.markdown(
                    f"**Regression:** ip = {cr.get('slope',0):.4f} C "
                    f"+ {cr.get('intercept',0):.4f}  "
                    f"({cr.get('linear_range',['?','?'])[0]:.4g}–"
                    f"{cr.get('linear_range',['?','?'])[1]:.4g} {conc_unit})"
                )

                st.markdown('<div class="sec-title">Plots</div>',
                            unsafe_allow_html=True)
                _show_images(out_dir)
                _download_zip(out_dir, "calibration_analysis.zip")

            except Exception as exc:
                st.error(f"Error: {exc}")
                import traceback; st.code(traceback.format_exc())


# ═════════════════════════════════════════════════════════════
# ➕  STANDARD ADDITION
# ═════════════════════════════════════════════════════════════
elif page == "➕  Standard Addition":
    st.title("➕ Standard Addition")
    st.caption("C_sample from x-intercept · 95 % prediction interval")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="sec-title">Parameters</div>', unsafe_allow_html=True)
        sa_analyte  = st.text_input("Analyte name", value="Analyte")
        sa_cu       = st.text_input("Concentration unit", value="μM")

    with right:
        st.markdown('<div class="sec-title">Upload standard addition data</div>',
                    unsafe_allow_html=True)
        st.caption("Two columns: `Addition` (amount added, same unit), `ip` (µA)")
        sa_upload = st.file_uploader("Standard addition CSV", type="csv", key="sa_up")

        if sa_upload:
            df_sa = pd.read_csv(sa_upload)
            st.dataframe(df_sa, use_container_width=True, hide_index=True)
            sa_upload.seek(0)

    if sa_upload and st.button("▶ Run Standard Addition", type="primary",
                                use_container_width=True):
        with st.spinner("Computing…"):
            tmpdir  = tempfile.mkdtemp()
            out_dir = os.path.join(tmpdir, "sa_out")
            os.makedirs(out_dir)

            try:
                sa_df    = pd.read_csv(sa_upload)
                add_col  = sa_df.columns[0]
                ip_col   = sa_df.columns[1]

                res = standard_addition(
                    additions_uM = sa_df[add_col].tolist(),
                    ip_uA        = sa_df[ip_col].tolist(),
                    analyte_name = sa_analyte,
                    output_dir   = out_dir,
                )
                st.success("✅ Analysis complete")

                m1, m2, m3 = st.columns(3)
                m1.metric(f"C_sample ({sa_cu})",
                           f"{res.get('C_sample_uM', 0):.4f}")
                m2.metric("R²", f"{res.get('R2', 0):.5f}")
                m3.metric("Slope",
                           f"{res.get('slope', 0):.4f} µA/{sa_cu}")

                st.markdown('<div class="sec-title">Plots</div>',
                            unsafe_allow_html=True)
                _show_images(out_dir)
                _download_zip(out_dir, "standard_addition.zip")

            except Exception as exc:
                st.error(f"Error: {exc}")
                import traceback; st.code(traceback.format_exc())
