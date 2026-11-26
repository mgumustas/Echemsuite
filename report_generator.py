"""
SilverstoneLab — Report Generator
Produces a publication-ready PDF and Word (.docx) report
from a completed project stored in Streamlit session state.
"""

from __future__ import annotations
import io, os, re, datetime
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════

def _fmt_metric(v) -> str:
    """Format a stored metric value for display.
    Unwraps single-element lists; joins multi-element lists; formats floats."""
    if isinstance(v, list):
        if len(v) == 0:
            return "—"
        if len(v) == 1:
            v = v[0]
        else:
            return ", ".join(_fmt_metric(x) for x in v)
    if isinstance(v, float):
        # pick a sensible number of sig figs
        if abs(v) >= 1000 or (abs(v) < 0.001 and v != 0):
            return f"{v:.4e}"
        return f"{v:.4g}"
    return str(v)


# Unicode → ASCII fallback table (used when no Unicode font is available)
_UNICODE_FALLBACK = str.maketrans({
    '⁻': '-', '⁺': '+',
    '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    'ⁿ': 'n', '½': '1/2', '¼': '1/4',
    '·': '*', '×': 'x',
    '→': '->', '≈': '~', '≤': '<=', '≥': '>=',
    '∥': '||', '∞': 'inf',
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
    'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
    'λ': 'lambda', 'μ': 'u', 'ν': 'nu', 'π': 'pi',
    'σ': 'sigma', 'τ': 'tau', 'φ': 'phi', 'χ': 'chi',
    'ψ': 'psi', 'ω': 'omega',
    'Ω': 'Ohm', 'Δ': 'Delta', 'Σ': 'Sigma',
    '°': 'deg',
})

_EMOJI_RE = re.compile(
    "["
    u"\U0001F300-\U0001F9FF"   # misc symbols, emoticons
    u"\U00002600-\U000027BF"   # misc symbols
    u"\U0001F600-\U0001F64F"
    u"\U0001F680-\U0001F6FF"
    "]+",
    flags=re.UNICODE,
)

def _strip_emoji(s: str) -> str:
    return _EMOJI_RE.sub("", s).strip()


def _safe_pdf(s: str, unicode_ok: bool) -> str:
    """Prepare text for ReportLab: strip emojis; if no Unicode font, transliterate."""
    s = _strip_emoji(str(s))
    if not unicode_ok:
        s = s.translate(_UNICODE_FALLBACK)
        # Drop any remaining non-Latin-1 characters
        s = s.encode("latin-1", errors="replace").decode("latin-1")
    return s


# ═══════════════════════════════════════════════════════════
# PDF  (reportlab)
# ═══════════════════════════════════════════════════════════

def generate_pdf(project: dict, output_path: str, report_type: str = "detailed") -> str:
    """Write a PDF report to output_path and return that path.

    report_type: 'summary' (metrics + interpretation only, no figures)
                 'detailed' (full output including figures, raw data tables)
    """
    try:
        from reportlab.lib.pagesizes   import A4
        from reportlab.lib             import colors
        from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units       import cm
        from reportlab.platypus        import (SimpleDocTemplate, Paragraph,
                                                Spacer, Table, TableStyle,
                                                Image as RLImage, PageBreak,
                                                HRFlowable)
        from reportlab.platypus        import KeepTogether
        from reportlab.pdfbase         import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from PIL                       import Image as PILImage
    except ImportError as e:
        raise ImportError(f"Missing package for PDF generation: {e}")

    # ── try to register a Unicode-capable font ───────────────
    _UNICODE_FONT = "Helvetica"   # fallback
    _unicode_ok   = False
    _FONT_CANDIDATES = [
        # DejaVu (available on most Linux / Streamlit Cloud)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        # Liberation Sans (alternative on Ubuntu)
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        # Noto (broader coverage)
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for _fp in _FONT_CANDIDATES:
        if os.path.exists(_fp):
            try:
                _fname = Path(_fp).stem.replace("-","").replace("_","")
                pdfmetrics.registerFont(TTFont(_fname, _fp))
                _UNICODE_FONT = _fname
                _unicode_ok   = True
                break
            except Exception:
                continue

    _BOLD_FONT = _UNICODE_FONT + "-Bold" if not _unicode_ok else _UNICODE_FONT
    # For Helvetica the bold variant is "Helvetica-Bold"; for TTFs we reuse the same font
    if not _unicode_ok:
        _BOLD_FONT = "Helvetica-Bold"

    # ── styles ──────────────────────────────────────────────
    styles = getSampleStyleSheet()
    navy   = colors.HexColor("#1f4e79")
    steel  = colors.HexColor("#2e75b6")
    light  = colors.HexColor("#f0f4f8")

    # Helper — apply _safe_pdf with current unicode_ok
    def S(text): return _safe_pdf(text, _unicode_ok)

    H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                         fontName=_UNICODE_FONT,
                         textColor=navy, fontSize=18, spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                         fontName=_UNICODE_FONT,
                         textColor=steel, fontSize=13, spaceBefore=14, spaceAfter=4)
    H3 = ParagraphStyle("H3", parent=styles["Heading3"],
                         fontName=_UNICODE_FONT,
                         textColor=steel, fontSize=11, spaceBefore=8, spaceAfter=3)
    BODY= ParagraphStyle("BODY", parent=styles["Normal"],
                          fontName=_UNICODE_FONT,
                          fontSize=10, leading=14, spaceAfter=6)
    CAPTION = ParagraphStyle("CAP", parent=styles["Normal"],
                               fontName=_UNICODE_FONT,
                               fontSize=8.5, textColor=colors.grey,
                               spaceAfter=10, alignment=1)
    BRAND= ParagraphStyle("BRAND", parent=styles["Normal"],
                           fontName=_UNICODE_FONT,
                           textColor=navy, fontSize=9, alignment=2)
    ITALIC = ParagraphStyle("ITALIC", parent=styles["Normal"],
                             fontName=_UNICODE_FONT,
                             fontSize=10, leading=14,
                             textColor=colors.HexColor("#333355"), spaceAfter=8)

    doc  = SimpleDocTemplate(output_path, pagesize=A4,
                              leftMargin=2.5*cm, rightMargin=2.5*cm,
                              topMargin=2.5*cm, bottomMargin=2*cm)
    story = []

    date_str = datetime.date.today().strftime("%d %B %Y")
    analyses = project.get("analyses", {})

    # ── title page ──────────────────────────────────────────
    story.append(Spacer(1, 1.5*cm))
    story.append(HRFlowable(width="100%", thickness=3, color=navy))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("SilverstoneLab", BRAND))
    story.append(Paragraph("Electrochemical Analysis Suite", BRAND))
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph(f"<b>{S(project.get('name','Untitled Project'))}</b>", H1))
    _rtype_label = "Summary Report" if report_type == "summary" else "Detailed Analysis Report"
    story.append(Paragraph(f"{_rtype_label} — {date_str}", BODY))
    if project.get("notes"):
        story.append(Paragraph(f"<i>{S(project['notes'])}</i>", ITALIC))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=steel))
    story.append(Spacer(1, 0.6*cm))

    # ── executive summary (AI) ───────────────────────────────
    exec_sum = project.get("executive_summary", "")
    if exec_sum:
        story.append(Paragraph("Executive Summary", H2))
        for para in exec_sum.split("\n\n"):
            if para.strip():
                story.append(Paragraph(S(para.strip()), BODY))
        story.append(Spacer(1, 0.4*cm))

    story.append(PageBreak())

    # ── helper: render a standard metrics table ─────────────
    def _pdf_metrics_table(metrics: dict):
        tbl_data = [["Parameter", "Value"]]
        for k, v in metrics.items():
            _v = _fmt_metric(v)
            if _v not in ("", "None", "N/A", "—"):
                tbl_data.append([S(str(k)), S(_v)])
        if len(tbl_data) > 1:
            tbl = Table(tbl_data, colWidths=[8*cm, 7*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), navy),
                ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                ("FONTNAME",      (0,0), (-1,0), _BOLD_FONT),
                ("FONTNAME",      (0,1), (-1,-1), _UNICODE_FONT),
                ("FONTSIZE",      (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, light]),
                ("GRID",          (0,0), (-1,-1), 0.4, colors.lightgrey),
                ("LEFTPADDING",   (0,0), (-1,-1), 6),
                ("RIGHTPADDING",  (0,0), (-1,-1), 6),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.3*cm))

    def _pdf_embed_plots(plot_bytes_list, title, fig_offset=0, dpi=300):
        for i, pb in enumerate(plot_bytes_list):
            try:
                img_buf = io.BytesIO(pb)
                pil = PILImage.open(img_buf)
                # Try to read the actual DPI stored in the image metadata;
                # fall back to the caller-supplied dpi for sizing only.
                img_dpi = pil.info.get("dpi", (dpi, dpi))
                img_dpi_x = img_dpi[0] if isinstance(img_dpi, (tuple, list)) else dpi
                img_dpi_x = max(img_dpi_x, 72)   # guard against 0 / very small
                w_px, h_px = pil.size
                w_cm = min(w_px / img_dpi_x * 2.54, 14)   # max 14 cm wide
                h_cm = h_px / w_px * w_cm
                img_buf.seek(0)
                rl_img = RLImage(img_buf, width=w_cm*cm, height=h_cm*cm)
                story.append(KeepTogether([
                    rl_img,
                    Paragraph(f"Figure {fig_offset+i+1}. {S(title)} — plot {i+1}.", CAPTION),
                ]))
            except Exception:
                pass

    # ── module sections ─────────────────────────────────────
    section_meta = {
        "cv":   "Cyclic Voltammetry — Scan Rate Study",
        "eis":  "Electrochemical Impedance Spectroscopy",
        "ph":   "pH — Working pH Optimisation",
        "cal":  "Calibration Curve, LOD & LOQ",
        "sa":   "Standard Addition",
        "sel":  "Selectivity / Interferent Study",
        "stab": "Sensor Stability",
        "surf": "Surface Characterization",
    }

    _fig_counter = 0
    for key, title in section_meta.items():
        if key not in analyses:
            continue
        a = analyses[key]

        story.append(Paragraph(S(title), H2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=steel))
        story.append(Spacer(1, 0.2*cm))

        # ── metrics table (always shown) ───────────────────
        metrics = a.get("metrics", {})
        if metrics:
            story.append(Paragraph("Key Metrics", H3))
            _pdf_metrics_table(metrics)

        # ── DETAILED-ONLY sections ─────────────────────────
        if report_type == "detailed":

            # EIS param dataframe
            eis_df = a.get("param_df_str", "")
            if key == "eis" and eis_df:
                story.append(Paragraph("Fit Parameters", H3))
                _eis_txt = _safe_pdf(eis_df[:1200], _unicode_ok)
                story.append(Paragraph(f"<font face='Courier' size='8'>{_eis_txt}</font>", BODY))
                story.append(Spacer(1, 0.2*cm))

            # Selectivity data table
            if key == "sel":
                sel_rows = a.get("rows", [])
                if sel_rows:
                    story.append(Paragraph("Interferent Results", H3))
                    _cols = list(sel_rows[0].keys()) if sel_rows else []
                    hdr = [S(str(c)) for c in _cols]
                    tbl_data2 = [hdr] + [[S(str(r.get(c,""))) for c in _cols] for r in sel_rows]
                    tbl2 = Table(tbl_data2, repeatRows=1)
                    tbl2.setStyle(TableStyle([
                        ("BACKGROUND",   (0,0), (-1,0), navy),
                        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
                        ("FONTNAME",     (0,0), (-1,-1), _UNICODE_FONT),
                        ("FONTSIZE",     (0,0), (-1,-1), 8),
                        ("GRID",         (0,0), (-1,-1), 0.4, colors.lightgrey),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,light]),
                        ("LEFTPADDING",  (0,0), (-1,-1), 5),
                        ("RIGHTPADDING", (0,0), (-1,-1), 5),
                    ]))
                    story.append(tbl2)
                    story.append(Spacer(1, 0.3*cm))

            # Stability data table
            if key == "stab":
                stab_rows = a.get("rows", [])
                if stab_rows:
                    story.append(Paragraph("Daily Recovery Data", H3))
                    _cols2 = list(stab_rows[0].keys()) if stab_rows else []
                    hdr2 = [S(str(c)) for c in _cols2]
                    tbl_data3 = [hdr2] + [[S(str(r.get(c,""))) for c in _cols2] for r in stab_rows]
                    tbl3 = Table(tbl_data3, repeatRows=1)
                    tbl3.setStyle(TableStyle([
                        ("BACKGROUND",   (0,0), (-1,0), navy),
                        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
                        ("FONTNAME",     (0,0), (-1,-1), _UNICODE_FONT),
                        ("FONTSIZE",     (0,0), (-1,-1), 8),
                        ("GRID",         (0,0), (-1,-1), 0.4, colors.lightgrey),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,light]),
                        ("LEFTPADDING",  (0,0), (-1,-1), 5),
                        ("RIGHTPADDING", (0,0), (-1,-1), 5),
                    ]))
                    story.append(tbl3)
                    story.append(Spacer(1, 0.3*cm))

            # Embedded figures
            plot_bytes_list = a.get("plot_bytes", [])
            if plot_bytes_list:
                story.append(Paragraph("Figures", H3))
                _pdf_embed_plots(plot_bytes_list, title, _fig_counter)
                _fig_counter += len(plot_bytes_list)

        # ── AI interpretation (always shown) ──────────────
        interp = a.get("interpretation", "")
        if interp:
            story.append(Paragraph("Expert Interpretation", H3))
            for para in interp.split("\n\n"):
                if para.strip():
                    story.append(Paragraph(S(para.strip()), ITALIC))
            story.append(Spacer(1, 0.2*cm))

        story.append(Spacer(1, 0.4*cm))

    # ── footer note ─────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Methods Note", H2))
    _methods = (
        "ECSA was calculated using the Randles-Sevcik equation for irreversible processes "
        "(ip = 2.99x10^5 * n * (alpha*n)^0.5 * A * D^0.5 * C * nu^0.5). "
        "The b-parameter (log ip vs log nu) "
        "distinguishes diffusion-controlled (b ~0.5) from adsorption-controlled (b ~1.0) "
        "processes. Heterogeneous rate constants were estimated via the Nicholson (1965) method. "
        "EIS spectra were fitted to a Randles-CPE equivalent circuit: Rs + (CPE || (Rct + Zw)). "
        "LOD and LOQ were calculated as 3s/m and 10s/m (IUPAC), respectively."
    )
    story.append(Paragraph(S(_methods), BODY))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=navy))
    story.append(Paragraph(
        f"Generated by SilverstoneLab Electrochemical Analysis Suite v3.1  |  {date_str} "
        "  |  SilverstoneLab", BRAND))

    doc.build(story)
    return output_path


# ═══════════════════════════════════════════════════════════
# WORD (.docx)
# ═══════════════════════════════════════════════════════════

def generate_docx(project: dict, output_path: str, report_type: str = "detailed") -> str:
    """Write a Word report to output_path and return that path.

    report_type: 'summary' (metrics + interpretation only)
                 'detailed' (full output including figures and raw tables)
    """
    try:
        from docx                        import Document
        from docx.shared                 import Pt, Cm, RGBColor, Inches
        from docx.enum.text              import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns                import qn
        from docx.oxml                   import OxmlElement
        from PIL                         import Image as PILImage
    except ImportError as e:
        raise ImportError(f"Missing package for Word generation: {e}")

    NAVY  = RGBColor(0x1f, 0x4e, 0x79)
    STEEL = RGBColor(0x2e, 0x75, 0xb6)

    def _set_color(run, rgb): run.font.color.rgb = rgb
    def _hr(doc):
        p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
        pPr = p._p.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr'); bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), '6')
        bottom.set(qn('w:space'), '1'); bottom.set(qn('w:color'), '1f4e79')
        pBdr.append(bottom); pPr.append(pBdr)

    doc = Document()

    # narrow margins
    for section in doc.sections:
        section.left_margin = Cm(2.5); section.right_margin = Cm(2.5)
        section.top_margin  = Cm(2.5); section.bottom_margin = Cm(2)

    date_str  = datetime.date.today().strftime("%d %B %Y")
    analyses  = project.get("analyses", {})

    # ── title ────────────────────────────────────────────────
    _hr(doc)
    t = doc.add_heading("", level=0)
    run = t.add_run("⚗ SilverstoneLab — Electrochemical Analysis Suite")
    run.font.size = Pt(16); _set_color(run, NAVY)
    t.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    h = doc.add_heading(project.get("name", "Untitled Project"), level=1)
    h.runs[0].font.size = Pt(22); _set_color(h.runs[0], NAVY)

    _rtype_lbl = "Summary Report" if report_type == "summary" else "Detailed Analysis Report"
    p = doc.add_paragraph(f"{_rtype_lbl} — {date_str}")
    p.runs[0].font.size = Pt(10)

    if project.get("notes"):
        ni = doc.add_paragraph(project["notes"])
        ni.runs[0].italic = True; ni.runs[0].font.size = Pt(10)

    _hr(doc); doc.add_paragraph()

    # ── exec summary ─────────────────────────────────────────
    exec_sum = project.get("executive_summary", "")
    if exec_sum:
        h2 = doc.add_heading("Executive Summary", level=2)
        _set_color(h2.runs[0], STEEL)
        for para in exec_sum.split("\n\n"):
            if para.strip():
                doc.add_paragraph(para.strip())
        doc.add_page_break()

    # ── helper: add a blue-header metrics table ──────────────
    def _docx_metrics_table(metrics: dict):
        doc.add_heading("Key Metrics", level=3)
        tbl = doc.add_table(rows=1, cols=2)
        tbl.style = "Table Grid"
        hdr = tbl.rows[0].cells
        hdr[0].text = "Parameter"; hdr[1].text = "Value"
        for c in hdr:
            for run in c.paragraphs[0].runs:
                run.bold = True; run.font.color.rgb = RGBColor(255,255,255)
            tc   = c._tc
            tcPr = tc.get_or_add_tcPr()
            shd  = OxmlElement('w:shd')
            shd.set(qn('w:val'), 'clear')
            shd.set(qn('w:color'), 'auto')
            shd.set(qn('w:fill'), '1f4e79')
            tcPr.append(shd)
        for k, v in metrics.items():
            _v = _fmt_metric(v)
            if _v not in ("", "None", "N/A", "—"):
                row = tbl.add_row().cells
                row[0].text = str(k); row[1].text = _v
        doc.add_paragraph()

    def _docx_data_table(heading: str, rows: list[dict]):
        """Generic list-of-dicts → Word table."""
        if not rows:
            return
        doc.add_heading(heading, level=3)
        cols = list(rows[0].keys())
        tbl = doc.add_table(rows=1, cols=len(cols))
        tbl.style = "Table Grid"
        hdr = tbl.rows[0].cells
        for ci, col in enumerate(cols):
            hdr[ci].text = str(col)
            for run in hdr[ci].paragraphs[0].runs:
                run.bold = True
        for r in rows:
            row = tbl.add_row().cells
            for ci, col in enumerate(cols):
                row[ci].text = str(r.get(col, ""))
        doc.add_paragraph()

    # ── module sections ──────────────────────────────────────
    section_titles = {
        "cv":   "📈 Cyclic Voltammetry — Scan Rate Study",
        "eis":  "🔵 Electrochemical Impedance Spectroscopy",
        "ph":   "🧪 pH — Working pH Optimisation",
        "cal":  "📊 Calibration Curve, LOD & LOQ",
        "sa":   "➕ Standard Addition",
        "sel":  "🧬 Selectivity / Interferent Study",
        "stab": "📅 Sensor Stability",
        "surf": "🔬 Surface Characterization",
    }

    _fig_ctr = 0
    for key, title in section_titles.items():
        if key not in analyses:
            continue
        a = analyses[key]

        h2 = doc.add_heading(title, level=2)
        _set_color(h2.runs[0], STEEL)

        # Metrics table (always)
        metrics = a.get("metrics", {})
        if metrics:
            _docx_metrics_table(metrics)

        # ── DETAILED-ONLY content ────────────────────────────
        if report_type == "detailed":

            # EIS fit parameters
            eis_df_str = a.get("param_df_str", "")
            if key == "eis" and eis_df_str:
                doc.add_heading("Fit Parameters", level=3)
                p = doc.add_paragraph(eis_df_str[:1200])
                p.runs[0].font.name = "Courier New"; p.runs[0].font.size = Pt(8)
                doc.add_paragraph()

            # Selectivity interferent table
            if key == "sel":
                _docx_data_table("Interferent Results", a.get("rows", []))

            # Stability daily recovery table
            if key == "stab":
                _docx_data_table("Daily Recovery Data", a.get("rows", []))

            # Figures — use stored DPI metadata for correct sizing at ≥300 DPI
            for i, pb in enumerate(a.get("plot_bytes", [])):
                try:
                    buf = io.BytesIO(pb)
                    pil = PILImage.open(buf)
                    w_px, h_px = pil.size
                    img_dpi = pil.info.get("dpi", (300, 300))
                    img_dpi_x = img_dpi[0] if isinstance(img_dpi, (tuple, list)) else 300
                    img_dpi_x = max(img_dpi_x, 72)
                    width_in = min(w_px / img_dpi_x, 6.0)   # max 6 inches
                    buf.seek(0)
                    doc.add_picture(buf, width=Inches(width_in))
                    cap = doc.add_paragraph(f"Figure {_fig_ctr+i+1}. {title}")
                    cap.runs[0].italic = True; cap.runs[0].font.size = Pt(9)
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception:
                    pass
            _fig_ctr += len(a.get("plot_bytes", []))

        # AI interpretation (always)
        interp = a.get("interpretation", "")
        if interp:
            doc.add_heading("Expert Interpretation", level=3)
            for para in interp.split("\n\n"):
                if para.strip():
                    p = doc.add_paragraph(para.strip())
                    if p.runs:
                        p.runs[0].italic = True

        doc.add_paragraph()

    # ── methods note ─────────────────────────────────────────
    doc.add_page_break()
    h2 = doc.add_heading("Methods Note", level=2)
    _set_color(h2.runs[0], STEEL)
    doc.add_paragraph(
        "ECSA was calculated using the Randles–Ševčík equation for irreversible processes. "
        "The b-parameter distinguishes diffusion-controlled (b ≈ 0.5) from adsorption-controlled "
        "(b ≈ 1.0) processes. Heterogeneous rate constants were estimated via the Nicholson (1965) "
        "method. EIS spectra were fitted to a Randles–CPE equivalent circuit. "
        "LOD and LOQ were calculated as 3s/m and 10s/m (IUPAC), respectively.")
    _hr(doc)
    footer_p = doc.add_paragraph(
        f"Generated by SilverstoneLab Electrochemical Analysis Suite v2.1 · {date_str} "
        "· © SilverstoneLab")
    footer_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer_p.runs[0].font.size = Pt(9)
    _set_color(footer_p.runs[0], NAVY)

    doc.save(output_path)
    return output_path


# ═══════════════════════════════════════════════════════════
# COMBINED
# ═══════════════════════════════════════════════════════════

def generate_reports(project: dict, out_dir: str,
                     report_type: str = "detailed") -> dict[str, str]:
    """Generate PDF and DOCX; return {'pdf': path, 'docx': path}.

    report_type: 'summary' | 'detailed'
    """
    os.makedirs(out_dir, exist_ok=True)
    stem   = project.get("name","report").replace(" ","_").replace("/","_")[:50]
    suffix = "_summary" if report_type == "summary" else "_detailed"
    paths  = {}
    try:
        paths["pdf"]  = generate_pdf(
            project, os.path.join(out_dir, f"{stem}{suffix}.pdf"),
            report_type=report_type)
    except Exception as e:
        paths["pdf_error"] = str(e)
    try:
        paths["docx"] = generate_docx(
            project, os.path.join(out_dir, f"{stem}{suffix}.docx"),
            report_type=report_type)
    except Exception as e:
        paths["docx_error"] = str(e)
    return paths
