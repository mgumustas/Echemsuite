"""
SilverstoneLab — Report Generator
Produces a publication-ready PDF and Word (.docx) report
from a completed project stored in Streamlit session state.
"""

from __future__ import annotations
import io, os, datetime
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════
# PDF  (reportlab)
# ═══════════════════════════════════════════════════════════

def generate_pdf(project: dict, output_path: str) -> str:
    """Write a PDF report to output_path and return that path."""
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
        from PIL                       import Image as PILImage
    except ImportError as e:
        raise ImportError(f"Missing package for PDF generation: {e}")

    # ── styles ──────────────────────────────────────────────
    styles = getSampleStyleSheet()
    navy   = colors.HexColor("#1f4e79")
    steel  = colors.HexColor("#2e75b6")
    light  = colors.HexColor("#f0f4f8")

    H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                         textColor=navy, fontSize=18, spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                         textColor=steel, fontSize=13, spaceBefore=14, spaceAfter=4)
    H3 = ParagraphStyle("H3", parent=styles["Heading3"],
                         textColor=steel, fontSize=11, spaceBefore=8, spaceAfter=3)
    BODY= ParagraphStyle("BODY", parent=styles["Normal"],
                          fontSize=10, leading=14, spaceAfter=6)
    CAPTION = ParagraphStyle("CAP", parent=styles["Normal"],
                               fontSize=8.5, textColor=colors.grey,
                               spaceAfter=10, alignment=1)
    BRAND= ParagraphStyle("BRAND", parent=styles["Normal"],
                           textColor=navy, fontSize=9, alignment=2)
    ITALIC = ParagraphStyle("ITALIC", parent=styles["Normal"],
                             fontSize=10, leading=14, italic=True,
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
    story.append(Paragraph("⚗ SilverstoneLab", BRAND))
    story.append(Paragraph("Electrochemical Analysis Suite", BRAND))
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph(f"<b>{project.get('name','Untitled Project')}</b>", H1))
    story.append(Paragraph(f"Analysis Report — {date_str}", BODY))
    if project.get("notes"):
        story.append(Paragraph(f"<i>{project['notes']}</i>", ITALIC))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=steel))
    story.append(Spacer(1, 0.6*cm))

    # ── executive summary (AI) ───────────────────────────────
    exec_sum = project.get("executive_summary", "")
    if exec_sum:
        story.append(Paragraph("Executive Summary", H2))
        for para in exec_sum.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), BODY))
        story.append(Spacer(1, 0.4*cm))

    story.append(PageBreak())

    # ── module sections ─────────────────────────────────────
    section_meta = {
        "cv":  ("📈 Cyclic Voltammetry — Scan Rate Study",
                 ["ECSA (cm²)", "ECSA R²", "b value", "Mechanism", "k° (cm/s)"]),
        "eis": ("🔵 Electrochemical Impedance Spectroscopy", []),
        "ph":  ("🧪 pH — Working pH Optimisation",
                 ["Slope (mV/pH)", "R²", "m/n ratio", "Optimal pH"]),
        "cal": ("📊 Calibration Curve, LOD & LOQ",
                 ["LOD", "LOQ", "r", "Slope", "Intercept", "Linear range"]),
        "sa":  ("➕ Standard Addition",
                 ["C_sample (µM)", "R²", "Slope"]),
    }

    for key, (title, _) in section_meta.items():
        if key not in analyses:
            continue
        a = analyses[key]

        story.append(Paragraph(title, H2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=steel))
        story.append(Spacer(1, 0.2*cm))

        # metrics table
        metrics = a.get("metrics", {})
        if metrics:
            story.append(Paragraph("Key Metrics", H3))
            tbl_data = [["Parameter", "Value"]]
            for k, v in metrics.items():
                if v not in (None, "", "N/A"):
                    tbl_data.append([str(k), str(v)])
            if len(tbl_data) > 1:
                tbl = Table(tbl_data, colWidths=[8*cm, 7*cm])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND",  (0,0), (-1,0), navy),
                    ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
                    ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                    ("FONTSIZE",    (0,0), (-1,-1), 9),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, light]),
                    ("GRID",        (0,0), (-1,-1), 0.4, colors.lightgrey),
                    ("LEFTPADDING", (0,0), (-1,-1), 6),
                    ("RIGHTPADDING",(0,0), (-1,-1), 6),
                    ("TOPPADDING",  (0,0), (-1,-1), 4),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 4),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 0.3*cm))

        # EIS param dataframe
        eis_df = a.get("param_df_str", "")
        if key == "eis" and eis_df:
            story.append(Paragraph("Fit Parameters", H3))
            story.append(Paragraph(f"<font face='Courier' size='8'>{eis_df[:1200]}</font>",
                                    BODY))
            story.append(Spacer(1, 0.2*cm))

        # plots
        plot_bytes_list = a.get("plot_bytes", [])
        if plot_bytes_list:
            story.append(Paragraph("Figures", H3))
            for i, pb in enumerate(plot_bytes_list):
                try:
                    img_buf = io.BytesIO(pb)
                    pil = PILImage.open(img_buf)
                    w_px, h_px = pil.size
                    dpi = 150
                    w_cm = min(w_px / dpi * 2.54, 14)
                    h_cm = h_px / w_px * w_cm
                    img_buf.seek(0)
                    rl_img = RLImage(img_buf, width=w_cm*cm, height=h_cm*cm)
                    story.append(KeepTogether([
                        rl_img,
                        Paragraph(f"Figure {i+1}. {title} — plot {i+1}.", CAPTION),
                    ]))
                except Exception:
                    pass

        # AI interpretation
        interp = a.get("interpretation", "")
        if interp:
            story.append(Paragraph("Expert Interpretation", H3))
            for para in interp.split("\n\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), ITALIC))
            story.append(Spacer(1, 0.2*cm))

        story.append(Spacer(1, 0.4*cm))

    # ── footer note ─────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Methods Note", H2))
    story.append(Paragraph(
        "ECSA was calculated using the Randles–Ševčík equation for irreversible processes "
        "(ip = 2.99×10⁵ · n · (αn)^½ · A · D^½ · C · ν^½). The b-parameter (log ip vs log ν) "
        "distinguishes diffusion-controlled (b ≈ 0.5) from adsorption-controlled (b ≈ 1.0) "
        "processes. Heterogeneous rate constants were estimated via the Nicholson (1965) method. "
        "EIS spectra were fitted to a Randles–CPE equivalent circuit: Rs + (CPE ∥ (Rct + Zw)). "
        "LOD and LOQ were calculated as 3s/m and 10s/m (IUPAC), respectively.", BODY))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=navy))
    story.append(Paragraph(
        f"Generated by SilverstoneLab Electrochemical Analysis Suite v2.1 · {date_str} "
        "· © SilverstoneLab", BRAND))

    doc.build(story)
    return output_path


# ═══════════════════════════════════════════════════════════
# WORD (.docx)
# ═══════════════════════════════════════════════════════════

def generate_docx(project: dict, output_path: str) -> str:
    """Write a Word report to output_path and return that path."""
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

    p = doc.add_paragraph(f"Analysis Report — {date_str}")
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

    # ── module sections ──────────────────────────────────────
    section_titles = {
        "cv":  "📈 Cyclic Voltammetry — Scan Rate Study",
        "eis": "🔵 Electrochemical Impedance Spectroscopy",
        "ph":  "🧪 pH — Working pH Optimisation",
        "cal": "📊 Calibration Curve, LOD & LOQ",
        "sa":  "➕ Standard Addition",
    }

    for key, title in section_titles.items():
        if key not in analyses:
            continue
        a = analyses[key]

        h2 = doc.add_heading(title, level=2)
        _set_color(h2.runs[0], STEEL)

        # metrics table
        metrics = a.get("metrics", {})
        if metrics:
            doc.add_heading("Key Metrics", level=3)
            tbl = doc.add_table(rows=1, cols=2)
            tbl.style = "Table Grid"
            hdr = tbl.rows[0].cells
            hdr[0].text = "Parameter"; hdr[1].text = "Value"
            for c in hdr:
                for run in c.paragraphs[0].runs:
                    run.bold = True; run.font.color.rgb = RGBColor(255,255,255)
                # blue header background (via XML shading)
                tc = c._tc
                tcPr = tc.get_or_add_tcPr()
                shd  = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:color'), 'auto')
                shd.set(qn('w:fill'), '1f4e79')
                tcPr.append(shd)
            for k, v in metrics.items():
                if v not in (None, "", "N/A"):
                    row = tbl.add_row().cells
                    row[0].text = str(k); row[1].text = str(v)
            doc.add_paragraph()

        # EIS string
        eis_df_str = a.get("param_df_str", "")
        if key == "eis" and eis_df_str:
            doc.add_heading("Fit Parameters", level=3)
            p = doc.add_paragraph(eis_df_str[:1200])
            p.runs[0].font.name = "Courier New"; p.runs[0].font.size = Pt(8)
            doc.add_paragraph()

        # plots
        for i, pb in enumerate(a.get("plot_bytes", [])):
            try:
                buf = io.BytesIO(pb)
                pil = PILImage.open(buf); w_px, h_px = pil.size
                buf.seek(0)
                width_in = min(w_px / 150, 6.0)
                doc.add_picture(buf, width=Inches(width_in))
                cap = doc.add_paragraph(f"Figure {i+1}. {title}")
                cap.runs[0].italic = True; cap.runs[0].font.size = Pt(9)
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            except Exception:
                pass

        # AI interpretation
        interp = a.get("interpretation", "")
        if interp:
            doc.add_heading("Expert Interpretation", level=3)
            for para in interp.split("\n\n"):
                if para.strip():
                    p = doc.add_paragraph(para.strip())
                    p.runs[0].italic = True if p.runs else None

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

def generate_reports(project: dict, out_dir: str) -> dict[str, str]:
    """Generate both PDF and DOCX; return {'pdf': path, 'docx': path}."""
    os.makedirs(out_dir, exist_ok=True)
    stem  = project.get("name","report").replace(" ","_").replace("/","_")[:50]
    paths = {}
    try:
        paths["pdf"]  = generate_pdf(project, os.path.join(out_dir, f"{stem}.pdf"))
    except Exception as e:
        paths["pdf_error"] = str(e)
    try:
        paths["docx"] = generate_docx(project, os.path.join(out_dir, f"{stem}.docx"))
    except Exception as e:
        paths["docx_error"] = str(e)
    return paths
