"""
pdf_report.py
-----------------
Generates a professional, multi-section PDF version of the resume
analysis report using ReportLab (Platypus).

Public function:
    generate_pdf_report(...) -> bytes

Returns raw PDF bytes, ready to hand to st.download_button() or attach
to an email.
"""

from io import BytesIO
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)

# Brand palette — kept in sync with the app's dark-theme CSS variables
# (--cc-primary / --cc-secondary / --cc-highlight), but the PDF itself
# stays on a white/light background by design: reports are meant to be
# printed and read in any PDF viewer, and a dark page background does
# not print or photocopy well. This gives a "light card" version of
# the same brand rather than a literal dark-mode PDF.
PRIMARY = colors.HexColor("#4F46E5")       # --cc-primary
PRIMARY_DARK = colors.HexColor("#3730A3")  # --cc-primary-dark
SECONDARY = colors.HexColor("#7C3AED")     # --cc-secondary
HIGHLIGHT = colors.HexColor("#06B6D4")     # --cc-highlight
TEXT_DARK = colors.HexColor("#111827")
TEXT_GREY = colors.HexColor("#6B7280")
LIGHT_ROW = colors.HexColor("#EEF2FF")
HIGHLIGHT_ROW = colors.HexColor("#ECFEFF")
GRID_LINE = colors.HexColor("#D1D5DB")

BRAND_NAME = "AI Career Copilot"


def _clean(text) -> str:
    """
    Escape for ReportLab's mini-XML markup and strip characters outside
    Latin-1 (emoji, stars, checkmarks, etc.) since the built-in PDF
    fonts don't include those glyphs and would render as blank boxes.
    """
    if text is None:
        return ""
    text = str(text).encode("latin-1", "ignore").decode("latin-1")
    return _xml_escape(text)


def _multiline(text) -> str:
    """Clean text and convert newlines to <br/> for a single Paragraph."""
    return _clean(text).replace("\r\n", "\n").replace("\n", "<br/>")


def _build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["Eyebrow"] = ParagraphStyle(
        "Eyebrow", parent=base["Normal"],
        fontSize=9, textColor=SECONDARY, alignment=TA_CENTER,
        spaceAfter=6, tracking=1,
    )
    styles["ReportTitle"] = ParagraphStyle(
        "ReportTitle", parent=base["Title"],
        fontSize=22, textColor=PRIMARY_DARK, spaceAfter=4,
    )
    styles["Subtitle"] = ParagraphStyle(
        "Subtitle", parent=base["Normal"],
        fontSize=10, textColor=TEXT_GREY, alignment=TA_CENTER, spaceAfter=18,
    )
    styles["H2"] = ParagraphStyle(
        "H2", parent=base["Heading2"],
        fontSize=14, textColor=PRIMARY_DARK,
        spaceBefore=16, spaceAfter=6,
    )
    styles["H3"] = ParagraphStyle(
        "H3", parent=base["Heading3"],
        fontSize=11.5, textColor=SECONDARY,
        spaceBefore=8, spaceAfter=4,
    )
    styles["Body"] = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontSize=10, textColor=TEXT_DARK, leading=14.5,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet", parent=styles["Body"],
        leftIndent=14, spaceAfter=4,
    )
    styles["Muted"] = ParagraphStyle(
        "Muted", parent=base["Normal"],
        fontSize=9, textColor=TEXT_GREY, spaceAfter=6,
    )
    return styles


def _section_heading(story, styles, title):
    story.append(Paragraph(_clean(title), styles["H2"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=8))


def _bullet_list(story, styles, items, empty_text="None"):
    if not items:
        story.append(Paragraph(_clean(empty_text), styles["Muted"]))
        return
    for item in items:
        story.append(Paragraph(f"&bull;&nbsp;&nbsp;{_clean(item)}", styles["Bullet"]))


def _styled_table(data, col_widths, header=True):
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID_LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_ROW]),
        ]
    else:
        style += [
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_ROW]),
        ]
    table.setStyle(TableStyle(style))
    return table


def _header_bar(canvas_obj, doc):
    """Thin brand-color bar across the top of every page (indigo ->
    purple, approximated with two adjoining rects since ReportLab's
    canvas has no native linear-gradient fill)."""
    canvas_obj.saveState()
    bar_height = 0.12 * inch
    page_width = letter[0]
    canvas_obj.setFillColor(PRIMARY)
    canvas_obj.rect(0, letter[1] - bar_height, page_width * 0.6, bar_height, stroke=0, fill=1)
    canvas_obj.setFillColor(SECONDARY)
    canvas_obj.rect(page_width * 0.6, letter[1] - bar_height, page_width * 0.4, bar_height, stroke=0, fill=1)
    canvas_obj.restoreState()


def _footer(canvas_obj, doc):
    _header_bar(canvas_obj, doc)
    canvas_obj.saveState()
    canvas_obj.setStrokeColor(GRID_LINE)
    canvas_obj.line(0.75 * inch, 0.65 * inch, letter[0] - 0.75 * inch, 0.65 * inch)
    canvas_obj.setFont("Helvetica", 8)
    canvas_obj.setFillColor(TEXT_GREY)
    canvas_obj.drawString(0.75 * inch, 0.5 * inch, BRAND_NAME)
    canvas_obj.drawRightString(
        letter[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}"
    )
    canvas_obj.restoreState()


def generate_pdf_report(
    name: str,
    email: str,
    phone: str,
    job_role: str,
    summary: str,
    jd_provided: bool,
    jd_match_score,
    ats_score: float,
    skills_component: float,
    jd_component: float,
    structure_component: float,
    contact_component: float,
    strength_count: int,
    strength_label: str,
    found_skills: list,
    missing_skills: list,
    missing_source: str,
    suggestions: list,
    general_tips: list,
    technical_questions: list,
    hr_questions: list,
    project_questions: list,
    cover_letter: str,
    tailored_resume: str = "",
) -> bytes:
    """Build the full professional PDF report and return it as bytes."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.85 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="AI Resume Analysis Report",
    )
    styles = _build_styles()
    story = []

    # ---- Title ----
    story.append(Paragraph("AI CAREER COPILOT", styles["Eyebrow"]))
    story.append(Paragraph("AI Resume Analysis Report", styles["ReportTitle"]))
    story.append(Paragraph(f"Generated by {BRAND_NAME}", styles["Subtitle"]))

    # ---- Candidate Details ----
    _section_heading(story, styles, "Candidate Details")
    cand_rows = [
        ["Name", _clean(name)],
        ["Email", _clean(email)],
        ["Phone", _clean(phone)],
        ["Target Role", _clean(job_role)],
    ]
    story.append(_styled_table(cand_rows, [110, 380], header=False))
    story.append(Spacer(1, 4))

    # ---- Resume Summary ----
    _section_heading(story, styles, "Resume Summary")
    story.append(Paragraph(_multiline(summary), styles["Body"]))

    # ---- Scores Overview ----
    _section_heading(story, styles, "Scores Overview")
    jd_match_display = f"{jd_match_score}%" if (jd_provided and jd_match_score is not None) else "No job description provided"
    score_rows = [
        ["Metric", "Value"],
        ["ATS Score", f"{ats_score}%"],
        ["Resume vs JD Match", jd_match_display],
        ["Resume Strength", f"{strength_count}/5 ({_clean(strength_label)})"],
    ]
    score_table = _styled_table(score_rows, [250, 240], header=True)
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 1), (-1, 1), HIGHLIGHT_ROW),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 10))

    # ---- ATS Score Breakdown ----
    story.append(Paragraph("ATS Score Breakdown", styles["H3"]))
    breakdown_rows = [
        ["Component", "Weight", "Score"],
        ["Skills", "40%", f"{skills_component:.1f}%"],
        ["JD Match", "30%", f"{jd_component:.1f}%"],
        ["Resume Structure", "15%", f"{structure_component:.1f}%"],
        ["Contact Info", "15%", f"{contact_component:.1f}%"],
    ]
    story.append(_styled_table(breakdown_rows, [230, 130, 130], header=True))

    # ---- Skills Detected ----
    _section_heading(story, styles, "Skills Detected")
    if found_skills:
        skills_text = ", ".join(s.title() for s in found_skills)
        story.append(Paragraph(_clean(skills_text), styles["Body"]))
    else:
        story.append(Paragraph("No skills detected.", styles["Muted"]))

    # ---- Missing Skills ----
    _section_heading(story, styles, f"Missing Skills (based on {missing_source})")
    _bullet_list(story, styles, [s.title() for s in missing_skills], "None - great coverage!")

    # ---- AI Improvement Suggestions ----
    _section_heading(story, styles, "AI Improvement Suggestions")
    _bullet_list(story, styles, suggestions, "Your resume covers all the key sections!")

    # ---- Resume Improvement Recommendations ----
    _section_heading(story, styles, "Resume Improvement Recommendations")
    _bullet_list(story, styles, general_tips)

    # ---- AI Interview Questions ----
    _section_heading(story, styles, "AI Interview Questions")
    story.append(Paragraph("Technical", styles["H3"]))
    _bullet_list(story, styles, technical_questions)
    story.append(Paragraph("HR", styles["H3"]))
    _bullet_list(story, styles, hr_questions)
    story.append(Paragraph("Project-Based", styles["H3"]))
    _bullet_list(story, styles, project_questions)

    # ---- AI Cover Letter ----
    story.append(PageBreak())
    _section_heading(story, styles, "AI Cover Letter")
    story.append(Paragraph(_multiline(cover_letter), styles["Body"]))

    # ---- Tailored Resume (only if generated) ----
    if tailored_resume and tailored_resume.strip():
        story.append(PageBreak())
        _section_heading(story, styles, "Tailored Resume")
        story.append(Paragraph(_multiline(tailored_resume), styles["Body"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buffer.seek(0)
    return buffer.getvalue()


def generate_simple_pdf(title: str, body_text: str, subtitle: str = "") -> bytes:
    """
    Build a simple, single-section professional PDF (title + optional
    subtitle + body text). Used for standalone attachments like the
    Tailored Resume PDF.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.85 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title=title,
    )
    styles = _build_styles()
    story = [
        Paragraph("AI CAREER COPILOT", styles["Eyebrow"]),
        Paragraph(_clean(title), styles["ReportTitle"]),
    ]
    if subtitle:
        story.append(Paragraph(_clean(subtitle), styles["Subtitle"]))
    else:
        story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=10))
    story.append(Paragraph(_multiline(body_text), styles["Body"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buffer.seek(0)
    return buffer.getvalue()