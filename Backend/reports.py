# Build local PDF and Word reports from a saved browser result.

from datetime import datetime, timezone, timedelta
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
import re
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator


ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE_NOTE = "Confidence is a model score, not a guarantee of accuracy."
FONT_LOCK = Lock()
REPORT_TEXT = {
    "en": {
        "title": "Claim verification report", "completed": "Completed", "claim": "Claim",
        "english_claim": "English claim translation", "screening": "Input screening",
        "screening_unavailable": "Screening unavailable", "invalid_input": "Check your input",
        "out_of_scope": "Enter a factual claim", "model_result": "Model result",
        "SUPPORTS": "Supports", "REFUTES": "Refutes", "confidence": "Model confidence",
        "legacy": "Saved before input screening was added.",
        "no_evidence": "No passages were found for this claim.", "explanation": "What the evidence says",
        "empty_explanation": "No explanation was returned.", "sources": "Sources",
        "explanation_unavailable": "Explanation unavailable.", "retry": "Please try again later.",
        "reviewed": "Passages reviewed", "evidence": "Evidence used", "passage": "passage",
        "passages": "passages", "original": "Original English evidence", "translation": "Translation",
        "confidence_note": CONFIDENCE_NOTE, "translation_notice": "",
        "prediction_caption": "Prediction with retrieved evidence", "screening_caption": "Input screening result",
        "no_evidence_caption": "No evidence result", "page": "Page",
    },
    "pt": {
        "title": "Relatório de verificação de afirmações", "completed": "Concluído em", "claim": "Afirmação original",
        "english_claim": "Tradução da afirmação para inglês", "screening": "Análise da entrada",
        "screening_unavailable": "Análise indisponível", "invalid_input": "Verifique a sua entrada",
        "out_of_scope": "Introduza uma afirmação factual", "model_result": "Resultado do modelo",
        "SUPPORTS": "Apoia", "REFUTES": "Refuta", "confidence": "Confiança do modelo",
        "legacy": "Guardado antes da introdução da análise da entrada.",
        "no_evidence": "Não foram encontradas passagens para esta afirmação.", "explanation": "O que dizem as evidências",
        "empty_explanation": "Não foi devolvida uma explicação.", "sources": "Fontes",
        "explanation_unavailable": "Explicação indisponível.", "retry": "Tente novamente mais tarde.",
        "reviewed": "Passagens analisadas", "evidence": "Evidências utilizadas", "passage": "passagem",
        "passages": "passagens", "original": "Evidência original em inglês", "translation": "Tradução para português",
        "confidence_note": "A confiança é uma pontuação do modelo, não uma garantia de exatidão.",
        "translation_notice": "Apoio multilingue baseado em tradução. A pesquisa e o BERT utilizam a afirmação em inglês. As traduções podem conter erros.",
        "prediction_caption": "Previsão com evidências recuperadas", "screening_caption": "Resultado da análise da entrada",
        "no_evidence_caption": "Resultado sem evidências", "page": "Página",
    },
    "af": {
        "title": "Verslag oor beweringsverifikasie", "completed": "Voltooi op", "claim": "Oorspronklike bewering",
        "english_claim": "Engelse vertaling van die bewering", "screening": "Invoerkontrole",
        "screening_unavailable": "Invoerkontrole nie beskikbaar nie", "invalid_input": "Gaan jou invoer na",
        "out_of_scope": "Voer 'n feitelike bewering in", "model_result": "Modelresultaat",
        "SUPPORTS": "Ondersteun", "REFUTES": "Weerlê", "confidence": "Modelvertroue",
        "legacy": "Gestoor voordat invoerkontrole bygevoeg is.",
        "no_evidence": "Geen teksgedeeltes is vir hierdie bewering gevind nie.", "explanation": "Wat die bewyse sê",
        "empty_explanation": "Geen verduideliking is teruggestuur nie.", "sources": "Bronne",
        "explanation_unavailable": "Verduideliking nie beskikbaar nie.", "retry": "Probeer asseblief later weer.",
        "reviewed": "Hersiene teksgedeeltes", "evidence": "Gebruikte bewyse", "passage": "teksgedeelte",
        "passages": "teksgedeeltes", "original": "Oorspronklike Engelse bewyse", "translation": "Afrikaanse vertaling",
        "confidence_note": "Vertroue is 'n modeltelling, nie 'n waarborg van akkuraatheid nie.",
        "translation_notice": "Meertalige ondersteuning gebruik vertaling. Soektogte en BERT gebruik die Engelse bewering. Vertalings kan foute bevat.",
        "prediction_caption": "Voorspelling met opgespoorde bewyse", "screening_caption": "Resultaat van invoerkontrole",
        "no_evidence_caption": "Resultaat sonder bewyse", "page": "Bladsy",
    },
}


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)


class Prediction(ReportModel):
    label: Literal["SUPPORTS", "REFUTES"]
    confidence: float = Field(ge=0, le=1)
    chunks: int | None = Field(default=None, ge=1)
    probabilities: dict[Literal["SUPPORTS", "REFUTES"], FiniteFloat] = Field(default_factory=dict)


class Explanation(ReportModel):
    status: Literal["ready", "unavailable", "error"]
    summary: str = Field(default="", max_length=5000)
    citations: list[str] = Field(default_factory=list, max_length=10)
    reason: str | None = Field(default=None, max_length=2000)


class Screening(ReportModel):
    status: Literal["accepted", "invalid_input", "out_of_scope", "unavailable"]
    reason: str | None = Field(default=None, max_length=2000)


class Passage(ReportModel):
    id: str = Field(pattern=r"^E[1-9]\d*$", max_length=8)
    page: str = Field(max_length=2000)
    text: str = Field(max_length=200000)
    url: str = Field(max_length=4096)
    pid: int | None = None
    sentence_id: int | None = None
    score: FiniteFloat | None = None
    emb_score: FiniteFloat | None = None
    translation: str | None = Field(default=None, min_length=1, max_length=200000)

    @field_validator("translation")
    @classmethod
    def nonempty_translation(cls, value):
        if value is not None and not value.strip():
            raise ValueError("An evidence translation cannot be blank")
        return value


class Result(ReportModel):
    claim: str = Field(min_length=1, max_length=4000)
    language: Literal["en", "pt", "af"] = "en"
    english_claim: str | None = Field(default=None, min_length=1, max_length=4000)
    status: Literal["prediction", "no_evidence", "rejected", "screening_unavailable"]
    top_k: int = Field(ge=1, le=10)
    prediction: Prediction | None
    explanation: Explanation
    screening: Screening | None = None
    evidence: list[Passage] = Field(max_length=10)
    notice: str = Field(default="", max_length=5000)
    timings: dict[str, FiniteFloat] = Field(default_factory=dict, max_length=20)

    @model_validator(mode="after")
    def consistent_result(self):
        ids = [item.id for item in self.evidence]
        if (len(set(ids)) != len(ids) or len(ids) > self.top_k
                or not set(self.explanation.citations) <= set(ids)
                or any(len(name) > 80 for name in self.timings)):
            raise ValueError("Inconsistent report evidence")
        if self.status == "prediction" and (self.prediction is None or not ids):
            raise ValueError("Incomplete report prediction")
        if self.english_claim is not None and not self.english_claim.strip():
            raise ValueError("An English claim cannot be blank")
        if self.language == "en" and self.english_claim is not None and self.english_claim != self.claim:
            raise ValueError("English claims must remain unchanged")
        if self.language != "en" and self.status == "prediction":
            if (not self.english_claim or self.screening is None or self.screening.status != "accepted"
                    or self.explanation.status != "ready" or any(not item.translation for item in self.evidence)):
                raise ValueError("Incomplete translated prediction")
        if self.language == "en" and any(item.translation is not None for item in self.evidence):
            raise ValueError("English results must not be labelled as translated evidence")
        if self.status == "no_evidence" and (self.prediction is not None or ids):
            raise ValueError("Inconsistent no-evidence result")
        if self.status == "prediction" and self.screening is not None:
            if self.screening.status != "accepted" or self.explanation.status != "ready":
                raise ValueError("Prediction was not accepted by screening")
        if self.status in ("rejected", "screening_unavailable"):
            allowed = ("invalid_input", "out_of_scope") if self.status == "rejected" else ("unavailable",)
            if (self.prediction is not None or self.screening is None or self.screening.status not in allowed
                    or not (self.screening.reason or "").strip() or self.explanation.status == "ready"
                    or self.explanation.summary or self.explanation.citations):
                raise ValueError("Inconsistent screening result")
        return self


class ReportRecord(ReportModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    saved_at: str = Field(min_length=20, max_length=40)
    result: Result

    @field_validator("saved_at")
    @classmethod
    def valid_completion_time(cls, value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("A completion timezone is required")
        return value


def clean_text(value):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "\ufffd", str(value))


def safe_source_url(value):
    try:
        parsed = urlsplit(value)
        if (parsed.scheme in ("https", "http") and parsed.hostname
                and (parsed.hostname == "wikipedia.org" or parsed.hostname.endswith(".wikipedia.org"))
                and not parsed.username and not parsed.password):
            return value
    except ValueError:
        pass
    return None


def report_blocks(record):
    result = record.result
    labels = REPORT_TEXT[result.language]
    completed = datetime.fromisoformat(record.saved_at.replace("Z", "+00:00"))
    date_format = "%d %B %Y, %H:%M CAT" if result.language == "en" else "%d/%m/%Y, %H:%M CAT"
    completed = completed.astimezone(timezone(timedelta(hours=2))).strftime(date_format)
    source_numbers = {item.id: index for index, item in enumerate(result.evidence, 1)}

    def citation_label(source_id):
        return "[" + str(source_numbers.get(source_id, source_id)) + "]"

    def readable_summary(text):
        def replace_group(match):
            ids = re.split(r",\s*", match.group(1))
            if not all(source_id in source_numbers for source_id in ids):
                return match.group(0)
            return "[" + ", ".join(str(source_numbers[source_id]) for source_id in ids) + "]"
        return re.sub(r"\[(E[1-9]\d*(?:,\s*E[1-9]\d*)*)\]", replace_group, text)

    blocks = [("title", labels["title"]), ("meta", labels["completed"] + " " + completed),
              ("heading", labels["claim"]), ("body", result.claim)]
    if result.language != "en":
        if result.english_claim:
            blocks.extend([("heading", labels["english_claim"]), ("english", result.english_claim)])
        blocks.append(("meta", labels["translation_notice"]))
    blocked = result.status in ("rejected", "screening_unavailable")
    if blocked:
        key = "screening_unavailable" if result.status == "screening_unavailable" else result.screening.status
        blocks.extend([("heading", labels["screening"]), ("body", labels[key]), ("body", result.screening.reason)])
    elif result.prediction is not None:
        blocks.extend([("heading", labels["model_result"]), ("body", labels[result.prediction.label]),
                       ("body", labels["confidence"] + f": {result.prediction.confidence:.1%}")])
        if result.screening is None:
            blocks.append(("meta", labels["legacy"]))
    else:
        blocks.extend([("heading", labels["model_result"]), ("body", labels["no_evidence"])])
    if not blocked and result.status != "no_evidence":
        blocks.append(("heading", labels["explanation"]))
        if result.explanation.status == "ready":
            summary = readable_summary(result.explanation.summary) or labels["empty_explanation"]
            blocks.extend(("body", paragraph.strip()) for paragraph in re.split(r"\n\s*\n", summary) if paragraph.strip())
            if result.explanation.citations:
                blocks.append(("meta", labels["sources"] + ": " + ", ".join(citation_label(item) for item in result.explanation.citations)))
        else:
            blocks.append(("body", labels["explanation_unavailable"] + " " + (result.explanation.reason or labels["retry"])))
    if result.evidence:
        count = len(result.evidence)
        heading = labels["reviewed"] if blocked else labels["evidence"]
        heading += " · " + str(count) + " " + labels["passage" if count == 1 else "passages"]
        blocks.append(("heading", heading))
        for item in result.evidence:
            blocks.append(("source", citation_label(item.id) + " " + item.page.replace("_", " ")))
            if result.language != "en":
                blocks.append(("meta", labels["original"]))
            blocks.extend([("english" if result.language != "en" else "body", item.text), ("url", item.url)])
            if item.translation and result.language != "en":
                blocks.extend([("meta", labels["translation"]), ("body", item.translation)])
    if result.prediction is not None:
        blocks.append(("meta", labels["confidence_note"]))
    return [(kind, clean_text(text)) for kind, text in blocks]


def filename(record, extension):
    stamp = datetime.fromisoformat(record.saved_at.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    return "truecheck-" + stamp + "-" + record.id[:64] + "." + extension


@lru_cache(maxsize=1)
def pdf_fonts():
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    candidates = [
        (Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf", "arialbd.ttf"),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), "DejaVuSans-Bold.ttf"),
        (Path(reportlab.__file__).parent / "fonts/Vera.ttf", "VeraBd.ttf"),
    ]
    with FONT_LOCK:
        regular, bold = next((path, path.with_name(bold)) for path, bold in candidates if path.is_file() and path.with_name(bold).is_file())
        pdfmetrics.registerFont(TTFont("TrueCheckBody", str(regular)))
        pdfmetrics.registerFont(TTFont("TrueCheckBold", str(bold)))
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    return "TrueCheckBody", "TrueCheckBold"


def pdf_report(record):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    normal, bold = pdf_fonts()
    labels = REPORT_TEXT[record.result.language]
    stream = BytesIO()
    document = SimpleDocTemplate(stream, pagesize=letter, rightMargin=54, leftMargin=54,
                                 topMargin=66, bottomMargin=54, title="TrueCheck - " + labels["title"], author="TrueCheck")
    styles = {
        "title": ParagraphStyle("title", fontName=bold, fontSize=22, leading=28, textColor=colors.black, spaceAfter=14, keepWithNext=True),
        "heading": ParagraphStyle("heading", fontName=bold, fontSize=13, leading=18, textColor=colors.black, spaceBefore=15, spaceAfter=7, keepWithNext=True),
        "source": ParagraphStyle("source", fontName=bold, fontSize=11, leading=16, spaceBefore=9, spaceAfter=5, keepWithNext=True),
        "body": ParagraphStyle("body", fontName=normal, fontSize=12, leading=18, spaceAfter=8, alignment=TA_LEFT, splitLongWords=True),
        "meta": ParagraphStyle("meta", fontName=normal, fontSize=10, leading=15, textColor=colors.HexColor("#596579"), spaceAfter=5, splitLongWords=True),
        "url": ParagraphStyle("url", fontName=normal, fontSize=10, leading=15, textColor=colors.HexColor("#1d4ed8"), spaceAfter=7, splitLongWords=True),
    }
    styles["english"] = styles["body"]

    def markup(text):
        value = escape(text).replace("\n", "<br/>")
        # CJK uses the PDF standard's Unicode font while Latin text stays embedded.
        return re.sub(r"[\u2e80-\ua000\uf900-\ufaff]+", lambda match: '<font name="STSong-Light">' + match.group() + '</font>', value)

    flow = []
    for kind, text in report_blocks(record):
        value = markup(text)
        if kind == "url" and safe_source_url(text):
            value = '<link href="' + escape(text, {'"': '&quot;'}) + '">' + value + '</link>'
        flow.append(Paragraph(value, styles[kind]))

    def furniture(canvas, doc):
        canvas.saveState()
        logo = ROOT / "Frontend/assets/logo.png"
        if logo.is_file():
            canvas.drawImage(str(logo), 54, 745, width=19, height=19, mask="auto")
        canvas.setFont(bold, 11)
        canvas.drawString(81, 750, "TrueCheck")
        canvas.setFont(normal, 10)
        canvas.setFillColor(colors.HexColor("#596579"))
        key = "prediction_caption" if record.result.prediction is not None else (
            "screening_caption" if record.result.status in ("rejected", "screening_unavailable") else "no_evidence_caption"
        )
        caption = labels[key]
        canvas.drawString(54, 32, caption)
        canvas.drawRightString(558, 32, labels["page"] + f" {doc.page}")
        canvas.restoreState()

    document.build(flow, onFirstPage=furniture, onLaterPages=furniture)
    return stream.getvalue()


def word_report(record):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor
    document = Document()
    labels = REPORT_TEXT[record.result.language]
    language_tag = {"en": "en-GB", "pt": "pt-PT", "af": "af-ZA"}[record.result.language]
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(.85), Inches(.75)
    section.left_margin = section.right_margin = Inches(.75)
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(8)
    for name, size in (("Title", 22), ("Heading 1", 14), ("Heading 2", 11)):
        style = document.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(14 if name != "Title" else 0)
        style.paragraph_format.space_after = Pt(7)
        borders = style.element.find("w:pPr/w:pBdr", namespaces=style.element.nsmap)
        if borders is not None:
            borders.getparent().remove(borders)
    for style in (normal, document.styles["Title"], document.styles["Heading 1"], document.styles["Heading 2"]):
        properties = style.element.get_or_add_rPr()
        fonts = properties.get_or_add_rFonts()
        for attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            fonts.attrib.pop(qn("w:" + attribute), None)
        language = OxmlElement("w:lang")
        language.set(qn("w:val"), language_tag)
        properties.append(language)
    header = section.header.paragraphs[0]
    logo = ROOT / "Frontend/assets/logo.png"
    if logo.is_file():
        header.add_run().add_picture(str(logo), width=Inches(.23))
    header.add_run("  TrueCheck").bold = True
    for kind, text in report_blocks(record):
        style = {"title": "Title", "heading": "Heading 1", "source": "Heading 2"}.get(kind)
        paragraph = document.add_paragraph(style=style)
        if kind == "url" and safe_source_url(text):
            from docx.opc.constants import RELATIONSHIP_TYPE
            link = OxmlElement("w:hyperlink")
            link.set(qn("r:id"), paragraph.part.relate_to(text, RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
            run = OxmlElement("w:r")
            properties = OxmlElement("w:rPr")
            color = OxmlElement("w:color")
            color.set(qn("w:val"), "1D4ED8")
            properties.append(color)
            run.append(properties)
            content = OxmlElement("w:t")
            content.text = text
            run.append(content)
            link.append(run)
            paragraph._p.append(link)
        else:
            run = paragraph.add_run(text)
            if kind == "english":
                language = OxmlElement("w:lang")
                language.set(qn("w:val"), "en-GB")
                run._r.get_or_add_rPr().append(language)
        if kind in ("meta", "url"):
            paragraph.paragraph_format.space_after = Pt(5)
            for run in paragraph.runs:
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor.from_string("596579" if kind == "meta" else "1D4ED8")
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("TrueCheck | " + labels["page"] + " ")
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    document.core_properties.author = "TrueCheck"
    document.core_properties.title = "TrueCheck - " + labels["title"]
    document.core_properties.language = language_tag
    document.core_properties.comments = "Generated locally from the selected verification result."
    stream = BytesIO()
    document.save(stream)
    return stream.getvalue()
