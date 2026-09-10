# -*- coding: utf-8 -*-
"""
SmartSheet - ExamVision V5.1
Professional Bilingual Gradio UI
Gradio 6 Compatible

IMPORTANT:
The grading engine in exam_grader.py is NOT modified.
This file only provides the UI and connects to the original engine.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
from PIL import Image

from exam_grader import ExamGrader, GradingError


# =========================================================
# CONFIG
# =========================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("smartsheet")

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# LOGO
# =========================================================

LOGO_PATH = PROJECT_ROOT / "images" / "smart_sheet.jpg"
LOGO_FALLBACK_URL = "https://img.icons8.com/fluency/96/000000/test-passed.png"


# =========================================================
# OUTPUTS
# =========================================================

OUTPUT_ROOT = PROJECT_ROOT / "outputs"

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DEFAULT ANSWER KEY
# =========================================================

DEFAULT_KEY = """1:A
2:B
3:C
4:D
5:A
6:B
7:C
8:D
9:A
10:B"""


# =========================================================
# LANGUAGE TEXTS
# =========================================================

TEXTS = {

    # =====================================================
    # PERSIAN
    # =====================================================

    "fa": {

        "title": "SmartSheet",

        "subtitle":
            "سیستم هوشمند تصحیح و تحلیل پاسخ‌نامه",

        "engine":
            "ExamVision V5.1",

        "language":
            "English",

        "description":
            """
### 🧠 تصحیح هوشمند آزمون

تصویر پاسخ‌نامه را وارد کنید، کلید پاسخ را ثبت کنید
و نتیجه تحلیل را دریافت کنید.

سیستم می‌تواند پاسخ‌های خالی، چندگزینه‌ای،
مبهم و ردیف‌های شناسایی‌نشده را تشخیص دهد.
""",

        "input_section":
            "📥 ورودی و تنظیمات",

        "settings":
            "⚙️ تنظیمات تصحیح",

        "image":
            "📷 تصویر پاسخ‌نامه",

        "answer_key":
            "📝 کلید پاسخ",

        "options":
            "تعداد گزینه‌ها",

        "columns":
            "تعداد ستون فیزیکی",

        "accept":
            "آستانه پذیرش",

        "review":
            "آستانه بازبینی",

        "start":
            "🚀 شروع تصحیح",

        "result":
            "📊 نتیجه تحلیل",

        "report":
            "گزارش تصحیح",

        "annotated":
            "🔍 تصویر تحلیل‌شده",

        "json":
            "📄 گزارش JSON",

        "csv":
            "📊 نتایج CSV",

        "footer":
            "سیستم هوشمند تحلیل و تصحیح آزمون",

        "key_hint":
            "مثلاً:\n1:A\n2:B\n3:C",

        "image_hint":
            "تصویر پاسخ‌نامه را اینجا قرار دهید",

        "language_button":
            "🌐 English",
    },


    # =====================================================
    # ENGLISH
    # =====================================================

    "en": {

        "title":
            "SmartSheet",

        "subtitle":
            "Intelligent Exam Grading & Answer Sheet Analysis",

        "engine":
            "ExamVision V5.1",

        "language":
            "فارسی",

        "description":
            """
### 🧠 Intelligent Exam Grading

Upload an answer sheet, enter the answer key,
and receive a complete analysis.

The system can detect blank answers, multiple marks,
ambiguous answers, and undetected rows.
""",

        "input_section":
            "📥 Input & Configuration",

        "settings":
            "⚙️ Grading Settings",

        "image":
            "📷 Answer Sheet",

        "answer_key":
            "📝 Answer Key",

        "options":
            "Number of Options",

        "columns":
            "Physical Columns",

        "accept":
            "Acceptance Threshold",

        "review":
            "Review Threshold",

        "start":
            "🚀 Start Grading",

        "result":
            "📊 Analysis Result",

        "report":
            "Grading Report",

        "annotated":
            "🔍 Analyzed Image",

        "json":
            "📄 JSON Report",

        "csv":
            "📊 CSV Results",

        "footer":
            "Intelligent Exam Analysis & Grading System",

        "key_hint":
            "Example:\n1:A\n2:B\n3:C",

        "image_hint":
            "Drop your answer sheet here",

        "language_button":
            "🌐 فارسی",
    }
}


# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

def create_run_directory():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    random_id = uuid.uuid4().hex[:8]

    run_dir = (
        OUTPUT_ROOT /
        f"{timestamp}_{random_id}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return run_dir


# =========================================================
# REPORT BUILDER
# =========================================================

def _build_report_text(
    data: dict,
    language: str = "fa"
) -> str:

    summary = data["summary"]

    quality = data.get(
        "image_quality",
        {}
    )

    # =====================================================
    # ENGLISH REPORT
    # =====================================================

    if language == "en":

        lines = [

            "╔" + "═" * 62 + "╗",

            "║"
            + " SmartSheet • ExamVision V5.1 ".center(62)
            + "║",

            "╚" + "═" * 62 + "╝",

            "",

            f"📋 Total Questions: "
            f"{summary['total']}",

            f"🏆 Score: "
            f"{summary['correct']} / "
            f"{summary['total']} "
            f"({summary['percentage']}%)",

            "",

            f"✅ Correct: "
            f"{summary['correct']}",

            f"❌ Wrong: "
            f"{summary['wrong']}",

            f"⚪ Blank: "
            f"{summary['blank']}",

            f"🔎 Review Required: "
            f"{summary['review']}",

            f"❓ Not Detected: "
            f"{summary['not_detected']}",

            "",

            f"📐 Valid Rows: "
            f"{data['row_count']} / "
            f"{summary['total']}",

            f"🔵 Image Candidates: "
            f"{data['candidate_count']}",
        ]

        if quality:

            lines.extend(
                [

                    "",

                    "━━━━━━━━━━ Image Quality ━━━━━━━━━━",

                    f"📏 Resolution: "
                    f"{quality.get('width', '-')} x {quality.get('height', '-')}",

                    f"🔬 Sharpness: "
                    f"{quality.get('sharpness', '-')}%",

                    f"🎚️ Contrast: "
                    f"{quality.get('contrast', '-')}%",

                    f"☀️ Brightness: "
                    f"{quality.get('brightness', '-')}%",

                    f"⭐ Overall: "
                    f"{quality.get('overall', '-')}%",
                ]
            )

        if summary.get(
            "not_detected",
            0
        ):

            lines.extend(
                [

                    "",

                    f"📊 Detected Questions: "
                    f"{summary.get('percentage_of_detected', 0)}%",
                ]
            )

        warnings = data.get(
            "warnings",
            []
        )

        if warnings:

            lines.extend(
                [

                    "",

                    "━━━━━━━━━━ Warnings ━━━━━━━━━━",
                ]
            )

            for warning in warnings:

                lines.append(
                    f"⚠️ {warning}"
                )

        lines.extend(
            [

                "",

                "━━━━━━━━━━ Answer Details ━━━━━━━━━━",
            ]
        )

        status_map = {

            "✅ صحیح":
                "✅ Correct",

            "❌ غلط":
                "❌ Wrong",

            "⚪ بدون پاسخ":
                "⚪ Blank",

            "🔎 مبهم / نیازمند بازبینی":
                "🔎 Review Required",

            "❓ ردیف شناسایی نشد":
                "❓ Not Detected",

            "✅ صحیح / نیاز به بازبینی":
                "✅ Correct / Review",

            "❌ غلط / نیاز به بازبینی":
                "❌ Wrong / Review",

            "🟠 چند پاسخی/باطل":
                "🟠 Multi-mark",
        }

        for result in data.get(
            "results",
            {}
        ).values():

            question = result.get(
                "question",
                "-"
            )

            student = result.get(
                "student",
                "-"
            )

            correct = result.get(
                "correct",
                "-"
            )

            confidence = result.get(
                "confidence",
                0
            )

            status = result.get(
                "status",
                ""
            )

            status = status_map.get(
                status,
                status
            )

            lines.append(
                f"Q{question:02d}: "
                f"Student={student} | "
                f"Key={correct} | "
                f"Confidence={confidence}% | "
                f"{status}"
            )

        return "\n".join(lines)

    # =====================================================
    # PERSIAN REPORT
    # =====================================================

    lines = [

        "╔" + "═" * 62 + "╗",

        "║"
        + " SmartSheet • ExamVision V5.1 ".center(62)
        + "║",

        "╚" + "═" * 62 + "╝",

        "",

        f"📋 تعداد سؤال: "
        f"{summary['total']}",

        f"🏆 نمره: "
        f"{summary['correct']} از "
        f"{summary['total']} "
        f"({summary['percentage']}%)",

        "",

        f"✅ صحیح: "
        f"{summary['correct']}",

        f"❌ غلط: "
        f"{summary['wrong']}",

        f"⚪ بدون پاسخ: "
        f"{summary['blank']}",

        f"🔎 نیازمند بازبینی: "
        f"{summary['review']}",

        f"❓ ردیف شناسایی‌نشده: "
        f"{summary['not_detected']}",

        "",

        f"📐 ردیف‌های معتبر: "
        f"{data['row_count']} از "
        f"{summary['total']}",

        f"🔵 Candidateهای تصویری: "
        f"{data['candidate_count']}",
    ]

    if quality:

        lines.extend(
            [

                "",

                "━━━━━━━━━━ کیفیت تصویر ━━━━━━━━━━",

                f"📏 وضوح: "
                f"{quality.get('width', '-')} × {quality.get('height', '-')}",

                f"🔬 شارپنس: "
                f"{quality.get('sharpness', '-')}%",

                f"🎚️ کنتراست: "
                f"{quality.get('contrast', '-')}%",

                f"☀️ روشنایی: "
                f"{quality.get('brightness', '-')}%",

                f"⭐ وضعیت کلی: "
                f"{quality.get('overall', '-')}%",
            ]
        )

    if summary.get(
        "not_detected",
        0
    ):

        lines.extend(
            [

                "",

                f"📊 درصد سؤالات شناسایی‌شده: "
                f"{summary.get('percentage_of_detected', 0)}%",
            ]
        )

    warnings = data.get(
        "warnings",
        []
    )

    if warnings:

        lines.extend(
            [

                "",

                "━━━━━━━━━━ هشدارها ━━━━━━━━━━",
            ]
        )

        for warning in warnings:

            lines.append(
                f"⚠️ {warning}"
            )

    lines.extend(
        [

            "",

            "━━━━━━━━━━ جزئیات پاسخ‌ها ━━━━━━━━━━",
        ]
    )

    for result in data.get(
        "results",
        {}
    ).values():

        question = result.get(
            "question",
            "-"
        )

        student = result.get(
            "student",
            "-"
        )

        correct = result.get(
            "correct",
            "-"
        )

        confidence = result.get(
            "confidence",
            0
        )

        status = result.get(
            "status",
            ""
        )

        lines.append(
            f"سؤال {question:02d}: "
            f"پاسخ={student} | "
            f"کلید={correct} | "
            f"اعتماد={confidence}% | "
            f"{status}"
        )

    return "\n".join(lines)


# =========================================================
# PROCESS EXAM
# =========================================================

def process_exam(
    image,
    answer_key_text,
    option_count,
    columns,
    accept_threshold,
    review_threshold,
    language,
    last_report_text=None,
    last_annotated=None,
    last_json=None,
    last_csv=None,
):

    # =====================================================
    # VALIDATION
    # =====================================================

    if image is None:

        if language == "en":

            raise gr.Error(
                "Please upload an answer sheet."
            )

        raise gr.Error(
            "لطفاً تصویر برگه را وارد کنید."
        )

    if (
        not answer_key_text
        or not answer_key_text.strip()
    ):

        if language == "en":

            raise gr.Error(
                "Please enter the answer key."
            )

        raise gr.Error(
            "لطفاً کلید پاسخ را وارد کنید."
        )

    # =====================================================
    # TEMP DIRECTORY
    # =====================================================

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="smartsheet_"
        )
    )

    try:

        # =================================================
        # SAVE INPUT IMAGE
        # =================================================

        input_path = (
            temp_dir /
            "exam_input.png"
        )

        if isinstance(
            image,
            Image.Image
        ):

            image.convert(
                "RGB"
            ).save(
                input_path,
                format="PNG"
            )

        elif hasattr(
            image,
            "shape"
        ):

            cv2.imwrite(
                str(input_path),
                cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR
                )
            )

        else:

            raise GradingError(
                "Unsupported image format."
            )

        # =================================================
        # ORIGINAL EXAM GRADER
        # =================================================

        grader = ExamGrader(

            str(input_path),

            answer_key_text,

            option_count=int(
                option_count
            ),

            columns=int(
                columns
            ),

            accept_threshold=float(
                accept_threshold
            ),

            review_threshold=float(
                review_threshold
            ),
        )

        # =================================================
        # RUN ORIGINAL LOGIC
        # =================================================

        data = grader.run()

        # =================================================
        # SAVE OUTPUTS
        # =================================================

        run_dir = (
            create_run_directory()
        )

        grader.save_outputs(
            str(run_dir)
        )

        json_path = (
            run_dir /
            "report.json"
        )

        csv_path = (
            run_dir /
            "results.csv"
        )

        # =================================================
        # ANNOTATED IMAGE
        # =================================================

        annotated = data.get(
            "annotated"
        )

        if annotated is None:

            annotated = grader.annotated

        if annotated is None:

            raise GradingError(
                "No output image was generated."
            )

        annotated_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        # =================================================
        # BUILD REPORT
        # =================================================

        report = _build_report_text(
            data,
            language
        )

        return (

            report,

            annotated_rgb,

            str(json_path),

            str(csv_path),
        )

    except GradingError as exc:

        logger.exception(
            "Grading error"
        )

        if language == "en":

            raise gr.Error(
                f"Grading error: {exc}"
            )

        raise gr.Error(
            f"خطا در تصحیح: {exc}"
        )

    except Exception as exc:

        logger.exception(
            "Unexpected error"
        )

        if language == "en":

            raise gr.Error(
                f"Unexpected error: {exc}"
            )

        raise gr.Error(
            f"خطای غیرمنتظره: {exc}"
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# =========================================================
# LANGUAGE SWITCH - با حفظ نتایج
# =========================================================

def toggle_language(
    current_language,
    report_text=None,
    annotated_image=None,
    json_file=None,
    csv_file=None,
    data_cache=None,
):

    # -----------------------------------------------------
    # Change language
    # -----------------------------------------------------

    if current_language == "fa":

        language = "en"

    else:

        language = "fa"

    t = TEXTS[language]

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    header_html = f"""
    <div class="smart-title">
        {t["title"]}
    </div>

    <div class="smart-subtitle">
        {t["subtitle"]}
    </div>

    <div class="smart-version">
        {t["engine"]}
    </div>
    """

    # -----------------------------------------------------
    # Footer
    # -----------------------------------------------------

    footer_html = f"""
    <div class="smart-footer">

        <strong>
            SmartSheet
        </strong>

        • ExamVision V5.1

        <br>

        {t["footer"]}

    </div>
    """

    # -----------------------------------------------------
    # اگر نتیجه تحلیلی وجود داشت، ترجمه کن
    # -----------------------------------------------------

    new_report_text = report_text

    if report_text and "سؤال" in report_text and language == "en":

        if data_cache:
            new_report_text = _build_report_text(data_cache, "en")
        else:
            new_report_text = report_text.replace(
                "گزارش تصحیح",
                "Grading Report"
            )

    elif report_text and "Question" in report_text and language == "fa":

        if data_cache:
            new_report_text = _build_report_text(data_cache, "fa")
        else:
            new_report_text = report_text.replace(
                "Grading Report",
                "گزارش تصحیح"
            )

    # =====================================================
    # GRADIO 6 - با حفظ نتایج
    # =====================================================

    return (

        # 1 - Language state
        language,

        # 2 - Header
        gr.HTML(
            value=header_html
        ),

        # 3 - Description
        gr.Markdown(
            value=t["description"],
            elem_classes="panel-card"
        ),

        # 4 - Input section
        gr.Markdown(
            value=f"## {t['input_section']}"
        ),

        # 5 - Settings heading
        gr.Markdown(
            value=f"### {t['settings']}"
        ),

        # 6 - Result section
        gr.Markdown(
            value=f"## {t['result']}"
        ),

        # 7 - Image input
        gr.Image(
            type="pil",
            label=t["image"],
            height=520
        ),

        # 8 - Answer key
        gr.Textbox(
            value=DEFAULT_KEY,
            label=t["answer_key"],
            lines=11,
            placeholder=t["key_hint"]
        ),

        # 9 - Options
        gr.Slider(
            minimum=2,
            maximum=5,
            step=1,
            value=4,
            label=t["options"]
        ),

        # 10 - Columns
        gr.Slider(
            minimum=1,
            maximum=4,
            step=1,
            value=1,
            label=t["columns"]
        ),

        # 11 - Accept threshold
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.01,
            value=0.62,
            label=t["accept"]
        ),

        # 12 - Review threshold
        gr.Slider(
            minimum=0.0,
            maximum=100.0,
            step=1.0,
            value=65.0,
            label=t["review"]
        ),

        # 13 - Process button
        gr.Button(
            value=t["start"],
            variant="primary",
            size="lg",
            elem_id="process-button"
        ),

        # 14 - Report (با حفظ مقدار قبلی)
        gr.Textbox(
            value=(
                new_report_text
                if new_report_text
                else t["report"]
            ),
            label=t["report"],
            lines=30,
            interactive=False,
            elem_id="report-box"
        ),

        # 15 - Annotated image (با حفظ مقدار قبلی)
        gr.Image(
            value=(
                annotated_image
                if annotated_image is not None
                else None
            ),
            label=t["annotated"],
            interactive=False,
            height=600
        ),

        # 16 - JSON (با حفظ مقدار قبلی)
        gr.File(
            value=(
                json_file
                if json_file is not None
                else None
            ),
            label=t["json"]
        ),

        # 17 - CSV (با حفظ مقدار قبلی)
        gr.File(
            value=(
                csv_file
                if csv_file is not None
                else None
            ),
            label=t["csv"]
        ),

        # 18 - Language button
        gr.Button(
            value=t["language_button"],
            elem_id="language-button",
            size="sm"
        ),

        # 19 - Footer
        gr.HTML(
            value=footer_html
        ),
    )


# =========================================================
# CUSTOM CSS
# =========================================================

CUSTOM_CSS = """

/* =====================================================
   GLOBAL
===================================================== */

.gradio-container {

    max-width: 1450px !important;

    margin: 0 auto !important;

    padding:
        20px
        25px
        45px
        25px !important;
}


/* =====================================================
   HEADER
===================================================== */

.smart-header {

    position: relative;

    text-align: center;

    padding:
        25px
        20px
        20px;

    margin-bottom: 15px;
}


.smart-title {

    font-size: 44px;

    font-weight: 900;

    letter-spacing: -1px;

    margin-top: 8px;

    line-height: 1.2;
}


.smart-subtitle {

    font-size: 18px;

    opacity: 0.72;

    margin-top: 9px;

    line-height: 1.8;
}


.smart-version {

    display: inline-block;

    margin-top: 13px;

    padding:
        7px
        17px;

    border-radius: 999px;

    background:
        rgba(80, 120, 255, 0.12);

    border:
        1px solid
        rgba(80, 120, 255, 0.20);

    font-size: 13px;

    font-weight: 700;
}


/* =====================================================
   LOGO - فقط خود عکس گرد، بدون کادر
===================================================== */

.smart-logo {

    position: fixed !important;

    top: 15px !important;

    left: 15px !important;

    z-index: 999999 !important;

    width: 60px !important;

    height: 60px !important;

    border-radius: 50% !important;

    overflow: hidden !important;
}

.smart-logo img {

    width: 100% !important;

    height: 100% !important;

    object-fit: cover !important;

    display: block !important;
}


/* =====================================================
   LANGUAGE BUTTON
===================================================== */

#language-button {

    position: fixed !important;

    top: 12px !important;

    right: 12px !important;

    left: auto !important;

    bottom: auto !important;

    width: 88px !important;

    min-width: 88px !important;

    max-width: 88px !important;

    height: 32px !important;

    min-height: 32px !important;

    max-height: 32px !important;

    padding:
        0 !important;

    margin:
        0 !important;

    border-radius:
        999px !important;

    font-size:
        11px !important;

    line-height:
        32px !important;

    font-weight:
        700 !important;

    z-index:
        999999 !important;

    box-shadow:
        0 4px 14px
        rgba(0, 0, 0, 0.10) !important;
}


#language-button button {

    width:
        100% !important;

    min-width:
        100% !important;

    max-width:
        100% !important;

    height:
        32px !important;

    min-height:
        32px !important;

    max-height:
        32px !important;

    padding:
        0 8px !important;

    border-radius:
        999px !important;

    font-size:
        11px !important;
}


/* =====================================================
   CARDS
===================================================== */

.panel-card {

    border-radius:
        22px !important;

    padding:
        20px !important;
}


/* =====================================================
   MARKDOWN
===================================================== */

.markdown {

    line-height:
        1.9 !important;
}


/* =====================================================
   INPUTS
===================================================== */

textarea,
input {

    text-align:
        right !important;
}


textarea {

    line-height:
        1.8 !important;
}


/* =====================================================
   IMAGE
===================================================== */

.image-container {

    border-radius:
        20px !important;

    overflow:
        hidden !important;
}


/* =====================================================
   PROCESS BUTTON
===================================================== */

#process-button {

    min-height:
        62px !important;

    border-radius:
        17px !important;

    font-size:
        18px !important;

    font-weight:
        850 !important;

    margin-top:
        14px !important;

    box-shadow:
        0 8px 25px
        rgba(0, 0, 0, 0.10);
}


/* =====================================================
   REPORT
===================================================== */

#report-box textarea {

    text-align:
        right !important;

    font-family:
        "Segoe UI",
        Tahoma,
        Arial,
        sans-serif !important;

    line-height:
        1.95 !important;

    font-size:
        14px !important;
}


/* =====================================================
   FOOTER
===================================================== */

.smart-footer {

    text-align:
        center;

    opacity:
        0.55;

    font-size:
        13px;

    line-height:
        1.9;

    padding-top:
        35px;

    padding-bottom:
        15px;
}


/* =====================================================
   MOBILE
===================================================== */

@media (max-width: 800px) {

    .gradio-container {

        padding:
            10px !important;
    }


    .smart-title {

        font-size:
            32px;
    }


    .smart-subtitle {

        font-size:
            15px;
    }


    .smart-logo {

        width: 45px !important;

        height: 45px !important;

        top: 10px !important;

        left: 10px !important;
    }


    #language-button {

        position:
            fixed !important;

        top:
            8px !important;

        right:
            8px !important;

        left:
            auto !important;

        bottom:
            auto !important;

        width:
            78px !important;

        min-width:
            78px !important;

        max-width:
            78px !important;

        height:
            29px !important;

        min-height:
            29px !important;

        max-height:
            29px !important;

        margin:
            0 !important;

        padding:
            0 !important;

        z-index:
            999999 !important;
    }


    #language-button button {

        height:
            29px !important;

        min-height:
            29px !important;

        max-height:
            29px !important;

        font-size:
            10px !important;
    }
}

"""


# =========================================================
# GRADIO APPLICATION
# =========================================================

with gr.Blocks(

    title="SmartSheet | ExamVision V5.1",

) as demo:

    # =====================================================
    # HIDDEN LANGUAGE STATE
    # =====================================================

    language_state = gr.Textbox(

        value="fa",

        visible=False,

        interactive=False,
    )

    # =====================================================
    # HEADER
    # =====================================================

    with gr.Column(
        elem_classes="smart-header"
    ):

        # -------------------------------------------------
        # LOGO - فقط خود عکس گرد بدون کادر
        # -------------------------------------------------

        if LOGO_PATH.exists():

            logo = gr.Image(
                value=str(LOGO_PATH),
                show_label=False,
                interactive=False,
                container=False,
                height=60,
                width=60,
                elem_classes="smart-logo"
            )

        else:

            logo = gr.HTML(
                f"""
                <div class="smart-logo">
                    <img src="{LOGO_FALLBACK_URL}" alt="SmartSheet">
                </div>
                """
            )

        # -------------------------------------------------
        # LANGUAGE BUTTON
        # -------------------------------------------------

        language_button = gr.Button(

            "🌐 English",

            elem_id="language-button",

            size="sm",
        )

        # -------------------------------------------------
        # HEADER TEXT
        # -------------------------------------------------

        header_html = gr.HTML(

            """
            <div class="smart-title">
                SmartSheet
            </div>

            <div class="smart-subtitle">
                سیستم هوشمند تصحیح و تحلیل پاسخ‌نامه
            </div>

            <div class="smart-version">
                ExamVision V5.1
            </div>
            """
        )

    # =====================================================
    # DESCRIPTION
    # =====================================================

    description = gr.Markdown(

        TEXTS["fa"]["description"],

        elem_classes="panel-card",
    )

    # =====================================================
    # INPUT SECTION TITLE
    # =====================================================

    input_section = gr.Markdown(

        "## 📥 ورودی و تنظیمات"
    )

    # =====================================================
    # MAIN INPUT AREA
    # =====================================================

    with gr.Row():

        # -------------------------------------------------
        # IMAGE COLUMN
        # -------------------------------------------------

        with gr.Column(

            scale=1,

            elem_classes="panel-card",
        ):

            image_input = gr.Image(

                type="pil",

                label=TEXTS["fa"]["image"],

                height=520,
            )

        # -------------------------------------------------
        # SETTINGS COLUMN
        # -------------------------------------------------

        with gr.Column(

            scale=1,

            elem_classes="panel-card",
        ):

            settings_heading = gr.Markdown(

                f"### {TEXTS['fa']['settings']}"
            )

            # ---------------------------------------------
            # ANSWER KEY
            # ---------------------------------------------

            answer_key = gr.Textbox(

                value=DEFAULT_KEY,

                label=TEXTS["fa"]["answer_key"],

                lines=11,

                placeholder=TEXTS["fa"]["key_hint"],
            )

            # ---------------------------------------------
            # OPTIONS / COLUMNS
            # ---------------------------------------------

            with gr.Row():

                option_count = gr.Slider(

                    minimum=2,

                    maximum=5,

                    step=1,

                    value=4,

                    label=TEXTS["fa"]["options"],
                )

                columns = gr.Slider(

                    minimum=1,

                    maximum=4,

                    step=1,

                    value=1,

                    label=TEXTS["fa"]["columns"],
                )

            # ---------------------------------------------
            # THRESHOLDS
            # ---------------------------------------------

            with gr.Row():

                accept_threshold = gr.Slider(

                    minimum=0.0,

                    maximum=1.0,

                    step=0.01,

                    value=0.62,

                    label=TEXTS["fa"]["accept"],
                )

                review_threshold = gr.Slider(

                    minimum=0.0,

                    maximum=100.0,

                    step=1.0,

                    value=65.0,

                    label=TEXTS["fa"]["review"],
                )

            # ---------------------------------------------
            # START
            # ---------------------------------------------

            process_button = gr.Button(

                TEXTS["fa"]["start"],

                variant="primary",

                size="lg",

                elem_id="process-button",
            )

    # =====================================================
    # RESULT SECTION
    # =====================================================

    result_section = gr.Markdown(

        "## 📊 نتیجه تحلیل"
    )

    # =====================================================
    # RESULT AREA
    # =====================================================

    with gr.Row():

        # -------------------------------------------------
        # REPORT
        # -------------------------------------------------

        with gr.Column(

            scale=1,

            elem_classes="panel-card",
        ):

            report_output = gr.Textbox(

                label=TEXTS["fa"]["report"],

                lines=30,

                interactive=False,

                elem_id="report-box",
            )

        # -------------------------------------------------
        # ANALYZED IMAGE
        # -------------------------------------------------

        with gr.Column(

            scale=1,

            elem_classes="panel-card",
        ):

            annotated_output = gr.Image(

                label=TEXTS["fa"]["annotated"],

                interactive=False,

                height=600,
            )

    # =====================================================
    # DOWNLOAD FILES
    # =====================================================

    with gr.Row():

        json_output = gr.File(

            label=TEXTS["fa"]["json"]
        )

        csv_output = gr.File(

            label=TEXTS["fa"]["csv"]
        )

    # =====================================================
    # FOOTER
    # =====================================================

    footer = gr.HTML(

        """
        <div class="smart-footer">

            <strong>
                SmartSheet
            </strong>

            • ExamVision V5.1

            <br>

            سیستم هوشمند تحلیل و تصحیح آزمون

        </div>
        """
    )

    # =====================================================
    # LANGUAGE BUTTON EVENT - با حفظ نتایج
    # =====================================================

    language_button.click(

        fn=toggle_language,

        inputs=[

            language_state,

            report_output,

            annotated_output,

            json_output,

            csv_output,

            # داده‌های کش (فعلاً خالی)
            gr.State(None),

        ],

        outputs=[

            # 1
            language_state,

            # 2
            header_html,

            # 3
            description,

            # 4
            input_section,

            # 5
            settings_heading,

            # 6
            result_section,

            # 7
            image_input,

            # 8
            answer_key,

            # 9
            option_count,

            # 10
            columns,

            # 11
            accept_threshold,

            # 12
            review_threshold,

            # 13
            process_button,

            # 14
            report_output,

            # 15
            annotated_output,

            # 16
            json_output,

            # 17
            csv_output,

            # 18
            language_button,

            # 19
            footer,
        ],
    )

    # =====================================================
    # PROCESS BUTTON EVENT
    # =====================================================

    process_button.click(

        fn=process_exam,

        inputs=[

            image_input,

            answer_key,

            option_count,

            columns,

            accept_threshold,

            review_threshold,

            language_state,
        ],

        outputs=[

            report_output,

            annotated_output,

            json_output,

            csv_output,
        ],
    )


# =========================================================
# LAUNCH
# =========================================================

if __name__ == "__main__":

    server_name = os.getenv(

        "GRADIO_SERVER_NAME",

        "127.0.0.1"
    )

    server_port = int(

        os.getenv(

            "GRADIO_SERVER_PORT",

            "7860"
        )
    )

    demo.launch(

        server_name=server_name,

        server_port=server_port,

        css=CUSTOM_CSS,
    )
