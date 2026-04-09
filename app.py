from __future__ import annotations

import html
import os
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

from utils.evaluation import (
    PerformanceReport,
    evaluate_dataset,
    has_labeled_dataset_structure,
    load_metrics_report,
)
from utils.gradcam import generate_gradcam
from utils.predictor import (
    DEFAULT_LABEL_MODE,
    DEFAULT_PREPROCESS_MODE,
    DEFAULT_THRESHOLD,
    get_model_input_size,
    predict_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "EfficientNetV2S_mouth_cancer.keras"
DEFAULT_METRICS_PATHS = (
    PROJECT_ROOT / "reports" / "training_report.json",
    PROJECT_ROOT / "evaluation" / "metrics.json",
    PROJECT_ROOT / "reports" / "metrics.json",
)
DEFAULT_DATASET_PATHS = (
    PROJECT_ROOT / "evaluation",
    PROJECT_ROOT / "eval",
    PROJECT_ROOT / "data" / "eval",
    PROJECT_ROOT / "data" / "test",
    PROJECT_ROOT / "dataset" / "test",
)
SERVING_PREPROCESS_MODE = DEFAULT_PREPROCESS_MODE
SERVING_LABEL_MODE = DEFAULT_LABEL_MODE
SERVING_UNCERTAINTY_MARGIN = 0.0


st.set_page_config(
    page_title="Neural Oral Cancer Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6efe4;
            --surface: rgba(255, 252, 248, 0.82);
            --surface-strong: rgba(255, 255, 255, 0.92);
            --ink: #1f2d2f;
            --muted: #5d6b6c;
            --accent: #0f766e;
            --accent-soft: rgba(15, 118, 110, 0.12);
            --sand: #d97706;
            --sand-soft: rgba(217, 119, 6, 0.12);
            --alert: #b42318;
            --alert-soft: rgba(180, 35, 24, 0.12);
            --line: rgba(31, 45, 47, 0.08);
            --shadow: 0 24px 70px rgba(74, 55, 38, 0.12);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 0% 0%, rgba(217, 119, 6, 0.16), transparent 30%),
                radial-gradient(circle at 100% 0%, rgba(15, 118, 110, 0.18), transparent 28%),
                linear-gradient(180deg, #f6efe4 0%, #f1e8dc 100%);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(217, 119, 6, 0.12), transparent 30%),
                linear-gradient(180deg, #18363a 0%, #102628 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #eef4f2 !important;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            font-family: "Georgia", "Times New Roman", serif;
            letter-spacing: -0.02em;
            color: var(--ink);
        }

        p, li, span, label, [data-testid="stMarkdownContainer"] {
            font-family: "Trebuchet MS", "Segoe UI Variable", sans-serif;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2rem 2rem 1.65rem;
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.45);
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(252, 246, 239, 0.78));
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            inset: auto -40px -60px auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(15, 118, 110, 0.18), transparent 68%);
        }

        .eyebrow {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.55rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.5rem;
            line-height: 1.02;
        }

        .hero p {
            max-width: 720px;
            margin: 0.85rem 0 0;
            font-size: 1.03rem;
            line-height: 1.65;
            color: var(--muted);
        }

        .note-strip {
            margin-top: 1.1rem;
            display: inline-flex;
            gap: 0.5rem;
            align-items: center;
            padding: 0.55rem 0.85rem;
            border-radius: 999px;
            background: rgba(31, 45, 47, 0.06);
            color: var(--ink);
            font-size: 0.9rem;
        }

        .metric-card {
            min-height: 148px;
            border-radius: 24px;
            border: 1px solid var(--line);
            background: var(--surface);
            box-shadow: var(--shadow);
            padding: 1rem 1rem 0.95rem;
            backdrop-filter: blur(10px);
        }

        .metric-card strong {
            display: block;
            font-size: 0.86rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.7rem;
        }

        .metric-card .metric-value {
            display: block;
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 2rem;
            line-height: 1;
            color: var(--ink);
            margin-bottom: 0.45rem;
        }

        .metric-card .metric-caption {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .panel-title {
            margin-top: 1.4rem;
            margin-bottom: 0.25rem;
            font-size: 1.45rem;
        }

        .subtle {
            color: var(--muted);
            font-size: 0.97rem;
            line-height: 1.55;
        }

        .summary-shell {
            background: var(--surface-strong);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.25rem 1.25rem 1.1rem;
            box-shadow: var(--shadow);
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }

        .status-pill.alert {
            background: var(--alert-soft);
            color: var(--alert);
        }

        .status-pill.safe {
            background: var(--accent-soft);
            color: var(--accent);
        }

        .result-title {
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 2rem;
            line-height: 1.08;
            margin-bottom: 0.3rem;
        }

        .result-copy {
            color: var(--muted);
            line-height: 1.65;
            font-size: 0.98rem;
            margin-bottom: 1rem;
        }

        .score-block {
            margin-top: 0.85rem;
        }

        .score-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.35rem;
            font-size: 0.95rem;
            color: var(--ink);
        }

        .score-rail {
            width: 100%;
            height: 11px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(31, 45, 47, 0.08);
        }

        .score-fill {
            height: 100%;
            border-radius: 999px;
        }

        .score-fill.alert {
            background: linear-gradient(90deg, #dc6a4a, #b42318);
        }

        .score-fill.safe {
            background: linear-gradient(90deg, #24a69a, #0f766e);
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .mini-card {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.8);
            padding: 0.85rem 0.9rem;
        }

        .mini-card .label {
            display: block;
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .mini-card .value {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--ink);
        }

        .legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 0.75rem;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .legend span {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }

        .swatch {
            width: 14px;
            height: 14px;
            border-radius: 999px;
            display: inline-block;
        }

        .swatch.warm {
            background: linear-gradient(135deg, #ffcf70, #d33d17);
        }

        .swatch.cool {
            background: linear-gradient(135deg, #81b9ff, #334d8f);
        }

        .side-panel {
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.05);
            padding: 0.95rem 1rem;
            margin-bottom: 0.9rem;
        }

        .side-panel .label {
            display: block;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
            opacity: 0.8;
        }

        .side-panel .value {
            font-size: 1rem;
            line-height: 1.55;
        }

        .side-panel .big {
            font-size: 1.6rem;
            font-weight: 700;
            font-family: "Georgia", "Times New Roman", serif;
        }

        .confusion-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.6rem;
            margin-top: 0.7rem;
        }

        .confusion-cell {
            padding: 0.75rem 0.8rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .confusion-cell strong {
            display: block;
            font-size: 1.05rem;
            margin-top: 0.2rem;
        }

        div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.7);
            border: 1px dashed rgba(15, 118, 110, 0.35);
            border-radius: 26px;
            padding: 1rem;
        }

        .stButton > button {
            border: none;
            border-radius: 999px;
            background: linear-gradient(135deg, #0f766e, #155e75);
            color: white;
            font-weight: 700;
            padding: 0.65rem 1.1rem;
            box-shadow: 0 16px 30px rgba(21, 94, 117, 0.28);
        }

        @media (max-width: 900px) {
            .hero {
                padding: 1.4rem;
            }

            .hero h1 {
                font-size: 2rem;
            }

            .mini-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_setting(name: str) -> str | None:
    try:
        if name in st.secrets:
            value = st.secrets[name]
            if value is not None:
                return str(value)
    except Exception:
        pass

    value = os.getenv(name)
    return value if value else None


def resolve_path(value: str | None, fallback: Path | None = None) -> Path | None:
    if value is None:
        return fallback

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def discover_metrics_report_path() -> Path | None:
    configured = resolve_path(get_setting("METRICS_REPORT_PATH"))
    if configured is not None:
        return configured

    for candidate in DEFAULT_METRICS_PATHS:
        if candidate.exists():
            return candidate

    return None


def discover_dataset_path() -> Path | None:
    configured = resolve_path(get_setting("EVAL_DATASET_DIR"))
    if configured is not None:
        return configured

    for candidate in DEFAULT_DATASET_PATHS:
        if candidate.exists() and has_labeled_dataset_structure(candidate):
            return candidate

    return None


@st.cache_resource(show_spinner=False)
def load_cancer_model(model_path: str):
    return load_model(model_path, compile=False)


def format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def get_serving_threshold(report: PerformanceReport | None) -> float:
    if report is not None and report.threshold is not None:
        return float(report.threshold)
    return DEFAULT_THRESHOLD


def metric_card(title: str, value: str, caption: str) -> str:
    return f"""
    <div class="metric-card">
        <strong>{html.escape(title)}</strong>
        <span class="metric-value">{html.escape(value)}</span>
        <div class="metric-caption">{html.escape(caption)}</div>
    </div>
    """


def render_top_cards(
    model_path: Path,
    input_size: int,
    report: PerformanceReport | None,
) -> None:
    accuracy_value = format_percent(report.accuracy) if report else "Unavailable"
    accuracy_caption = (
        f"Measured on {report.total_samples} labeled images."
        if report and report.total_samples is not None
        else "Add a labeled evaluation set or metrics JSON to display it."
    )

    cards = st.columns(3, gap="large")
    with cards[0]:
        st.markdown(
            metric_card(
                "Model",
                model_path.stem.replace("_", " "),
                "EfficientNetV2S classifier loaded from the local workspace.",
            ),
            unsafe_allow_html=True,
        )
    with cards[1]:
        st.markdown(
            metric_card(
                "Input Profile",
                f"{input_size} x {input_size}",
                "RGB mouth image, single-image screening inference.",
            ),
            unsafe_allow_html=True,
        )
    with cards[2]:
        st.markdown(
            metric_card("Accuracy", accuracy_value, accuracy_caption),
            unsafe_allow_html=True,
        )


def render_sidebar(
    model_path: Path,
    input_size: int,
    model,
) -> tuple[PerformanceReport | None, float]:
    report: PerformanceReport | None = None
    report_path = discover_metrics_report_path()
    dataset_path = discover_dataset_path()

    if report_path is not None and report_path.exists():
        try:
            report = load_metrics_report(report_path)
        except Exception as exc:
            st.sidebar.warning(f"Could not read metrics report: {exc}")

    if "live_evaluation_report" in st.session_state:
        try:
            report = PerformanceReport(**st.session_state["live_evaluation_report"])
        except TypeError:
            st.session_state.pop("live_evaluation_report", None)

    threshold = get_serving_threshold(report)

    with st.sidebar:
        st.markdown("## Project Console")
        st.caption(
            "Confidence here is a model score, not a medical certainty. Treat it as screening support only."
        )

        st.markdown(
            f"""
            <div class="side-panel">
                <span class="label">Loaded Model</span>
                <div class="value"><strong>{html.escape(model_path.name)}</strong></div>
                <div class="value">Input size: {input_size} x {input_size}</div>
                <div class="value">Validated threshold: {threshold:.2f}</div>
                <div class="value">Pipeline: legacy normalized RGB</div>
                <div class="value">Serving class rule: lower score means cancer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Accuracy")
        if report is not None and report.accuracy is not None:
            source_text = report.source.replace("_", " ").title()
            sample_text = (
                f"{report.total_samples} samples"
                if report.total_samples is not None
                else "sample count unavailable"
            )
            st.markdown(
                f"""
                <div class="side-panel">
                    <span class="label">Current Report</span>
                    <div class="big">{format_percent(report.accuracy)}</div>
                    <div class="value">{html.escape(source_text)} - {html.escape(sample_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Precision", format_percent(report.precision))
            with metric_cols[1]:
                st.metric("Recall", format_percent(report.recall))

            if report.specificity is not None or report.f1_score is not None:
                metric_cols = st.columns(2)
                with metric_cols[0]:
                    st.metric("Specificity", format_percent(report.specificity))
                with metric_cols[1]:
                    st.metric("F1 Score", format_percent(report.f1_score))

            if all(
                value is not None
                for value in (report.tp, report.tn, report.fp, report.fn)
            ):
                st.markdown(
                    f"""
                    <div class="confusion-grid">
                        <div class="confusion-cell">True Positive<strong>{report.tp}</strong></div>
                        <div class="confusion-cell">True Negative<strong>{report.tn}</strong></div>
                        <div class="confusion-cell">False Positive<strong>{report.fp}</strong></div>
                        <div class="confusion-cell">False Negative<strong>{report.fn}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if report.notes:
                st.caption(report.notes)

        else:
            st.info(
                "No trustworthy accuracy number is stored in this workspace yet. "
                "Add `METRICS_REPORT_PATH` or a labeled `EVAL_DATASET_DIR` to measure it."
            )

        if dataset_path is not None:
            if dataset_path.exists() and has_labeled_dataset_structure(dataset_path):
                button_label = "Refresh Local Accuracy" if report is not None else "Run Local Accuracy"
                if st.button(button_label, use_container_width=True):
                    try:
                        with st.spinner(f"Evaluating labeled images in {dataset_path.name}..."):
                            live_report = evaluate_dataset(
                                model=model,
                                dataset_dir=dataset_path,
                                image_size=input_size,
                                threshold=threshold,
                                preprocess_mode=SERVING_PREPROCESS_MODE,
                                label_mode=SERVING_LABEL_MODE,
                                uncertainty_margin=SERVING_UNCERTAINTY_MARGIN,
                            )
                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")
                    else:
                        st.session_state["live_evaluation_report"] = asdict(live_report)
                        st.rerun()
                st.caption(f"Dataset path: {dataset_path}")
            elif get_setting("EVAL_DATASET_DIR"):
                st.warning(
                    "Configured evaluation dataset was found, but it does not match the expected "
                    "`cancer/` and `non_cancer/` style folder structure."
                )

        st.markdown("### Usage Guidance")
        st.markdown(
            "- Use sharp, well-lit mouth images.\n"
            "- The displayed accuracy is measured on the local holdout dataset, not on a single image.\n"
            "- Heatmaps show model attention, not proof of disease.\n"
            "- Any positive result still needs clinician review."
        )

    return report, threshold


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">Screening Workspace</div>
            <h1>Neural Oral Cancer Detection</h1>
            <p>
                Upload a mouth image to generate a binary screening prediction, view class-level
                confidence, and inspect a Grad-CAM attention map that shows what regions influenced
                the model most strongly.
            </p>
            <div class="note-strip">Educational and research use only. This interface is not a clinical diagnosis system.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_score_bar(label: str, value: float, tone: str) -> None:
    percentage = max(0.0, min(100.0, value * 100.0))
    st.markdown(
        f"""
        <div class="score-block">
            <div class="score-row">
                <span>{html.escape(label)}</span>
                <strong>{percentage:.1f}%</strong>
            </div>
            <div class="score-rail">
                <div class="score-fill {html.escape(tone)}" style="width: {percentage:.1f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_summary(prediction) -> None:
    if prediction.is_uncertain:
        pill_class = "alert"
        summary_title = "Borderline result"
        copy = (
            "The model score landed close to the decision threshold, so the app is marking this image "
            "for review instead of forcing a confident yes/no label."
        )
    else:
        pill_class = "alert" if prediction.is_cancer else "safe"
        summary_title = (
            "Escalate for review" if prediction.is_cancer else "Lower-risk screening signal"
        )
        copy = (
            "The model leaned toward cancer-associated visual features in this image. "
            "Treat this as a flag for professional review."
            if prediction.is_cancer
            else "The model leaned toward the non-cancer class for this image, but the result still "
            "should not replace clinician assessment."
        )

    st.markdown(
        f"""
        <div class="summary-shell">
            <span class="status-pill {pill_class}">{html.escape(prediction.label)}</span>
            <div class="result-title">{html.escape(summary_title)}</div>
            <div class="result-copy">{html.escape(copy)}</div>
            <div class="mini-grid">
                <div class="mini-card">
                    <span class="label">Model Confidence</span>
                    <span class="value">{prediction.confidence * 100:.1f}%</span>
                </div>
                <div class="mini-card">
                    <span class="label">Decision Boundary</span>
                    <span class="value">{prediction.threshold:.2f}</span>
                </div>
                <div class="mini-card">
                    <span class="label">Raw Model Score</span>
                    <span class="value">{prediction.raw_score:.4f}</span>
                </div>
                <div class="mini-card">
                    <span class="label">Interpretation</span>
                    <span class="value">{html.escape(prediction.interpretation.replace('_', ' ').title())}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_score_bar("Cancer signal", prediction.cancer_probability, "alert")
    render_score_bar("No-cancer signal", prediction.non_cancer_probability, "safe")
    st.caption(
        "This project's trained model uses a threshold where lower raw scores map to the cancer class and higher scores map to the no-cancer class."
    )


def render_accuracy_panel(report: PerformanceReport | None) -> None:
    st.markdown("### Accuracy Snapshot")
    if report is None or report.accuracy is None:
        st.info(
            "Accuracy cannot be stated honestly from this workspace yet because there is no labeled evaluation set or saved metrics report available."
        )
        return

    cols = st.columns(4)
    metrics = [
        ("Accuracy", report.accuracy),
        ("Precision", report.precision),
        ("Recall", report.recall),
        ("Specificity", report.specificity),
    ]
    for column, (label, value) in zip(cols, metrics):
        with column:
            st.markdown(
                metric_card(
                    label,
                    format_percent(value),
                    "Measured from labeled evaluation data.",
                ),
                unsafe_allow_html=True,
            )

    if report.total_samples is not None:
        st.caption(
            f"Evaluation coverage: {report.total_samples} labeled images"
            + (
                f" with {report.skipped_samples} skipped unreadable files."
                if report.skipped_samples
                else "."
            )
        )


def main() -> None:
    inject_styles()

    model_path = resolve_path(get_setting("MODEL_PATH"), DEFAULT_MODEL_PATH)
    if model_path is None:
        st.error("Model path is not configured.")
        return

    render_hero()

    try:
        model = load_cancer_model(str(model_path))
    except Exception as exc:
        st.error(f"Unable to load the Keras model from `{model_path}`.")
        st.exception(exc)
        return

    input_size = get_model_input_size(model)
    report, threshold = render_sidebar(
        model_path,
        input_size,
        model,
    )
    render_top_cards(model_path, input_size, report)

    st.markdown("## Upload & Screen", unsafe_allow_html=False)
    st.markdown(
        '<p class="subtle">JPG, JPEG, and PNG files work best. This app now uses one fixed, validated pipeline so the result stays consistent.</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a mouth image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear oral photograph for the most stable result.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="summary-shell">
                <div class="result-title">Ready for an image</div>
                <div class="result-copy">
                    Once you upload a mouth image, the app will show the predicted class, both class
                    probabilities, a Grad-CAM attention map, and any available evaluation metrics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_accuracy_panel(report)
        return

    file_bytes = np.frombuffer(uploaded_file.getbuffer(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("This file could not be decoded as a valid image. Please try another JPG or PNG.")
        return

    try:
        prediction = predict_image(
            model=model,
            img_bgr=img_bgr,
            image_size=input_size,
            threshold=threshold,
            preprocess_mode=SERVING_PREPROCESS_MODE,
            label_mode=SERVING_LABEL_MODE,
            uncertainty_margin=SERVING_UNCERTAINTY_MARGIN,
        )
    except Exception as exc:
        st.error("Prediction failed while processing the uploaded image.")
        st.exception(exc)
        return

    summary_col, image_col = st.columns([1.15, 0.85], gap="large")
    with summary_col:
        render_prediction_summary(prediction)
    with image_col:
        st.image(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            caption="Uploaded image",
            use_container_width=True,
        )

    attention_tab, interpretation_tab, performance_tab = st.tabs(
        ["Attention Map", "Interpretation", "Accuracy"]
    )

    with attention_tab:
        with st.spinner("Generating Grad-CAM attention map..."):
            try:
                cam = generate_gradcam(
                    model,
                    img_bgr,
                    focus_on=prediction.focus_class,
                    image_size=input_size,
                    preprocess_mode=SERVING_PREPROCESS_MODE,
                )
            except Exception as exc:
                st.error("Grad-CAM generation failed for this image.")
                st.exception(exc)
                cam = None

        if cam is not None:
            heatmap_col, notes_col = st.columns([1.15, 0.85], gap="large")
            with heatmap_col:
                st.image(
                    cv2.cvtColor(cam, cv2.COLOR_BGR2RGB),
                    caption="Grad-CAM overlay",
                    use_container_width=True,
                )
            with notes_col:
                st.markdown("### How to read the map")
                st.markdown(
                    """
                    Hotter colors suggest the image regions that most influenced the predicted class.
                    Cooler colors had less influence on the model's decision for this specific image.
                    """
                )
                st.markdown(
                    """
                    <div class="legend">
                        <span><i class="swatch warm"></i> Higher model attention</span>
                        <span><i class="swatch cool"></i> Lower model attention</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with interpretation_tab:
        st.markdown("### Model setup")
        st.markdown(
            f"- The app uses the bundled `EfficientNetV2S_mouth_cancer.keras` model for screening.\n"
            f"- The serving threshold is fixed at `{threshold:.2f}` because that was the best value on the labeled holdout split.\n"
            "- Lower raw scores map to the cancer class, and higher raw scores map to the no-cancer class.\n"
            "- The fine-tuned checkpoint was not promoted because it scored worse than the bundled model on the holdout split.\n"
            "- Grad-CAM explains the predicted class for the uploaded image."
        )
        st.markdown("### Practical caution")
        st.markdown(
            "- Confidence is not clinical certainty.\n"
            "- A positive screen should be treated as a review signal, not a diagnosis.\n"
            "- A negative screen does not rule out disease."
        )

    with performance_tab:
        render_accuracy_panel(report)
        if report is not None and report.notes:
            st.markdown("### Evaluation notes")
            st.info(report.notes)
        elif report is None:
            st.markdown("### How to enable real accuracy reporting")
            st.markdown(
                "- Set `EVAL_DATASET_DIR` to a folder containing labeled subfolders such as `cancer/` and `non_cancer/`.\n"
                "- Or set `METRICS_REPORT_PATH` to a JSON file with saved evaluation metrics.\n"
                "- Then use the sidebar to display or recompute the metrics."
            )


if __name__ == "__main__":
    main()
