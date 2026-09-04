import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Blood Work Analyzer",
    page_icon="🩸",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .app-title {
            font-size: 2.2rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            color: #6b7280;
            margin-bottom: 1.4rem;
        }
        .section-label {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        [data-testid="stFileUploader"] {
            padding-top: 0.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">🩸 Blood Work Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload a blood report to extract lab values and generate a simple health summary with an Indian diet plan.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LLM setup — follows the attached notebook
# ---------------------------------------------------------
@st.cache_resource
def get_llm():
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    return ChatGoogleGenerativeAI(model=model_name)


def response_text(response) -> str:
    """Handle LangChain AIMessage variants safely."""
    if hasattr(response, "text") and response.text:
        return response.text
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)
    return str(response)


# ---------------------------------------------------------
# Report reading
# ---------------------------------------------------------
def read_uploaded_report(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages).strip()

    raise ValueError("Please upload a PDF or TXT blood report.")


# ---------------------------------------------------------
# Analysis stages — adapted directly from the notebook
# ---------------------------------------------------------
def extract_blood_values(llm, blood_report: str) -> str:
    extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract all test values and classify each one as High, Low, or Normal
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Do not invent a reference range if the report does not provide one. If a status cannot be determined
from the report, mark it as UNKNOWN.

Blood Report:
{blood_report}
"""

    extraction_response = llm.invoke(extraction_prompt)
    return response_text(extraction_response)


def generate_health_and_diet(llm, extracted_values: str) -> str:
    diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based only on the blood work analysis below, write:
1. A short health summary in 3 lines explaining the patient's condition in simple language.
2. A short, practical Indian diet plan having only two sections:
   (1) Foods to avoid
   (2) Foods to eat more of
Do not include any other sections in the diet plan.

Blood Work Analysis:
{extracted_values}
"""

    diet_response = llm.invoke(diet_prompt)
    return response_text(diet_response)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "extracted_values" not in st.session_state:
    st.session_state.extracted_values = ""
if "diet_values" not in st.session_state:
    st.session_state.diet_values = ""
if "report_text" not in st.session_state:
    st.session_state.report_text = ""


# ---------------------------------------------------------
# Layout: report on left, two result panels on right
# ---------------------------------------------------------
left_col, right_col = st.columns([0.9, 2.1], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown('<div class="section-label">1. Blood Report</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload report",
            type=["pdf", "txt"],
            label_visibility="collapsed",
        )

        try:
            uploaded_text = read_uploaded_report(uploaded_file)
        except Exception as exc:
            uploaded_text = ""
            st.error(f"Could not read the report: {exc}")

        # Allows either file upload or manual paste.
        default_text = uploaded_text if uploaded_text else st.session_state.report_text
        blood_report = st.text_area(
            "Report text",
            value=default_text,
            height=440,
            placeholder="Upload a PDF/TXT report or paste the blood report here...",
        )

        analyze = st.button(
            "Analyze Blood Report",
            type="primary",
            use_container_width=True,
        )

        if st.button("Clear", use_container_width=True):
            st.session_state.extracted_values = ""
            st.session_state.diet_values = ""
            st.session_state.report_text = ""
            st.rerun()

        st.caption("This tool summarizes report information and is not a medical diagnosis.")


with right_col:
    result_col_1, result_col_2 = st.columns(2, gap="medium")

    with result_col_1:
        with st.container(border=True):
            st.markdown('<div class="section-label">2. Lab Value Analysis</div>', unsafe_allow_html=True)
            result_placeholder_1 = st.empty()
            if st.session_state.extracted_values:
                result_placeholder_1.markdown(st.session_state.extracted_values)
            else:
                result_placeholder_1.info("Extracted test values and HIGH / LOW / NORMAL status will appear here.")

    with result_col_2:
        with st.container(border=True):
            st.markdown('<div class="section-label">3. Health Summary & Diet</div>', unsafe_allow_html=True)
            result_placeholder_2 = st.empty()
            if st.session_state.diet_values:
                result_placeholder_2.markdown(st.session_state.diet_values)
            else:
                result_placeholder_2.info("The 3-line health summary and Indian diet recommendations will appear here.")


# ---------------------------------------------------------
# Run analysis after UI has been drawn
# ---------------------------------------------------------
if analyze:
    if not blood_report.strip():
        st.error("Please upload or paste a blood report first.")
    else:
        try:
            st.session_state.report_text = blood_report
            llm = get_llm()

            with result_col_1:
                with st.spinner("Extracting blood values..."):
                    extracted_values = extract_blood_values(llm, blood_report)
                    st.session_state.extracted_values = extracted_values
                    result_placeholder_1.markdown(extracted_values)

            with result_col_2:
                with st.spinner("Preparing health summary and diet plan..."):
                    diet_values = generate_health_and_diet(llm, extracted_values)
                    st.session_state.diet_values = diet_values
                    result_placeholder_2.markdown(diet_values)

        except Exception as exc:
            st.error(
                "Analysis failed. Check that your Google Gemini API key is configured "
                f"and that the selected model is available. Details: {exc}"
            )
