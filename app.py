# app.py

import streamlit as st
import fitz
from transformers import pipeline

st.set_page_config(
    page_title="Study Assistant AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Study Assistant AI")

st.markdown(
    "Upload your PDF and get a smart summary instantly."
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )
    return summarizer

summarizer = load_model()

# -----------------------------
# Extract Text From PDF
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""

    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")

    for page in pdf_document:
        text += page.get_text()

    return text

# -----------------------------
# Split Long Text
# -----------------------------
def split_text(text, max_chunk=1000):
    chunks = []

    for i in range(0, len(text), max_chunk):
        chunks.append(text[i:i + max_chunk])

    return chunks

# -----------------------------
# Generate Summary
# -----------------------------
def generate_summary(text):

    chunks = split_text(text)

    summaries = []

    for chunk in chunks[:5]:

        summary = summarizer(
            chunk,
            max_length=120,
            min_length=40,
            do_sample=False
        )

        summaries.append(summary[0]["summary_text"])

    final_summary = "\n\n".join(summaries)

    return final_summary

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload PDF File",
    type=["pdf"]
)

if uploaded_file is not None:

    with st.spinner("Reading PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)

    st.success("PDF Uploaded Successfully ✅")

    st.subheader("📄 Extracted Text Preview")

    st.text_area(
        "Preview",
        extracted_text[:3000],
        height=250
    )

    if st.button("Generate Summary"):

        with st.spinner("Generating Summary..."):

            summary = generate_summary(extracted_text)

        st.subheader("🧠 AI Summary")

        st.write(summary)

        st.download_button(
            "Download Summary",
            summary,
            file_name="summary.txt"
        )