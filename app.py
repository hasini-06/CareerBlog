import streamlit as st
from agent import generate_roadmap
from PyPDF2 import PdfReader

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

st.title("Career Roadmap Generator")

query = st.text_input("What is your career goal? (e.g., I want to become a Data Analyst)")

uploaded_file = st.file_uploader("Upload a PDF or text file (optional)", type=["pdf", "txt"])

retrieved = []

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
    else:  # PDF
        text = extract_text_from_pdf(uploaded_file)

    # wrap as list of tuples (chunk, score)
    retrieved = [(text, 1.0)]

if st.button("Generate Roadmap"):
    if not query.strip():
        st.warning("Please enter your career goal.")
    else:
        # Pass retrieved only if we have context, otherwise empty list
        roadmap = generate_roadmap(query, retrieved if retrieved else [])
        st.markdown(roadmap)
