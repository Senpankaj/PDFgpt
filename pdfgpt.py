import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
import os

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def summarize_text(text, summary_length="detailed"):
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
    max_chunk_length = 1024  # Max input length for the model

    # Split text into chunks
    text_chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    # Summarize each chunk
    summaries = []
    for chunk in text_chunks:
        if summary_length == "detailed":
            summary = summarization_pipeline(chunk, max_length=512, min_length=100, do_sample=False)
        else:
            summary = summarization_pipeline(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    # Join all summaries together
    return " ".join(summaries)

def get_answer_from_text(text, question):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    answer = qa_pipeline(question=question, context=text)
    return answer['answer'].strip()

# Streamlit app
st.title("Personal ChatGPT for PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    extracted_text = extract_text_from_pdf("temp.pdf")
    
    st.header("Summaries")
    detailed_summary = summarize_text(extracted_text, summary_length="detailed")
    brief_summary = summarize_text(extracted_text, summary_length="brief")
    
    st.subheader("Detailed Summary")
    st.write(detailed_summary)
    
    st.subheader("Brief Summary")
    st.write(brief_summary)
    
    st.header("Ask Questions")
    question = st.text_input("Enter your question")
    if question:
        answer = get_answer_from_text(extracted_text, question)
        st.write(f"**Answer:** {answer}")

    st.header("Ask another question")
    question = st.text_input("Enter another question", key="another_question")
    if question:
        answer = get_answer_from_text(extracted_text, question)
        st.write(f"**Answer:** {answer}")
