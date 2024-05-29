import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import torch

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
    return text

def get_answer_from_text(text, question):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)
    answer = qa_pipeline(question=question, context=text)
    return answer['answer'].strip()

# Streamlit app
st.title("Personal PDF ChatGPT")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    extracted_text = extract_text_from_pdf("temp.pdf")
    
    st.header("Ask Questions About Your PDF")
    question = st.text_input("Enter your question")
    if question:
        answer = get_answer_from_text(extracted_text, question)
        st.write(f"**Answer:** {answer}")

    st.header("Ask Another Question")
    question = st.text_input("Enter another question", key="another_question")
    if question:
        answer = get_answer_from_text(extracted_text, question)
        st.write(f"**Answer:** {answer}")
