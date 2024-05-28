import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_answer_from_text(text, question):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
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
