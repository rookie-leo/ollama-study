import os
import tempfile
import time

import fitz # pip install PyMuPDF
import ollama
import streamlit as st


def save_temp_pdf(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return temp_path, temp_dir

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text

st.title("PDF Inspector")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_path, temp_dir = save_temp_pdf(uploaded_file)
    start_time = time.perf_counter()
    pdf_text = extract_text_from_pdf(pdf_path)

    prompt = st.text_area("Faça uma pergunta baseada no conteudo do PDF.")
    button = st.button("Perguntar")

    if button:
        if prompt:
            combined_prompt = (
                'Analise o conteudo do PDF:\n\n'
                f'{pdf_text}\n\n'
                f'Pergunta: {prompt}'
            )
            response = ollama.generate(model='llama3.1:8b', prompt=combined_prompt)

            end_time = time.perf_counter()
            total_time = end_time - start_time

            st.subheader('Resposta')
            st.markdown(response['response'])

            st.success(f'Tempo de processamento: **{total_time:.2f} segundos**')

    st.info(f'Arquivo temporario salvo em {temp_dir}\n Será deletado automaticamente')
