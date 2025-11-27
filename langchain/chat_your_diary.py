import os
import json
import streamlit as st
from datetime import datetime
import spacy

from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# ===============================
# Load SpaCy Model
# ===============================
nlp = spacy.load("en_core_web_md")

# ===============================
# Prepare diary file
# ===============================
if not os.path.exists("diary.json"):
    with open("diary.json", "w", encoding="utf-8") as f:
        json.dump({}, f)

st.title("Chat with Your Diary 📝")


# ===============================
# DIARY ENTRY FORM
# ===============================
today = datetime.now().date()
selected_date = st.date_input("Select a date", value=today)

with st.form(key="form_diary", clear_on_submit=True):
    note = st.text_area("Write your diary:", height=250)
    submit = st.form_submit_button("Save")

if submit and note:
    date_str = str(selected_date)

    with open("diary.json", "r", encoding="utf-8") as f:
        diary_data = json.load(f)

    # Append new entry or create new date
    if date_str in diary_data:
        if note not in diary_data[date_str]:
            diary_data[date_str] += "\n\n" + note
    else:
        diary_data[date_str] = note

    with open("diary.json", "w", encoding="utf-8") as f:
        json.dump(diary_data, f, ensure_ascii=False, indent=2)

    st.success("Diary saved!")


# ===============================
# QUESTION INPUT
# ===============================
question = st.text_input("Enter your question:")
ask = st.button("ASK")

if ask and question:
    with open("diary.json", "r", encoding="utf-8") as f:
        diary_data = json.load(f)

    if not diary_data:
        st.warning("Your diary is empty! Write something first.")
        st.stop()

    # ===============================
    # Find the most relevant diary entries
    # ===============================
    similarities = []
    question_vec = nlp(question)

    for date_str, diary_text in diary_data.items():
        diary_vec = nlp(diary_text)
        score = question_vec.similarity(diary_vec)
        similarities.append((score, f"Date: {date_str}\nDiary:\n{diary_text}"))

    # Select top 3 most relevant diary entries
    top_entries = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]

    st.subheader("Most Relevant Diary Entrie(s):")
    combined_text = ""

    for score, entry in top_entries:
        st.markdown(f"**Similarity:** {score:.2f}<br>{entry}", unsafe_allow_html=True)
        combined_text += entry + "\n\n"

    # ===============================
    # LLM SECTION
    # ===============================
    llm = OllamaLLM(model="llama3.1:8b")

    template = """
You are a helpful assistant. Use ONLY the diary text below to answer the user's question.

QUESTION:
{question}

DIARY CONTENT:
{text}

ANSWER:
"""

    prompt = PromptTemplate(
        input_variables=["question", "text"],
        template=template
    )

    final_prompt = prompt.format(question=question, text=combined_text)

    # Invoke the model
    answer = llm.invoke(final_prompt)

    st.subheader("Answer:")
    st.markdown(answer)
