# pip install opencv-python
import tempfile
import time

import cv2
import ollama
import streamlit as st

def video_to_frames(video_path, extract_every_seconds=2):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * extract_every_seconds)

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            temp_frame = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_frame.name, frame)
            frames.append(temp_frame.name)

        frame_count += 1

    cap.release()
    return frames


def save_temp_video(uploaded_file):
    """Salva o vídeo enviado pelo usuário em arquivo temporário."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
        temp.write(uploaded_file.read())
        return temp.name


st.title("Video Describer")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "webm"])

if uploaded_file is not None:
    start_time = time.perf_counter()
    st.info("Processando vídeo...")

    video_path = save_temp_video(uploaded_file)
    frames = video_to_frames(video_path, extract_every_seconds=2)

    if not frames:
        st.error("Nenhum frame pôde ser extraído.")
        st.stop()

    description = ""

    st.info(f"{len(frames)} frames extraídos. Enviando para análise...")

    for frame_path in frames:
        response = ollama.chat(model="llava:7b", messages=[{
            "role": "user",
            "content": "Descreva a imagem em uma frase curta.",
            "images": [frame_path]
        }])

        description += f"{response['message']['content']}\n"

    prompt = (
        "Com base nas seguintes descrições dos frames do vídeo, "
        "escreva um resumo geral do que está acontecendo:\n\n"
        f"{description}"
    )

    answer = ollama.generate(model="llama3.1:8b", prompt=prompt)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    st.subheader("Descrição do vídeo")
    st.markdown(answer["response"])
    st.success(f'Tempo total de processamento: {total_time:.2f} segundos')
