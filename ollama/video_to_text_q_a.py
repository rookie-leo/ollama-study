import tempfile
import subprocess
import whisper
import streamlit as st
import ollama
import os

FFMPEG = os.getenv("FFMPEG_PATH")
if not FFMPEG:
    raise RuntimeError("Erro: variável de ambiente FFMPEG_PATH não definida no PyCharm.")

os.environ["FFMPEG_BINARY"] = FFMPEG
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG)

model = whisper.load_model("base")


def extract_audio(uploaded_file):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    temp_video.close()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    command = [
        FFMPEG, "-y",
        "-i", temp_video.name,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        temp_wav
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        st.error("Erro no FFmpeg:")
        st.code(result.stderr.decode())
        return None

    return temp_wav


def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]


st.title("Video → Texto → Q&A")

uploaded_file = st.file_uploader("Envie seu vídeo .mp4", type="mp4")

if uploaded_file:
    audio_path = extract_audio(uploaded_file)

    if audio_path:
        transcription = transcribe_audio(audio_path)

        st.subheader("Transcrição")
        st.write(transcription)

        question = st.text_area("Pergunta baseada no vídeo:")

        if st.button("Perguntar"):
            prompt = f"Conteúdo do vídeo:\n{transcription}\n\nPergunta: {question}"

            response = ollama.generate(
                model="llama3.1:8b",
                prompt=prompt
            )

            st.subheader("Resposta")
            st.write(response["response"])
