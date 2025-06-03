import streamlit as st
import os
import io
import cv2
import re
import base64
import pandas as pd
from datetime import timedelta
from google.cloud import vision
from PIL import Image
from groq import Groq  # pip install groq

# ======= 1. Streamlit konfigurācija =======
st.set_page_config(page_title="Video OCR ar Google Vision + LLM", layout="wide")
st.title("🎥 OCR no video ar Google Cloud Vision un LLaMA")
st.markdown("Augšupielādē video. Tiks analizēti kadri ik pa X sekundēm un informācija izvadīta strukturētā tabulā.")

# ======= 2. Lietotāja iestatījumi =======
output_interval = st.number_input("Kadru intervāls (sekundēs)", min_value=1, max_value=30, value=2)
start_second = st.number_input("Sākt apstrādi no (sekundes)", min_value=0, max_value=3600, value=0)
CREDENTIALS_JSON = "llamaocr-c8c9f2801d4f.json"
FRAMES_DIR = "frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

# ======= 3. API inicializācija =======
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_JSON
vision_client = vision.ImageAnnotatorClient()
llm_client = Groq(api_key=st.secrets["API_KEY"])

# ======= 4. Noklusējuma prompta teksts =======
default_prompt = """Analizē kadrus un izvelc strukturētu informāciju par redzamajiem produktiem.

Izgūstamie lauki:
- Produkta veids
- Produkta nosaukums 
- Produkta vienība (%/kg/l/ml)
- Ražotājs (ja ir)
- Pārdošanas cena
- Cena par vienību
- Atlaide (%) (ja ir)
- Cena pirms atlaides
- Pārdošanas cena (ar lojalitātes karti) (ja ir)
- Cena par vienību (ar lojalitātes karti) (ja ir)
- Valsts (ja ir)
- Svītrkods (8 vai 13 cipari, sākas ar 0 vai 4)

**Ja informācija par produktu atkārtojas vairākos kadros, tad apvieno tos vienā rindā.**

**Rezultātu attēlo vienā horizontālā Markdown tabulā**:
- Katra **rinda** ir viens produkts.
- Katra **kolonna** ir viens no iepriekš minētajiem lauku nosaukumiem, tieši šādā secībā.
- Nenorādi neko tādu, kas nav redzams vai pilnībā saprotams.
- Ja informācija nav zināma, ievieto `-`.
- Cenas pieraksti ar komatu, nevis ar punktu (piemēram, `2,99`).
- Norādot pārdošanas cenu, nelieto valūtas simbolus (piemēram, € vai EUR).
"""

# ======= 5. Video apstrāde =======
uploaded_video = st.file_uploader("⬆️ Augšupielādē .mp4 video", type=["mp4"])

if uploaded_video and st.button("Sākt apstrādi"):
    with st.spinner("Video tiek apstrādāts..."):
        video_path = "video_temp.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        video = cv2.VideoCapture(video_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps * output_interval)
        start_frame = int(fps * start_second)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame
        saved_frames = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = str(timedelta(seconds=int(frame_count / fps)))
                frame_filename = f"{FRAMES_DIR}/frame_{frame_count}_{timestamp.replace(':', '-')}.jpg"
                cv2.imwrite(frame_filename, frame)
                saved_frames.append((frame_filename, timestamp))
            frame_count += 1
        video.release()

        # ======= 6. OCR + LLM analīze =======
        structured_results = []
        for frame_path, timestamp in saved_frames:
            with open(frame_path, 'rb') as img_file:
                image_bytes = img_file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            try:
                response = llm_client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": default_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ]
                )

                content = response.choices[0].message.content

                # Markdown tabulas apstrāde
                lines = [line.strip() for line in content.strip().splitlines() if line.strip().startswith("|")]
                if len(lines) >= 3:
                    header = [h.strip().replace("**", "") for h in lines[0].strip("|").split("|")]
                    for data_line in lines[2:]:
                        values = [v.strip() for v in data_line.strip("|").split("|")]
                        if len(values) == len(header):
                            row = {"Kadrs": os.path.basename(frame_path), "Laiks": timestamp}
                            for h, v in zip(header, values):
                                row[h] = v
                            structured_results.append(row)
                else:
                    st.warning(f"Nepietiekama tabula kadram: {os.path.basename(frame_path)}")
            except Exception as e:
                st.error(f"Kļūda analizējot kadru '{frame_path}': {e}")

        # ======= 7. Rezultātu attēlošana =======
        if structured_results:
            df_structured = pd.DataFrame(structured_results)
            st.success(f"Apstrādāti {len(df_structured)} kadri ar LLM tabulas analīzi.")
            st.dataframe(df_structured)
            csv = df_structured.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Lejupielādēt strukturēto CSV", data=csv, file_name="strukturēts_ocr.csv", mime="text/csv")
        else:
            st.warning("Netika iegūta neviena strukturēta tabula no kadriem.")
