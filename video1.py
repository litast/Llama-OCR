import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import pytesseract # Joprojām saglabājam, bet EasyOCR būs galvenais
import pandas as pd
import numpy as np
import os
import easyocr # Jaunā bibliotēka

# --- Tesseract konfigurācija (var atkomentēt, ja vēlaties izmantot EasyOCR vietā) ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Streamlit lapas iestatījumi ---
st.set_page_config(layout="wide", page_title="Cenu Zīmju Analīze no Video")

st.title("Cenu Zīmju Analīze no Video")

st.markdown("""
Šī lietotne analizē augšupielādētu **MP4 video failu**, lai noteiktu cenu zīmes un iegūtu tekstu no tām.
Tiek izmantots **pielāgots YOLOv8 modelis** cenu zīmju noteikšanai un **EasyOCR** teksta atpazīšanai.
Rezultāti tiek apkopoti no secīgiem kadriem, lai uzlabotu precizitāti.
""")

# --- YOLOv8 modeļa ielāde ---
@st.cache_resource
def load_yolo_model(model_path):
    """Ielādē pielāgoto YOLOv8 modeli."""
    try:
        model = YOLO(model_path) # Ielādējam jūsu apmācīto modeli
        return model
    except Exception as e:
        st.error(f"Kļūda ielādējot YOLO modeli no {model_path}: {e}. Pārliecinieties, ka fails eksistē un 'ultralytics' ir pareizi instalēts.")
        return None

# --- EasyOCR lasītāja ielāde ---
@st.cache_resource
def load_easyocr_reader():
    """Ielādē EasyOCR lasītāju."""
    try:
        # Pirmā reize var aizņemt laiku, jo tiek lejupielādēti modeļi.
        # Var norādīt vairākas valodas, piemēram, ['en', 'lv']
        # Iestatiet gpu=True, ja jums ir GPU un vēlaties to izmantot, pretējā gadījumā gpu=False (lēnāk).
        reader = easyocr.Reader(['en', 'lv'], gpu=True) # Varam mēģināt gan angļu, gan latviešu valodā
        return reader
    except Exception as e:
        st.error(f"Kļūda ielādējot EasyOCR lasītāju: {e}. Pārliecinieties, ka 'easyocr' ir pareizi instalēts.")
        return None

# Norādiet ceļu uz jūsu apmācīto YOLO modeli
# Tas būs kaut kas līdzīgs 'runs/detect/price_tag_detector_v1/weights/best.pt'
YOLO_MODEL_PATH = "path/to/your/runs/detect/price_tag_detector_v1/weights/best.pt" # <--- ATJAUNINĀT ŠO CEĻU!

model = load_yolo_model(YOLO_MODEL_PATH)
ocr_reader = load_easyocr_reader()

# --- Failu augšupielāde ---
uploaded_file = st.file_uploader("Augšupielādējiet MP4 video failu", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file) # Parādīt augšupielādēto video

    temp_video_path = "temp_uploaded_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.header("Analīzes Iestatījumi")
    frames_per_second = st.sidebar.slider(
        "Kadrus sekundē apstrādāt (FPS):", 
        min_value=0.5, max_value=10.0, value=2.0, step=0.5,
        help="Norādiet, cik kadru no katras sekundes tiks analizēti. Zemāka vērtība paātrina apstrādi."
    )
    start_second = st.sidebar.number_input(
        "Sākt analīzi no sekundes:", 
        min_value=0, value=0, step=1,
        help="Norādiet video sekundi, no kuras sākt cenu zīmju noteikšanu."
    )
    # Jauns iestatījums rezultātu apkopošanai
    result_buffer_size = st.sidebar.slider(
        "Kadrus apkopot rezultātiem:",
        min_value=1, max_value=20, value=5, step=1,
        help="Cik secīgus kadrus ņemt vērā, lai apkopotu atpazīto tekstu (balsošanas princips). Lielāks skaitlis uzlabo stabilitāti, bet var palēnināt."
    )
    
    st.subheader("Analīzes rezultāti")

    if st.button("Sākt analīzi"):
        if model is None or ocr_reader is None:
            st.error("Nevar turpināt. Modeļi netika ielādēti pareizi. Pārbaudiet kļūdas augstāk.")
            if os.path.exists(temp_video_path): os.remove(temp_video_path)
            st.stop()

        progress_text = st.empty()
        price_tags_data = []
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Nevarēja atvērt video failu. Lūdzu, pārliecinieties, ka tas ir derīgs MP4 fails.")
            if os.path.exists(temp_video_path): os.remove(temp_video_path)
            st.stop()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_skip_interval = int(video_fps / frames_per_second) if frames_per_second > 0 else 1
        if frame_skip_interval < 1: frame_skip_interval = 1

        start_frame = int(start_second * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame_index = start_frame
        
        # Buferis rezultātu apkopošanai
        # Glabās dict: {bounding_box_id: [(frame_index, recognized_text, confidence)]}
        # Vienkāršībai mēs apkopojam tekstu bez atsevišķu cenu zīmju ID izsekošanas starp kadriem
        # Tiks apkopots teksts visām atpazītajām cenu zīmēm katrā kadrā
        recognized_texts_buffer = [] 

        st.info(f"Sāku apstrādāt video. Oriģinālais FPS: {video_fps:.2f}. Apstrādās {frames_per_second:.1f} kadrus/sekundē no {start_second}. sekundes.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if (current_frame_index - start_frame) % frame_skip_interval == 0:
                progress_percentage = int((current_frame_index / total_frames) * 100)
                progress_text.text(f"Apstrādā kadrus: {progress_percentage}% (Kadrs {current_frame_index}/{total_frames})")

                results = model(frame, conf=0.3, iou=0.5) 

                current_frame_price_tags = []
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confidences = r.boxes.conf.cpu().numpy()
                    
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = map(int, boxes[i])
                        confidence = confidences[i]
                        
                        if confidence > 0.5: 
                            price_tag_roi = frame[y1:y2, x1:x2]

                            if price_tag_roi.shape[0] > 0 and price_tag_roi.shape[1] > 0:
                                # Pirms apstrādes uzlabot attēlu (piemēram, pelēktoņi, kontrasts, sliekšņa noteikšana)
                                gray_roi = cv2.cvtColor(price_tag_roi, cv2.COLOR_BGR2GRAY)
                                _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                
                                # Izmantojiet EasyOCR
                                try:
                                    # EasyOCR var atgriezt vairākas teksta rindas, ja tās atpazīst kā atsevišķus apgabalus
                                    easyocr_results = ocr_reader.readtext(thresh_roi)
                                    extracted_text = " ".join([res[1] for res in easyocr_results]).strip()
                                except Exception as e:
                                    extracted_text = f"Kļūda EasyOCR: {e}"

                                current_frame_price_tags.append({
                                    "bbox": (x1, y1, x2, y2),
                                    "confidence": confidence,
                                    "text": extracted_text,
                                    "frame": current_frame_index
                                })
                
                # Pievienot pašreizējā kadra cenu zīmes buferim
                recognized_texts_buffer.append(current_frame_price_tags)
                # Uzturēt bufera izmēru
                if len(recognized_texts_buffer) > result_buffer_size:
                    recognized_texts_buffer.pop(0) # Noņemt vecāko kadru

                # --- Rezultātu apkopošana (vienkāršotā balsošana) ---
                # Šī vienkāršā apkopošana apstrādā visas atpazītās cenu zīmes buferī.
                # Lai precīzāk izsekotu katru individuālo cenu zīmi, būtu nepieciešams izsekot to ID starp kadriem.
                
                # Savāc visus atpazītos tekstus no bufera
                all_texts_in_buffer = [item['text'] for frame_data in recognized_texts_buffer for item in frame_data if item['text'] not in ["", "Teksts netika atpazīts"]]
                
                final_recognized_text = "Teksts netika atpazīts"
                if all_texts_in_buffer:
                    # Atrod visbiežāk sastopamo tekstu buferī
                    from collections import Counter
                    text_counts = Counter(all_texts_in_buffer)
                    final_recognized_text = text_counts.most_common(1)[0][0] # Iegūst visbiežāko tekstu

                # Pievienojam datus galvenajai tabulai (ņemam pēdējā kadra bbox kā reprezentatīvo)
                if current_frame_price_tags: # Ja šajā kadrā kaut kas tika atpazīts
                    # Varam izvēlēties visprecīzāko cenu zīmi no pēdējā kadra, lai pievienotu rindu
                    best_tag_in_frame = max(current_frame_price_tags, key=lambda x: x['confidence'])

                    price_tags_data.append({
                        "Kadrs": current_frame_index,
                        "Laiks (sek.)": f"{(current_frame_index / video_fps):.2f}",
                        "X1": best_tag_in_frame['bbox'][0],
                        "Y1": best_tag_in_frame['bbox'][1],
                        "X2": best_tag_in_frame['bbox'][2],
                        "Y2": best_tag_in_frame['bbox'][3],
                        "Noteikšanas uzticamība": f"{best_tag_in_frame['confidence']:.2f}",
                        "Apkopotais teksts": final_recognized_text 
                    })

            current_frame_index += 1
            
        cap.release()
        
        if price_tags_data:
            df = pd.DataFrame(price_tags_data)
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Lejupielādēt datus kā CSV",
                data=csv,
                file_name="cenu_zimes_ar_apkopošanu.csv",
                mime="text/csv",
            )
        else:
            st.info("Video failā netika atrastas cenu zīmes (vai arī YOLO modelis tās neatpazina ar norādītajiem sliekšņiem norādītajā segmentā).")
        
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

else:
    st.info("Lūdzu, augšupielādējiet MP4 video failu, lai sāktu analīzi.")

st.markdown("""
---
### Rezultātu uzlabošanas ieteikumi:
* **Pielāgots YOLO modelis:** Šis ir obligāts solis, lai precīzi noteiktu cenu zīmes. Jo vairāk un daudzveidīgākus datus apmācībai izmantosiet, jo labāks būs modelis.
* **Precīzāka rezultātu apkopošana:** Lai precīzāk apkopotu rezultātus, būtu nepieciešams ieviest objektu izsekošanas (object tracking) algoritmu (piemēram, DeepSORT). Tas ļautu katrai atpazītajai cenu zīmei piešķirt unikālu ID un tad apkopot tekstu un citus datus par katru konkrētu cenu zīmi visos kadros, kuros tā parādās. Tas ir sarežģītāk, bet sniedz ievērojamu precizitātes pieaugumu.
* **Uzlabota attēlu pirmsapstrāde:** Var veikt specifiskāku attēlu pirmsapstrādi cenu zīmju attēliem, piemēram, perspektīvas korekciju, lai padarītu tekstu vieglāk lasāmu.
""")