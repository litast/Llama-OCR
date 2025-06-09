import streamlit as st 
from PIL import Image, ExifTags
import base64
from groq import Groq
import pandas as pd
from datetime import datetime
import re

# Konfigurācija
st.set_page_config(
    page_title="Llama OCR - teksta izvilkšana",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

default_prompt = """Analizē attēlu un izvelc strukturētu informāciju par visiem redzamajiem produktiem.

**Izgūstamie lauki**:
- Groza prece = 1 / Cita prece = 0
- Kategorija (Memorands) — viens no: Maize, Piens, Piena produkti, Dārzeņi (svaigi), Augļi, ogas (svaigas), Gaļa, Zivis (svaigas), Milti, graudaugi, Olas, Eļļa (augu)
- Grupa (Memorands) — piem.: 01.1.1.3. Maize, 01.1.4.2. Piens, 01.1.4.5. Siers un biezpiens, 01.1.5.2. Sviests, utt.
- Veids (Memorands) — piem.: Baltmaize, Rupjmaize, Piens (pasterizēts), Siers
- Preces nosaukums, info (veikalā)
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

**Rezultātu attēlo vienā horizontālā Markdown tabulā**:
- Katra **rinda** ir viens produkts.
- Katra **kolonna** ir viens no iepriekš minētajiem lauku nosaukumiem, tieši šādā secībā.
- Nenorādi neko tādu, kas nav redzams vai pilnībā saprotams.
- Ja informācija nav zināma, ievieto `-`.
- Cenas pieraksti ar komatu, nevis ar punktu (piemēram, `2,99`).
- Norādot pārdošanas cenu, nelieto valūtas simbolus (piemēram, € vai EUR).
"""

# Palīgfunkcijas
def extract_datetime_from_metadata(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        exif = {
            ExifTags.TAGS.get(k, k): v
            for k, v in image.getexif().items()
        }
        if "DateTimeOriginal" in exif:
            dt_str = exif["DateTimeOriginal"]
            dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            return dt.date(), dt.time(), exif.get("Compression", None)
        else:
            return None, None, exif.get("Compression", None)
    except:
        return None, None, None

def extract_datetime_from_filename(filename):
    match = re.search(r"(\d{8})[_-](\d{6})", filename)
    if match:
        date_str, time_str = match.groups()
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        return dt.date(), dt.time()
    return None, None

# UI
st.title("🦙 Teksta izvilkšana no attēliem")
st.markdown("Ar **llama-4-scout-17b-16e-instruct** palīdzību izvelk produktus no attēliem un parāda salīdzināmā tabulā.")

# Sānu josla: uzstādījumi
with st.sidebar:
    st.header("📋 Uzstādījumi")

    st.subheader("⚙️ Informācijas ieguve")
    use_metadata = st.checkbox("Automātiski iegūt datumu un laiku no faila")

    if not use_metadata:
        date_value = st.date_input("Datums", value=datetime.today(), format="DD.MM.YYYY")
        time_value = st.time_input("Laiks")

    employee = st.text_input("Darbinieks (obligāti)")

    st.subheader("🏪 Tirdzniecības vieta")
    merchant = st.selectbox("Tirgotājs", ["Maxima", "Rimi", "Lidl", "Top!", "Elvi"])
    city = st.selectbox("Pilsēta", ["Rīga", "Daugavpils", "Liepāja", "Jelgava", "Valmiera"])
    store_address = st.text_input("Veikala adrese (obligāti)")

    st.header("📷 Augšupielādēt attēlus")
    uploaded_files = st.file_uploader("Izvēlies attēlus...", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    process = st.button("Izvilkt tekstu 🔍", type="primary")

# Uzvedne
st.subheader("📝 Uzvedne")
with st.expander("Rediģēt uzvedni pirms apstrādes:", expanded=False):
    custom_prompt = st.text_area("Uzvedne:", value=default_prompt, height=400)

# Notīrīšanas poga
col_btn1, col_btn2 = st.columns([6, 1])
with col_btn2:
    if st.button("Notīrīt 🗑️"):
        if 'ocr_table_rows' in st.session_state:
            del st.session_state['ocr_table_rows']
        st.rerun()

# Datu apstrāde
if process:
    if not employee.strip():
        st.warning("⚠️ Lūdzu, ievadi darbinieka vārdu!")
    elif not store_address.strip():
        st.warning("⚠️ Lūdzu, ievadi veikala adresi!")
    elif not uploaded_files:
        st.warning("⚠️ Lūdzu, augšupielādē vismaz vienu attēlu!")
    else:
        st.session_state['ocr_table_rows'] = []
        client = Groq(api_key=st.secrets["API_KEY"])

        for uploaded_file in uploaded_files:
            with st.spinner(f"Apstrādā: {uploaded_file.name}"):
                try:
                    image_bytes = uploaded_file.getvalue()
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")

                    if use_metadata:
                        date_val, time_val, compression = extract_datetime_from_metadata(uploaded_file)
                        if (compression == 6 or compression == 'Lossy') or (date_val is None and time_val is None):
                            date_val, time_val = extract_datetime_from_filename(uploaded_file.name)
                        if date_val is None or time_val is None:
                            date_val = datetime.today().date()
                            time_val = datetime.today().time()
                    else:
                        date_val = date_value
                        time_val = time_value

                    response = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": custom_prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }
                        ]
                    )

                    content = response.choices[0].message.content
                    lines = [line.strip() for line in content.strip().splitlines() if line.strip().startswith("|")]
                    if len(lines) >= 3:
                        header = [h.strip().replace("**", "") for h in lines[0].strip("|").split("|")]
                        for data_line in lines[2:]:
                            values = [v.strip() for v in data_line.strip("|").split("|")]
                            if len(values) == len(header):
                                row = {
                                    "Fails": uploaded_file.name,
                                    "Datums": date_val.strftime("%d.%m.%Y"),
                                    "Laiks": time_val.strftime("%H:%M"),
                                    "Darbinieks": employee,
                                    "Tirgotājs": merchant,
                                    "Pilsēta": city,
                                    "Veikala adrese": store_address
                                }
                                for h, v in zip(header, values):
                                    row[h] = v
                                if "Produkta veids" in row:
                                    row["Produkta veids"] = row["Produkta veids"].capitalize()
                                st.session_state['ocr_table_rows'].append(row)
                    else:
                        st.warning(f"Nepietiekama tabulas struktūra failā: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Neizdevās apstrādāt '{uploaded_file.name}': {e}")

# Tabula un CSV
if 'ocr_table_rows' in st.session_state and st.session_state['ocr_table_rows']:
    df_all = pd.DataFrame(st.session_state['ocr_table_rows'])

    double_header_map = {}
    for col in df_all.columns:
        if col in ["Datums", "Laiks", "Darbinieks"]:
            double_header_map[col] = "Pārbaude"
        elif col in ["Tirgotājs", "Pilsēta", "Veikala adrese", "Fails"]:
            double_header_map[col] = ""
        else:
            double_header_map[col] = "Prece"

    multi_columns = pd.MultiIndex.from_tuples(
        [(double_header_map[col], col) for col in df_all.columns]
    )
    df_all.columns = multi_columns

    st.subheader("📊 Strukturēti produktu dati no visiem attēliem")
    st.dataframe(df_all)

    flat_df = df_all.copy()
    flat_df.columns = [col[1] for col in flat_df.columns]
    csv = flat_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Lejupielādēt CSV", data=csv, file_name="produktu_tabula.csv", mime="text/csv")
else:
    st.markdown(
        "<p style='color: #333; background-color: #e9ecef; padding: 10px; border-radius: 5px;'>"
        "Augšupielādē attēlus un spied <strong>Izvilkt tekstu</strong>, lai iegūtu tabulu ar produktiem."
        "</p>", unsafe_allow_html=True
    )
