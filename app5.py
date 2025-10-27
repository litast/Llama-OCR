import streamlit as st
from PIL import Image, ExifTags
import base64
from groq import Groq
import pandas as pd
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import unicodedata
import os

## Šī ir app3.py bet cita kolonnu secība **

# Konfigurācija
st.set_page_config(
    page_title="Llama OCR - teksta izvilkšana v5",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

default_prompt = """Analizē cenu zīmes attēlā un izvelc strukturētu informāciju par produktiem.

**Katram produktam izgūstamie lauki**:
(Obligāti ievēro tieši šādu lauku secību un nosaukumus).
- Groza prece (vinmēr tukšs lauks).
- Kategorija (Memorands) — izvēlies no saraksta, ja prece tam atbilst: `Maize`, `Piens`, `Piena produkti`, `Dārzeņi (svaigi)`, `Augļi, ogas (svaigas)`, `Gaļa`, `Zivis (svaigas)`, `Milti, graudaugi`, `Olas`, `Eļļa (augu)`; ja nē — atstāj tukšu lauku.
- Grupa (Memorands) — izvēlies no saraksta, ja prece tam atbilst: `01.1.1.3. Maize`, `01.1.4.2. Piens`, `01.1.4.5. Siers un biezpiens`, `01.1.5.2. Sviests`, `01.1.4.6. Krējums`, `01.1.4.4. Jogurts`, `01.1.4.5. Svaigi dārzeņi`, `01.1.7.4. Kartupeļi`, `01.1.6.1. Svaigi augļi`, `01.1.2.2. Cūkgaļa`, `01.1.2.4. Mājputnu gaļa`, `01.1.2.1. Liellopu un teļa gaļa`, `01.1.2.3. Aitu un kazu gaļa`, `01.1.3.1. Svaiga zivis`, `01.1.1.2. Milti, citi graudaugi`, `01.1.4.7. Olas`, `01.1.5.  Augu eļļa`; ja nē — atstāj tukšu lauku.
- Veids (Memorands) — izvēlies no saraksta, ja prece tam atbilst: Baltmaize, Rupjmaize, Piens (pasterizēts), Siers, Biezpiens, Sviests, Krējums, Jogurts, Kefīrs, Paniņas, Sīpoli, BurkāniĶiploki, Bietes, Tomāti, Gurķi, Galviņkāposti, Ziedkāposti, Lapu salāti, Ķirbji, Kabači, Kartupeļi, Āboli, Bumbieri, Zemenes, Dzērvenes, Brūklenes, Krūmmellenes, Jāņogas, Upenes, Avenes, Cūkgaļa, Cūkgaļa - malta, Mājputnu gaļa, Mājputnu gaļa (malta), Liellopu gaļa, Teļa gaļa, Aitu gaļa, Kazu gaļa, Zivis - svaigas, Zivis - atdzesētas, Kviešu milti, Pilngraudu milti, Griķi, Vistu olas, Olīveļļa, Rapšu eļļa, Saulespuķu eļļa; ja nē — atstāj tukšu lauku.
- Preces nosaukums, info (veikalā) (arī ražotāja nosaukumu, ja ir).
- Ražotāja valsts (ja ir).
- Cena.
- Cena ar atlaidi (ja ir).
- Mērvienība (Grami, Kg, Litrs, Mililitri, Gab.) - norādi mērvienību, kas norādīta produkta nosaukumā.
- Produkta vienība, piemēram, 0.5l ir 0.5.
- Cena par vienību.
- Cena ar klienta karti (ja ir).
- Cena par vienību ar klienta karti (ja ir).
- Grozs: tukšs lauks.
- Groza redzamība: tukšs lauks.
- Preces pieejamība veikalā: tukšs lauks.
- Piezīmes.
- Svītrkods (EAN-13 formātā, bez punktiem un atstarpēm).
- Mērvienība par vienību (€/L, €/Kg, €/Gab.).

**Rezultātu attēlo vienā horizontālā Markdown tabulā**:
- Nenorādi nekādas kolonnas ārpus šī saraksta.
- Katra **rinda** ir viens produkts.
- Katra **kolonna** ir viens no iepriekš minētajiem laukiem, tieši šādā secībā.
- Nenorādi neko tādu, kas nav skaidri redzams vai pilnībā saprotams. Ja tas tā ir, atstāj tukšu lauku.
- Ja prece atrodas starp memoranda grupām, tad Groza prece ir `1`. Visas pārējās preces ir `0`.
- Norādot cenas, nelieto valūtas simbolus (piemēram, € vai EUR).
- Cena: norādi standarta cenu pirms akcijas atlaides, bez lojalitātes kartes.
- Atlaide: norādi tikai, ja ir norādīts cenas samazinājums procentos (%).
- Ja cenu zīmē ir norādīta cena ar `Mans Rimi karti`, `Paldies karti` vai citu lojalitātes karti, ievieto to `Cena ar klienta karti` un 'Cena par vienību ar klienta karti' laukos.
- Svītrkods parasti ir izvietots zem vai pa labi no stabiņveida līnijām.
- Piezīmēs norādi būtisku informāciju, kas varētu būt noderīga, piemēram, ja ir norādīta - atlaide (%), lojalitātes kartes nosaukumu.
"""

# Palīgfunkcijas
def extract_datetime_from_metadata(uploaded_file):
    try:
        # Pārliecināmies, ka faila sākums tiek iestatīts uz 0, lai to pareizi nolasītu
        uploaded_file.seek(0)

        image = Image.open(uploaded_file)

        # Pārbaudām, vai attēlam ir EXIF dati
        if image.getexif() is None:
            return None, None, None, None, None
        
        exif = {
            ExifTags.TAGS.get(k, k): v for k, v in image.getexif().items()
        }

        # Apple/iPhone un citi bieži izmanto "DateTimeOriginal"
        dt_str = exif.get("DateTimeOriginal")

        # Ja nav DateTimeOriginal, mēģinām izmantot "DateTime" (modifikācijas datums)
        if dt_str is None:
            dt_str = exif.get("DateTime")
        
        dt = None
        if dt_str:
            try:
                # Standarta EXIF datuma/laika formāts
                dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                # Ja formāts ir cits, piemēram, nestandarta, var veikt papildu mēģinājumus šeit
                pass

        date_val = dt.date() if dt else None
        time_val = dt.time() if dt else None
            
        # Iegūstam papildu metadatus
        compression = exif.get("Compression", None)
        make = exif.get("Make", None)        # Ražotājs (piemēram, Apple)
        model = exif.get("Model", None)      # Modelis (piemēram, iPhone 15 Pro Max)
            
        # Atgriežam datumu, laiku, kompresiju, ražotāju un modeli
        return date_val, time_val, compression, make, model
    
    except Exception as e:
        st.warning(f"EXIF kļūda: {e}")
        return None, None, None, None, None

def extract_datetime_from_filename(filename):
    match = re.search(r"(\d{8})[_-](\d{6})", filename)
    if match:
        date_str, time_str = match.groups()
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        return dt.date(), dt.time()
    return None, None

def process_image(uploaded_file, use_metadata, custom_prompt, employee, merchant, city, store_address, date_value, time_value, client):
    try:
        image_bytes = uploaded_file.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Iegūt datumu un laiku no EXIF, faila nosaukuma vai ievades
        if use_metadata:
            date_val, time_val, compression, make, model = extract_datetime_from_metadata(uploaded_file) # Izsaukums labots!
            
            if date_val is None or time_val is None:
                date_val_fname, time_val_fname = extract_datetime_from_filename(uploaded_file.name)

                if date_val is None:
                    date_val = date_val_fname
                if time_val is None:
                    time_val = time_val_fname
        else:
            date_val = date_value
            time_val = time_value
            # Ja netiek izmantoti metadati, make un model ir None
            make = None
            model = None
        
        # Formatējam datumu un laiku kā tekstu vai tukšu, ja nav
        date_str = date_val.strftime("%d.%m.%Y") if date_val else ""
        time_str = time_val.strftime("%H:%M") if time_val else ""

        # Groq vaicājums
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": custom_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ]
        )

        content = response.choices[0].message.content
        lines = [line.strip() for line in content.strip().splitlines() if line.strip().startswith("|")]
        extracted_rows = []

        if len(lines) >= 3:
            header = [h.strip().replace("**", "") for h in lines[0].strip("|").split("|")]
            for data_line in lines[2:]:
                values = [v.strip() for v in data_line.strip("|").split("|")]
                if len(values) == len(header):
                    row = {
                        "Fails": uploaded_file.name,
                        "Datums": date_str,
                        "Laiks": time_str,
                        "Darbinieks": employee,
                        "Tirgotājs": merchant,
                        "Pilsēta": city,
                        "Veikals (nosaukums vai adrese)": store_address
                    }
                    for h, v in zip(header, values):
                        row[h] = v
                    extracted_rows.append(row)
        return extracted_rows

    except Exception as e:
        return [{"Fails": uploaded_file.name, "Kļūda": str(e)}]

# Noņem garumzīmes no darbinieka vārda
def remove_diacritics(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

# UI
st.title("🖼️ Teksta izvilkšana no attēliem")
st.markdown("Ar **llama-4-scout-17b-16e-instruct** palīdzību izvelk produktus no attēliem un parāda salīdzināmā tabulā.")

# Sānu josla
with st.sidebar:
    st.header("📋 Uzstādījumi")
    employee = st.text_input("👤 Darbinieks (obligāti)", placeholder="Norādi darbinieka vārdu un uzvārdu")
    use_metadata = st.checkbox("Izgūt datumu un laiku no faila EXIF vai nosaukuma", value=True)

    if not use_metadata:
        date_value = st.date_input("📅 Datums", value=datetime.today(), format="DD.MM.YYYY")
        time_value = st.time_input("⏰ Laiks")
    else:
        date_value = None
        time_value = None

    merchant = st.selectbox("🏪 Tirgotājs", ["MAXIMA", "RIMI", "LIDL", "TOP", "ELVI", "DEPO", "NARVESEN"])
    city = st.selectbox("🌆 Pilsēta", ["Rīga", "Piņķi", "Daugavpils", "Liepāja", "Jelgava", "Jaunolaine", "Olaine", "Valmiera"])
    store_address = st.text_input("📍 Veikala adrese (obligāti)", placeholder="Norādi veikala adresi vai nosaukumu")
    uploaded_files = st.file_uploader("🖼️ Izvēlies attēlus", type=None, accept_multiple_files=True)
    process = st.button("Izvilkt tekstu 🔍", type="primary")

# Uzvedne
st.subheader("📝 Uzvedne")
with st.expander("Rediģēt uzvedni pirms apstrādes", expanded=False):
    custom_prompt = st.text_area("Uzvedne:", value=default_prompt, height=400)

# Apstrāde bez ThreadPoolExecutor
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

        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        processed_files = 0

        for f in uploaded_files:
            result = process_image(
                f, use_metadata, custom_prompt, employee,
                merchant, city, store_address, date_value, time_value, client
            )
            for row in result:
                if "Kļūda" in row:
                    st.error(f"{row['Fails']}: {row['Kļūda']}")
                else:
                    st.session_state['ocr_table_rows'].append(row)

            processed_files += 1
            progress_percent = int((processed_files / total_files) * 100)
            progress_bar.progress(progress_percent)

        if progress_bar:
            progress_bar.empty()

# Notīrīšanas poga
col_btn1, col_btn2 = st.columns([6, 1])
with col_btn2:
    if st.button("Notīrīt 🗑️"):
        if 'ocr_table_rows' in st.session_state:
            del st.session_state['ocr_table_rows']
        st.rerun()

# Tabula
if 'ocr_table_rows' in st.session_state and st.session_state['ocr_table_rows']:
    df_all = pd.DataFrame(st.session_state['ocr_table_rows'])

    # Decimālatdalītāja aizvietošana cenām
    cena_kolonnas = [
        "Cena", "Cena ar atlaidi", "Cena par vienību",
        "Cena ar klienta karti", "Cena par vienību ar klienta karti"
    ]
    for kol in cena_kolonnas:
        if kol in df_all.columns:
            df_all[kol] = df_all[kol].astype(str).str.replace(',', '.', regex=False)

    # Izveidojot dubulto virsrakstu (multi-index)
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

    st.subheader("📊 Strukturēti dati")
    st.dataframe(df_all)

    flat_df = df_all.copy()
    flat_df.columns = [col[1] for col in flat_df.columns]
    
    # Izveido faila nosaukumu
    today_str = datetime.today().strftime("%Y%m%d")
    clean_employee = remove_diacritics(employee).replace(" ", "_")
    base_filename = f"{today_str}_{clean_employee}_produktu_tabula"

    # CSV lejupielāde
    csv = flat_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Lejupielādēt CSV", data=csv, file_name=f"{base_filename}.csv", mime="text/csv")

    # Excel lejupielāde
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        flat_df.to_excel(writer, index=False, sheet_name="Produkti")
    excel_data = excel_buffer.getvalue()

    st.download_button(
        label="⬇️ Lejupielādēt Excel",
        data=excel_data,
        file_name=f"{base_filename}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.markdown(
        "<p style='color: #333; background-color: #e9ecef; padding: 10px; border-radius: 5px;'>"
        "Augšupielādē attēlus un spied <strong>Izvilkt tekstu</strong>, lai iegūtu tabulu ar produktiem."
        "</p>", unsafe_allow_html=True
    )
