import streamlit as st 
from PIL import Image
import base64
from groq import Groq
import pandas as pd

# Konfigurācija
st.set_page_config(
    page_title="Llama OCR - teksta izvilkšana",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stils
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
.stMarkdown, .stText { color: #000000 !important; }
.element-container div.stMarkdown p { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# Virsraksts
st.title("🦙 OCR - teksta izvilkšana no attēliem")
st.markdown("Izvelk strukturētu tekstu no attēliem, piemēram, cenu zīmēm, un apvieno rezultātus salīdzināmā tabulā.")

# Notīrīt
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Notīrīt 🗑️"):
        st.session_state.clear()
        st.experimental_rerun()

# Sānu josla: failu augšupielāde
with st.sidebar:
    st.header("Augšupielādēt attēlus")
    uploaded_files = st.file_uploader("Izvēlies attēlus...", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if uploaded_files and st.button("Izvilkt tekstu 🔍", type="primary"):
        st.session_state['ocr_table_rows'] = []

        
        client = Groq(api_key=st.secrets["API_KEY"])

        for uploaded_file in uploaded_files:
            with st.spinner(f"Apstrādā: {uploaded_file.name}"):
                try:
                    image_bytes = uploaded_file.getvalue()
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")

                    response = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"""Analizē attēlu '{uploaded_file.name}' un izvelc strukturētu informāciju par visiem redzamajiem produktiem.
                                     Izgūsti: produkta veids, produkta nosaukums, produkta vienība, ražotājs (ja ir), pārdošanas cena, cena par vienību (kg/l/ml), atlaide (ja ir),
                                     cena pirms atlaides, valsts (ja ir), svītrkods (ja ir). Rezultātu attēlo Markdown tabulā ar vienotu kolonnu nosaukumu katrā rindā.
                                     Cenai par vienību jābūt skaidri norādītai kā cena par kg, l vai ml. Vienība ir jānorāda kā kg, l vai ml, atkarībā no produkta.
                                     Pārdošanas cena ir cena, kas tiek maksāta par produktu, un tā var būt ar atlaidi vai bez tās. Valūtu pie pārdošanas cenas nenorādi.
                                     Svītrkods ir skaitļu virkne, kas sākas ar 0 vai 4, un ir 8 vai 13 cipari garš. Ja informācija nav pieejama, ievelc svītriņu, lai nepaliek tukši lauki.
                                     Cena izsākāma ar punktu nevis ar komatu."""},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }
                        ]
                    )

                    content = response.choices[0].message.content

                    # Apstrādā Markdown tabulu
                    lines = [line.strip() for line in content.strip().splitlines() if line.strip().startswith("|")]
                    if len(lines) >= 3:
                        # Tīra Markdown header no '**' un citiem simboliem
                        header = [h.strip().replace("**", "") for h in lines[0].strip("|").split("|")]

                        for data_line in lines[2:]:  # Sākam no 3. rindas (0 = header, 1 = ---)
                            values = [v.strip() for v in data_line.strip("|").split("|")]
                            if len(values) == len(header):
                                row = {"Fails": uploaded_file.name}
                                for h, v in zip(header, values):
                                    row[h] = v

                                # Sākam "Produkta veids" ar lielo burtu, ja tāds lauks ir
                                if "Produkta veids" in row:
                                    row["Produkta veids"] = row["Produkta veids"].capitalize()

                                st.session_state['ocr_table_rows'].append(row)
                    else:
                        st.warning(f"Nepietiekama tabulas struktūra failā: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Neizdevās apstrādāt '{uploaded_file.name}': {e}")

# Rezultātu tabula
if 'ocr_table_rows' in st.session_state and st.session_state['ocr_table_rows']:
    df_all = pd.DataFrame(st.session_state['ocr_table_rows'])
    st.subheader("📊 Strukturēti produktu dati no visiem attēliem")
    st.dataframe(df_all)

    csv = df_all.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Lejupielādēt CSV", data=csv, file_name="produktu_tabula.csv", mime="text/csv")
else:
    st.markdown(
        "<p style='color: #333; background-color: #e9ecef; padding: 10px; border-radius: 5px;'>"
        "Augšupielādē attēlus un spied 'Izvilkt tekstu', lai iegūtu tabulu ar produktiem."
        "</p>", unsafe_allow_html=True
    )
