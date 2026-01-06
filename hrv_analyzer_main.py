import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="HRV Engineer Dashboard", layout="wide", page_icon="🫀")

# Titolo e Header
st.title("🫀 HRV Engineer Dashboard")
st.markdown("### Monitoraggio Ingegneristico - Caricamento Massivo")

# --- NOME DEL DATABASE ---
DB_FILE = 'hrv_database.csv'

# --- FUNZIONI DI SERVIZIO (BACKEND) ---

def parse_rr_file(file_content):
    """
    Legge il contenuto raw del file.
    Estrae gli intervalli RR e calcola rMSSD e RHR.
    """
    try:
        content = file_content.decode("utf-8").splitlines()
        rr_intervals = []
        
        for line in content:
            line = line.strip()
            if line.isdigit():
                val = int(line)
                # Filtro fisiologico (300ms - 2000ms)
                if 300 < val < 2000: 
                    rr_intervals.append(val)
        
        if len(rr_intervals) < 10:
            return None, None

        # --- CALCOLI MATEMATICI ---
        rr_array = np.array(rr_intervals)
        
        # 1. rMSSD
        diffs = np.diff(rr_array)
        squared_diffs = np.square(diffs)
        rmssd = np.sqrt(np.mean(squared_diffs))
        
        # 2. RHR
        mean_rr = np.mean(rr_array)
        rhr = 60000 / mean_rr
        
        return round(rmssd, 2), round(rhr, 1)
        
    except Exception as e:
        return None, None

def extract_date_from_filename(filename):
    """
    Estrae la data dal nome del file formato 'YYYY-MM-DD HH-MM-SS.txt'.
    """
    try:
        # Rimuove l'estensione (.txt o .csv)
        name_clean = os.path.splitext(filename)[0]
        # Converte la stringa in oggetto datetime
        timestamp = datetime.strptime(name_clean, "%Y-%m-%d %H-%M-%S")
        return timestamp
    except ValueError:
        return None

def load_db():
    """Carica il database CSV esistente o ne crea uno vuoto."""
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, parse_dates=['Date'])
    else:
        return pd.DataFrame(columns=['Date', 'rMSSD', 'RHR', 'Sleep', 'Feel', 'Status'])

def get_traffic_light(rmssd, rhr, sleep, feel, df_history):
    """
    Algoritmo decisionale basato su media mobile 7gg.
    """
    if df_history.empty or len(df_history) < 3:
        return "⚪ DATI INSUFFICIENTI (Start)"
    
    # Calcolo Baseline (Media ultimi 7 record inseriti PRIMA di questo)
    last_7 = df_history.tail(7)
    base_rmssd = last_7['rMSSD'].mean()
    base_rhr = last_7['RHR'].mean()
    
    # Logica IF-THEN
    if rmssd < base_rmssd * 0.85 and rhr > base_rhr * 1.05:
        return "🔴 RIPOSO (Crash)"
    elif rmssd < base_rmssd * 0.90 or rhr > base_rhr * 1.03 or feel < 6:
        return "🟡 CAUTELA"
    elif rmssd > base_rmssd * 1.30 and sleep < 7:
        return "🟡 PARADOSSO"
    else:
        return "🟢 GO"

def process_batch_upload(uploaded_files, default_sleep, default_feel):
    """
    Gestisce il caricamento di più file, li ordina e aggiorna il DB.
    """
    df_current = load_db()
    
    new_entries = []
    
    # Barra di progresso
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        # 1. Estrazione Data dal Nome File
        file_date = extract_date_from_filename(uploaded_file.name)
        
        if file_date is None:
            st.error(f"❌ Nome file non valido: {uploaded_file.name}. Deve essere 'YYYY-MM-DD HH-MM-SS'.")
            continue
            
        # 2. Parsing Metriche
        rmssd, rhr = parse_rr_file(uploaded_file.getvalue())
        
        if rmssd is not None:
            new_entries.append({
                'Date': file_date,
                'rMSSD': rmssd,
                'RHR': rhr,
                'Sleep': default_sleep, # Valori di default per importazione massiva
                'Feel': default_feel,
                'Status': 'Da Calcolare'
            })
        else:
            st.warning(f"⚠️ File vuoto o corrotto: {uploaded_file.name}")
        
        progress_bar.progress((i + 1) / total_files)

    # Se abbiamo nuovi dati
    if new_entries:
        df_new = pd.DataFrame(new_entries)
        
        # 3. Ordinamento Fondamentale: Ordiniamo i nuovi file per data
        df_new = df_new.sort_values(by='Date')

        # 4. Calcolo dello Status Iterativo
        # Dobbiamo calcolare lo status riga per riga simulando il passare del tempo
        # Uniamo temporaneamente al vecchio DB per avere lo storico
        
        final_rows_to_add = []
        
        # Creiamo una copia di lavoro dello storico attuale
        temp_history = df_current.copy()
        
        for index, row in df_new.iterrows():
            # Controlliamo duplicati
            if not temp_history.empty:
                # Se esiste già una data uguale (giorno preciso), saltiamo o sovrascriviamo?
                # Qui saltiamo per sicurezza se la data esatta esiste già
                if row['Date'] in temp_history['Date'].values:
                    continue
            
            # Calcola status usando la storia accumulata fino a quel momento
            status = get_traffic_light(
                row['rMSSD'], 
                row['RHR'], 
                row['Sleep'], 
                row['Feel'], 
                temp_history
            )
            
            row['Status'] = status
            final_rows_to_add.append(row)
            
            # Aggiungiamo questa riga alla history per il calcolo della prossima iterazione
            temp_history = pd.concat([temp_history, row.to_frame().T], ignore_index=True)

        # 5. Salvataggio Finale
        if final_rows_to_add:
            df_final_add = pd.DataFrame(final_rows_to_add)
            df_updated = pd.concat([df_current, df_final_add], ignore_index=True)
            df_updated = df_updated.sort_values(by='Date')
            # Rimuove duplicati esatti se presenti
            df_updated = df_updated.drop_duplicates(subset=['Date'], keep='last')
            
            df_updated.to_csv(DB_FILE, index=False)
            st.success(f"✅ Importati correttamente {len(final_rows_to_add)} nuovi file!")
            return df_updated
        else:
            st.info("Nessun dato nuovo da aggiungere (forse duplicati?).")
            return df_current
    
    return df_current

# --- INTERFACCIA UTENTE (SIDEBAR) ---

with st.sidebar:
    st.header("📂 Importazione Dati")
    
    # Widget per file multipli
    uploaded_files = st.file_uploader(
        "Carica i file .txt (Nomi: YYYY-MM-DD HH-MM-SS)", 
        type=['txt'], 
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.markdown("**Impostazioni per importazione massiva**")
    st.caption("Poiché stai caricando dati passati, inserisci valori medi per questi parametri:")
    
    default_sleep = st.number_input("Sonno Default (h)", 4.0, 12.0, 7.5, 0.5)
    default_feel = st.slider("Feel Default (1-10)", 1, 10, 7)
    
    if uploaded_files:
        if st.button("🚀 Elabora File Caricati"):
            df = process_batch_upload(uploaded_files, default_sleep, default_feel)
            st.balloons()

# --- DASHBOARD PRINCIPALE ---

df = load_db()

if not df.empty:
    # Mostra l'ultima lettura in grande (KPI)
    last_entry = df.iloc[-1]
    
    st.subheader(f"📅 Ultimo Dato: {last_entry['Date'].strftime('%d/%m/%Y %H:%M')}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("rMSSD", f"{last_entry['rMSSD']} ms")
    col2.metric("RHR", f"{last_entry['RHR']} bpm")
    col3.metric("Sonno", f"{last_entry['Sleep']} h")
    col4.metric("Status", last_entry['Status'])

    status_msg = last_entry['Status']
    if "🟢" in status_msg:
        st.success(f"## {status_msg}")
    elif "🟡" in status_msg:
        st.warning(f"## {status_msg}")
    elif "🔴" in status_msg:
        st.error(f"## {status_msg}")

    st.divider()

    # --- GRAFICI DEI TREND ---
    st.subheader("📈 Analisi Storica")
    
    tab1, tab2 = st.tabs(["Fisiologia (rMSSD & RHR)", "Database"])
    
    with tab1:
        st.line_chart(df.set_index('Date')[['rMSSD', 'RHR']], color=["#0000FF", "#FF0000"])
    
    with tab2:
        st.dataframe(df.sort_values('Date', ascending=False))

else:
    st.info("👋 Il database è vuoto. Carica i file dalla barra laterale.")
