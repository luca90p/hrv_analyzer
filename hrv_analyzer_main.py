import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="HRV Engineer Dashboard", layout="wide", page_icon="🫀")

# Titolo e Header
st.title("🫀 HRV Engineer Dashboard")
st.markdown("### Monitoraggio Ingegneristico - Integrazione Garmin & HRV")

# --- NOME DEL DATABASE ---
DB_FILE = 'hrv_database.csv'

# --- 1. PARSING FILE HRV (RAW TXT) ---
def parse_rr_file(file_content):
    """
    Legge il contenuto raw del file TXT con gli intervalli RR.
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
    """Estrae la data dal nome file 'YYYY-MM-DD HH-MM-SS.txt'."""
    try:
        name_clean = os.path.splitext(filename)[0]
        timestamp = datetime.strptime(name_clean, "%Y-%m-%d %H-%M-%S")
        return timestamp
    except ValueError:
        return None

# --- 2. PARSING FILE GARMIN (CSV) ---
def parse_garmin_file(uploaded_file):
    """
    Legge il CSV di riposo Garmin e standardizza le colonne.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Mapping dinamico delle colonne
        date_col = df.columns[0] 
        
        df = df.rename(columns={
            date_col: 'Date',
            'Dormire': 'Sleep_Start',
            'Durata': 'Sleep_Duration',
            'Qualità': 'Sleep_Quality'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Pulizia Durata (da "7h 53min" a float 7.88)
        def clean_duration(val):
            if pd.isna(val): return 0
            val = str(val).lower().replace('h', '').replace('min', '')
            parts = val.split()
            if len(parts) == 2:
                return round(float(parts[0]) + float(parts[1])/60, 2)
            elif len(parts) == 1:
                return float(parts[0])
            return 0
            
        df['Sleep_Hours'] = df['Sleep_Duration'].apply(clean_duration)
        
        # Mappa Qualità in voto numerico
        quality_map = {'Eccellente': 9, 'Buono': 8, 'Discreto': 6, 'Scarso': 4}
        df['Feel_Score'] = df['Sleep_Quality'].map(quality_map).fillna(5)
        
        return df[['Date', 'Sleep_Start', 'Sleep_Hours', 'Feel_Score', 'Sleep_Quality']]
    except Exception as e:
        st.error(f"Errore lettura Garmin: {e}")
        return pd.DataFrame()

# --- GESTIONE DATABASE ---
def load_db():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, parse_dates=['Date'])
    else:
        return pd.DataFrame(columns=['Date', 'rMSSD', 'RHR', 'Sleep', 'Feel', 'Status'])

def get_traffic_light(rmssd, rhr, sleep, feel, df_history):
    """Algoritmo decisionale basato su media mobile 7gg."""
    if df_history.empty or len(df_history) < 3:
        return "⚪ DATI INSUFFICIENTI (Start)"
    
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

# --- PROCESSO DI MERGE (CORE LOGIC) ---
def process_data_integration(hrv_files, garmin_file):
    st.write("⏳ Elaborazione in corso...")
    
    # A. Elaborazione HRV
    hrv_data = []
    for f in hrv_files:
        dt = extract_date_from_filename(f.name)
        if dt:
            rmssd, rhr = parse_rr_file(f.getvalue())
            if rmssd:
                hrv_data.append({
                    'Date': dt,
                    'Date_Day': dt.date(),
                    'rMSSD': rmssd, 
                    'RHR': rhr
                })
    
    df_hrv = pd.DataFrame(hrv_data)
    if df_hrv.empty:
        st.warning("Nessun dato HRV valido trovato.")
        return load_db()

    # B. Elaborazione Garmin
    df_garmin = pd.DataFrame()
    if garmin_file:
        df_garmin = parse_garmin_file(garmin_file)
        if not df_garmin.empty:
            df_garmin['Date_Day'] = df_garmin['Date'].dt.date 
            
    # C. Unione
    final_rows = []
    df_hrv = df_hrv.sort_values(by='Date')
    df_current = load_db()
    temp_history = df_current.copy()

    for _, row in df_hrv.iterrows():
        entry = {
            'Date': row['Date'],
            'rMSSD': row['rMSSD'],
            'RHR': row['RHR'],
            'Sleep': 7.5, 
            'Feel': 7,    
            'Status': 'Calcolo...'
        }
        
        if not df_garmin.empty:
            match = df_garmin[df_garmin['Date_Day'] == row['Date_Day']]
            if not match.empty:
                garmin_row = match.iloc[0]
                entry['Sleep'] = garmin_row['Sleep_Hours']
                entry['Feel'] = garmin_row['Feel_Score']
        
        status = get_traffic_light(
            entry['rMSSD'], entry['RHR'], entry['Sleep'], entry['Feel'], temp_history
        )
        entry['Status'] = status
        
        final_rows.append(entry)
        temp_history = pd.concat([temp_history, pd.DataFrame([entry])], ignore_index=True)

    # D. Salvataggio
    if final_rows:
        df_new = pd.DataFrame(final_rows)
        df_updated = pd.concat([df_current, df_new], ignore_index=True)
        df_updated = df_updated.drop_duplicates(subset=['Date'], keep='last')
        df_updated = df_updated.sort_values(by='Date')
        
        df_updated.to_csv(DB_FILE, index=False)
        st.success(f"✅ Database aggiornato! Elaborati {len(final_rows)} record.")
        return df_updated
    
    return df_current

# --- INTERFACCIA UTENTE (SIDEBAR) ---

with st.sidebar:
    st.header("📂 1. Carica HRV")
    hrv_files = st.file_uploader(
        "File .txt (Nomi: YYYY-MM-DD HH-MM-SS)", 
        type=['txt'], 
        accept_multiple_files=True,
        key="hrv_uploader"
    )
    
    st.markdown("---")
    
    st.header("⌚ 2. Carica Garmin")
    garmin_file = st.file_uploader(
        "File 'Riposo.csv' Garmin",
        type=['csv'],
        key="garmin_uploader"
    )
    
    st.markdown("---")
    
    if hrv_files:
        if st.button("🚀 Elabora e Unisci Dati"):
            df = process_data_integration(hrv_files, garmin_file)
            st.balloons()

# --- DASHBOARD PRINCIPALE ---

df = load_db()

if not df.empty:
    # KPI
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

    # --- GRAFICI DEI TREND (CORRETTO) ---
    st.subheader("📈 Analisi Storica")
    
    # Creiamo 3 TAB ora invece di 2
    tab1, tab2, tab3 = st.tabs(["Fisiologia (rMSSD & RHR)", "Sonno & Recupero", "Database Completo"])
    
    # TAB 1: Fisiologia
    with tab1:
        st.caption("Andamento HRV e Battiti a Riposo")
        st.line_chart(df.set_index('Date')[['rMSSD', 'RHR']], color=["#0000FF", "#FF0000"])
    
    # TAB 2: Sonno & Recupero (AGGIUNTO)
    with tab2:
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            st.markdown("##### 🌙 Durata Sonno (ore)")
            # Usa bar_chart per evidenziare le ore
            st.bar_chart(df.set_index('Date')['Sleep'], color="#6A0DAD") 
            
        with col_graph2:
            st.markdown("##### ⚡ Sensazione al Risveglio (1-10)")
            # Usa line_chart per il trend
            st.line_chart(df.set_index('Date')['Feel'], color="#FFA500")

    # TAB 3: Dati
    with tab3:
        st.dataframe(df.sort_values('Date', ascending=False))

else:
    st.info("👋 Il database è vuoto. Carica i file HRV (e opzionalmente Garmin) dalla barra laterale.")
