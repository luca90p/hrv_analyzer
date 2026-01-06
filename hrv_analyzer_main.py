import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="HRV Engineer Dashboard", layout="wide", page_icon="🫀")

# Titolo e Header
st.title("🫀 HRV Engineer Dashboard")
st.markdown("### Monitoraggio Ingegneristico - Comandi Separati")

# --- NOME DEL DATABASE ---
DB_FILE = 'hrv_database.csv'

# --- 1. PARSING FILE HRV (RAW TXT) ---
def parse_rr_file(file_content):
    """Legge il contenuto raw del file TXT con gli intervalli RR."""
    try:
        content = file_content.decode("utf-8").splitlines()
        rr_intervals = []
        for line in content:
            line = line.strip()
            if line.isdigit():
                val = int(line)
                if 300 < val < 2000: # Filtro fisiologico
                    rr_intervals.append(val)
        
        if len(rr_intervals) < 10: return None, None

        rr_array = np.array(rr_intervals)
        # rMSSD & RHR
        diffs = np.diff(rr_array)
        rmssd = np.sqrt(np.mean(np.square(diffs)))
        rhr = 60000 / np.mean(rr_array)
        
        return round(rmssd, 2), round(rhr, 1)
    except Exception:
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
    """Legge il CSV Garmin e standardizza le colonne."""
    try:
        df = pd.read_csv(uploaded_file)
        date_col = df.columns[0] # Assume la prima colonna come data
        
        df = df.rename(columns={
            date_col: 'Date',
            'Dormire': 'Sleep_Start',
            'Durata': 'Sleep_Duration',
            'Qualità': 'Sleep_Quality'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Pulizia Durata
        def clean_duration(val):
            if pd.isna(val): return 0
            val = str(val).lower().replace('h', '').replace('min', '')
            parts = val.split()
            if len(parts) == 2: return round(float(parts[0]) + float(parts[1])/60, 2)
            elif len(parts) == 1: return float(parts[0])
            return 0
            
        df['Sleep_Hours'] = df['Sleep_Duration'].apply(clean_duration)
        
        # Mapping Qualità
        quality_map = {'Eccellente': 9, 'Buono': 8, 'Discreto': 6, 'Scarso': 4}
        df['Feel_Score'] = df['Sleep_Quality'].map(quality_map).fillna(5)
        
        return df[['Date', 'Sleep_Hours', 'Feel_Score']]
    except Exception as e:
        st.error(f"Errore lettura Garmin: {e}")
        return pd.DataFrame()

# --- GESTIONE DATABASE E CALCOLI ---
def load_db():
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE, parse_dates=['Date'])
        # Assicuriamoci che tutte le colonne esistano
        required_cols = ['Date', 'rMSSD', 'RHR', 'Sleep', 'Feel', 'Status']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan if col != 'Status' else 'Da Calcolare'
        return df
    else:
        return pd.DataFrame(columns=['Date', 'rMSSD', 'RHR', 'Sleep', 'Feel', 'Status'])

def recalculate_status(df):
    """Ricalcola lo status (Traffic Light) per tutto il dataframe."""
    df = df.sort_values('Date').reset_index(drop=True)
    
    new_statuses = []
    
    # Per calcolare le medie mobili correttamente
    for i in range(len(df)):
        current_row = df.iloc[i]
        
        # Se mancano dati critici, saltiamo il calcolo
        if pd.isna(current_row['rMSSD']) or pd.isna(current_row['RHR']):
            new_statuses.append("⚪ DATI PARZIALI")
            continue

        # Storico: prendiamo fino a 7 giorni precedenti (escludendo oggi)
        history = df.iloc[:i].tail(7)
        
        if len(history) < 3:
            new_statuses.append("⚪ DATI INSUFFICIENTI")
            continue
            
        base_rmssd = history['rMSSD'].mean()
        base_rhr = history['RHR'].mean()
        
        rmssd = current_row['rMSSD']
        rhr = current_row['RHR']
        sleep = current_row['Sleep'] if not pd.isna(current_row['Sleep']) else 7.5
        feel = current_row['Feel'] if not pd.isna(current_row['Feel']) else 7
        
        # Logica IF-THEN
        status = "🟢 GO"
        if rmssd < base_rmssd * 0.85 and rhr > base_rhr * 1.05:
            status = "🔴 RIPOSO (Crash)"
        elif rmssd < base_rmssd * 0.90 or rhr > base_rhr * 1.03 or feel < 6:
            status = "🟡 CAUTELA"
        elif rmssd > base_rmssd * 1.30 and sleep < 7:
            status = "🟡 PARADOSSO"
            
        new_statuses.append(status)
        
    df['Status'] = new_statuses
    return df

# --- LOGICA AGGIORNAMENTO SEPARATO ---

def process_hrv_upload(uploaded_files):
    """Carica SOLO HRV e aggiorna/inserisce nel DB."""
    current_db = load_db()
    
    new_data = []
    for f in uploaded_files:
        dt = extract_date_from_filename(f.name)
        if dt:
            rmssd, rhr = parse_rr_file(f.getvalue())
            if rmssd:
                new_data.append({'Date': dt, 'rMSSD': rmssd, 'RHR': rhr})
    
    if not new_data:
        st.warning("Nessun dato valido trovato nei file.")
        return

    df_new = pd.DataFrame(new_data)
    
    # Merge Intelligente:
    # 1. Convertiamo Date in colonna chiave
    current_db['Date'] = pd.to_datetime(current_db['Date'])
    df_new['Date'] = pd.to_datetime(df_new['Date'])
    
    # 2. Iteriamo sui nuovi dati per aggiornare o inserire
    count_updated = 0
    count_new = 0
    
    for _, row in df_new.iterrows():
        mask = current_db['Date'] == row['Date']
        if current_db[mask].empty:
            # Nuova riga
            new_row = row.to_dict()
            new_row['Sleep'] = np.nan # Lasciamo vuoto se non c'è
            new_row['Feel'] = np.nan
            current_db = pd.concat([current_db, pd.DataFrame([new_row])], ignore_index=True)
            count_new += 1
        else:
            # Aggiorna esistente (solo colonne HRV)
            idx = current_db[mask].index[0]
            current_db.at[idx, 'rMSSD'] = row['rMSSD']
            current_db.at[idx, 'RHR'] = row['RHR']
            count_updated += 1
            
    # Ricalcola status e salva
    final_db = recalculate_status(current_db)
    final_db.to_csv(DB_FILE, index=False)
    st.success(f"✅ HRV Elaborato: {count_new} nuovi record, {count_updated} aggiornati.")

def process_garmin_upload(garmin_file):
    """Carica SOLO Garmin e aggiorna/inserisce nel DB."""
    current_db = load_db()
    df_garmin = parse_garmin_file(garmin_file)
    
    if df_garmin.empty: return

    current_db['Date'] = pd.to_datetime(current_db['Date'])
    # Normalizziamo le date al 'giorno' per il matching (evita mismatch di ore)
    current_db['Date_Day'] = current_db['Date'].dt.date
    df_garmin['Date_Day'] = df_garmin['Date'].dt.date
    
    count_merged = 0
    count_added = 0
    
    for _, row in df_garmin.iterrows():
        mask = current_db['Date_Day'] == row['Date_Day']
        
        if current_db[mask].empty:
            # Se la data non esiste (es. hai il sonno ma non hai ancora misurato HRV)
            # Creiamo la riga usando la data del Garmin
            new_row = {
                'Date': row['Date'], # Usiamo il timestamp del Garmin
                'rMSSD': np.nan,
                'RHR': np.nan,
                'Sleep': row['Sleep_Hours'],
                'Feel': row['Feel_Score']
            }
            current_db = pd.concat([current_db, pd.DataFrame([new_row])], ignore_index=True)
            # Rigeneriamo la colonna Date_Day per i prossimi cicli
            current_db['Date_Day'] = current_db['Date'].dt.date
            count_added += 1
        else:
            # Aggiorna riga esistente
            idx = current_db[mask].index[0]
            current_db.at[idx, 'Sleep'] = row['Sleep_Hours']
            current_db.at[idx, 'Feel'] = row['Feel_Score']
            count_merged += 1
            
    # Pulizia colonna temporanea
    if 'Date_Day' in current_db.columns:
        current_db = current_db.drop(columns=['Date_Day'])
        
    final_db = recalculate_status(current_db)
    final_db.to_csv(DB_FILE, index=False)
    st.success(f"✅ Garmin Elaborato: {count_merged} giorni aggiornati, {count_added} nuovi inseriti.")

# --- INTERFACCIA UTENTE (SIDEBAR) ---

with st.sidebar:
    st.header("📂 1. Importazione HRV")
    st.caption("Carica i file .txt delle misurazioni")
    hrv_files = st.file_uploader("File TXT HRV", type=['txt'], accept_multiple_files=True, key="hrv_up")
    
    if hrv_files:
        if st.button("💾 Elabora File HRV"):
            process_hrv_upload(hrv_files)
            st.rerun() # Ricarica la pagina per vedere i dati
            
    st.markdown("---")
    
    st.header("⌚ 2. Importazione Garmin")
    st.caption("Carica il file 'Riposo.csv' unico")
    garmin_file = st.file_uploader("File CSV Garmin", type=['csv'], key="garm_up")
    
    if garmin_file:
        if st.button("🔄 Aggiorna Dati Sonno"):
            process_garmin_upload(garmin_file)
            st.rerun()

# --- DASHBOARD PRINCIPALE ---

df = load_db()

if not df.empty:
    df = df.sort_values('Date')
    last_entry = df.iloc[-1]
    
    st.subheader(f"📅 Ultimo Aggiornamento: {last_entry['Date'].strftime('%d/%m/%Y')}")
    
    # Gestione visualizzazione NaN nei metric
    rmssd_val = f"{last_entry['rMSSD']} ms" if pd.notna(last_entry['rMSSD']) else "--"
    sleep_val = f"{last_entry['Sleep']} h" if pd.notna(last_entry['Sleep']) else "--"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("rMSSD", rmssd_val)
    col2.metric("RHR", f"{last_entry['RHR']} bpm" if pd.notna(last_entry['RHR']) else "--")
    col3.metric("Sonno", sleep_val)
    col4.metric("Status", last_entry['Status'])

    # Messaggio Status
    status_msg = str(last_entry['Status'])
    if "🟢" in status_msg: st.success(f"## {status_msg}")
    elif "🟡" in status_msg: st.warning(f"## {status_msg}")
    elif "🔴" in status_msg: st.error(f"## {status_msg}")
    else: st.info(f"## {status_msg}")

    st.divider()

    # --- TABS GRAFICI ---
    tab1, tab2, tab3 = st.tabs(["🫀 Fisiologia", "🌙 Sonno & Recupero", "📝 Database"])
    
    with tab1:
        st.line_chart(df.set_index('Date')[['rMSSD', 'RHR']], color=["#0000FF", "#FF0000"])
    
    with tab2:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### Durata Sonno")
            st.bar_chart(df.set_index('Date')['Sleep'], color="#6A0DAD")
        with col_g2:
            st.markdown("##### Sensazione (Feel)")
            st.line_chart(df.set_index('Date')['Feel'], color="#FFA500")

    with tab3:
        st.dataframe(df.sort_values('Date', ascending=False))

else:
    st.info("👋 Database vuoto. Usa la barra laterale per caricare i dati (HRV o Garmin).")
