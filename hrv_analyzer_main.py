import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="HRV Engineer Dashboard", layout="wide", page_icon="🫀")

# Titolo e Header
st.title("🫀 HRV Engineer Dashboard")
st.markdown("### Monitoraggio Ingegneristico dello Stato di Forma")

# --- NOME DEL DATABASE ---
# Il file verrà creato automaticamente nella stessa cartella dello script
DB_FILE = 'hrv_database.csv'

# --- FUNZIONI DI SERVIZIO (BACKEND) ---

def parse_rr_file(uploaded_file):
    """
    Legge il file raw .txt esportato da Elite HRV.
    Estrae gli intervalli RR e calcola rMSSD e RHR.
    """
    try:
        # Legge il contenuto del file caricato
        content = uploaded_file.getvalue().decode("utf-8").splitlines()
        rr_intervals = []
        
        for line in content:
            line = line.strip()
            # Elite HRV txt contiene solo numeri (ms)
            if line.isdigit():
                val = int(line)
                # Filtro fisiologico (scarta artefatti <300ms o >2000ms)
                if 300 < val < 2000: 
                    rr_intervals.append(val)
        
        if len(rr_intervals) < 10:
            st.error("Il file sembra vuoto o corrotto.")
            return None, None

        # --- CALCOLI MATEMATICI ---
        rr_array = np.array(rr_intervals)
        
        # 1. rMSSD (Root Mean Square of Successive Differences)
        diffs = np.diff(rr_array)
        squared_diffs = np.square(diffs)
        rmssd = np.sqrt(np.mean(squared_diffs))
        
        # 2. RHR (Resting Heart Rate)
        mean_rr = np.mean(rr_array)
        rhr = 60000 / mean_rr
        
        return round(rmssd, 2), round(rhr, 1)
        
    except Exception as e:
        st.error(f"Errore nel parsing del file: {e}")
        return None, None

def load_db():
    """Carica il database CSV esistente o ne crea uno vuoto."""
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE, parse_dates=['Date'])
    else:
        return pd.DataFrame(columns=['Date', 'rMSSD', 'RHR', 'Sleep', 'Feel', 'Status'])

def save_entry(date, rmssd, rhr, sleep, feel, status):
    """Salva una nuova entrata nel CSV."""
    df = load_db()
    
    # Controlla se esiste già una lettura per questa data (evita duplicati)
    date_only = pd.to_datetime(date).date()
    # Se la colonna Date esiste, facciamo un check
    if not df.empty:
        df['Date_Only'] = df['Date'].dt.date
        if date_only in df['Date_Only'].values:
            st.warning(f"⚠️ Esiste già un dato per il {date_only}. Il vecchio dato è stato sovrascritto.")
            df = df[df['Date_Only'] != date_only] # Rimuove la vecchia entry
        df = df.drop(columns=['Date_Only'], errors='ignore')
    
    new_row = pd.DataFrame({
        'Date': [date],
        'rMSSD': [rmssd],
        'RHR': [rhr],
        'Sleep': [sleep],
        'Feel': [feel],
        'Status': [status]
    })
    
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values('Date')
    df.to_csv(DB_FILE, index=False)
    return df

def get_traffic_light(rmssd, rhr, sleep, feel, df_history):
    """
    L'ALGORITMO DECISIONALE.
    Confronta i dati di oggi con la media mobile degli ultimi 7 giorni.
    """
    if len(df_history) < 3:
        return "⚪ DATI INSUFFICIENTI (Continua a misurare)"
    
    # Calcolo Baseline (Media ultimi 7 giorni disponibili)
    last_7 = df_history.tail(7)
    base_rmssd = last_7['rMSSD'].mean()
    base_rhr = last_7['RHR'].mean()
    
    st.info(f"📊 **La tua Baseline (7gg):** rMSSD {base_rmssd:.1f} ms | RHR {base_rhr:.1f} bpm")
    
    # --- LOGICA IF-THEN (PROTOCOLLO INGEGNERISTICO) ---
    
    # SCENARIO ROSSO: Crollo HRV e Aumento RHR (Probabile malattia/sovrallenamento)
    if rmssd < base_rmssd * 0.85 and rhr > base_rhr * 1.05:
        return "🔴 RIPOSO (Crash Sistemico)"
    
    # SCENARIO GIALLO: Uno dei parametri è fuori posto o ti senti male
    elif rmssd < base_rmssd * 0.90 or rhr > base_rhr * 1.03 or feel < 6:
        return "🟡 CAUTELA (Carico Ridotto / Z2)"
    
    # SCENARIO PARADOSSO: HRV troppo alto ma recupero incompleto (Saturazione Parasimpatica)
    elif rmssd > base_rmssd * 1.30 and sleep < 7:
        return "🟡 PARADOSSO (Occhio ai Falsi Positivi)"
        
    # SCENARIO VERDE: Tutto ok
    else:
        return "🟢 GO (Via Libera Qualità)"

# --- INTERFACCIA UTENTE (SIDEBAR) ---

with st.sidebar:
    st.header("📝 Nuova Lettura")
    st.write("Carica qui il file .txt esportato da Elite HRV")
    
    uploaded_file = st.file_uploader("Upload File", type=['txt'])
    
    if uploaded_file:
        # Prendi la data di oggi come default
        date_input = st.date_input("Data Lettura", datetime.today())
        
        # Parsa il file
        rmssd_val, rhr_val = parse_rr_file(uploaded_file)
        
        if rmssd_val:
            st.success(f"✅ Dati Estratti: rMSSD {rmssd_val} | RHR {rhr_val}")
            st.markdown("---")
            
            # Input Soggettivi
            sleep_val = st.number_input("Ore di Sonno", min_value=0.0, max_value=12.0, value=7.5, step=0.5)
            feel_val = st.slider("Sensazione (1=Zombie, 10=Top)", 1, 10, 7)
            
            if st.button("💾 Salva e Calcola Status"):
                # 1. Carica storico
                history_df = load_db()
                # 2. Calcola Algoritmo
                status = get_traffic_light(rmssd_val, rhr_val, sleep_val, feel_val, history_df)
                # 3. Salva
                save_entry(pd.to_datetime(date_input), rmssd_val, rhr_val, sleep_val, feel_val, status)
                st.balloons() # Animazione
                st.success(f"Dato salvato! Verdetto: {status}")

# --- DASHBOARD PRINCIPALE ---

df = load_db()

if not df.empty:
    # Mostra l'ultima lettura in grande (KPI)
    last_entry = df.iloc[-1]
    
    st.subheader(f"📅 Situazione del {last_entry['Date'].strftime('%d/%m/%Y')}")
    
    # Colonne metriche
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("rMSSD (HRV)", f"{last_entry['rMSSD']} ms")
    col2.metric("RHR (Cuore)", f"{last_entry['RHR']} bpm")
    col3.metric("Sonno", f"{last_entry['Sleep']} h")
    col4.metric("Feel", f"{last_entry['Feel']}/10")

    # Box Colorato con il Verdetto
    status_msg = last_entry['Status']
    if "🟢" in status_msg:
        st.success(f"## {status_msg}")
    elif "🟡" in status_msg:
        st.warning(f"## {status_msg}")
    else:
        st.error(f"## {status_msg}")

    st.divider()

    # --- GRAFICI DEI TREND ---
    st.subheader("📈 Trend Storici")
    
    tab1, tab2 = st.tabs(["Fisiologia (rMSSD & RHR)", "Lifestyle (Sonno & Feel)"])
    
    with tab1:
        st.markdown("**Linea Blu: HRV (Più alta è meglio) | Linea Rossa: RHR (Più bassa è meglio)**")
        # Grafico combinato semplice
        chart_data = df.set_index('Date')[['rMSSD', 'RHR']]
        st.line_chart(chart_data, color=["#0000FF", "#FF0000"]) # Blu e Rosso
    
    with tab2:
        st.markdown("**Ore di Sonno e Sensazione soggettiva**")
        st.line_chart(df.set_index('Date')[['Sleep', 'Feel']])

    # --- TABELLA DATI ---
    with st.expander("📂 Vedi Database Completo (Raw Data)"):
        st.dataframe(df.sort_values('Date', ascending=False).style.format({
            'rMSSD': '{:.2f}',
            'RHR': '{:.1f}',
            'Sleep': '{:.1f}'
        }))

else:
    st.info("👋 Ciao! Il database è vuoto. Carica il tuo primo file .txt dalla barra laterale sinistra per iniziare.")
