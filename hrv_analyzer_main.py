import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="HRV Engineer Dashboard", layout="wide", page_icon="🫀")

# Titolo e Header
st.title("🫀 HRV Engineer Dashboard")
st.markdown("### Monitoraggio Ingegneristico: Carico vs Recupero")

# --- NOME DEL DATABASE ---
DB_FILE = 'hrv_database.csv'

# --- 1. PARSING FILE HRV (RAW TXT) ---
def parse_rr_file(file_content):
    try:
        content = file_content.decode("utf-8").splitlines()
        rr_intervals = []
        for line in content:
            line = line.strip()
            if line.isdigit():
                val = int(line)
                if 300 < val < 2000: rr_intervals.append(val)
        
        if len(rr_intervals) < 10: return None, None

        rr_array = np.array(rr_intervals)
        diffs = np.diff(rr_array)
        rmssd = np.sqrt(np.mean(np.square(diffs)))
        rhr = 60000 / np.mean(rr_array)
        
        return round(rmssd, 2), round(rhr, 1)
    except: return None, None

def extract_date_from_filename(filename):
    try:
        name_clean = os.path.splitext(filename)[0]
        timestamp = datetime.strptime(name_clean, "%Y-%m-%d %H-%M-%S")
        return timestamp
    except ValueError: return None

# --- 2. PARSING GARMIN SONNO ---
def parse_garmin_sleep(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        date_col = df.columns[0]
        df = df.rename(columns={
            date_col: 'Date', 'Dormire': 'Sleep_Start', 
            'Durata': 'Sleep_Duration', 'Qualità': 'Sleep_Quality'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        
        def clean_duration(val):
            if pd.isna(val): return 0
            val = str(val).lower().replace('h', '').replace('min', '')
            parts = val.split()
            if len(parts) == 2: return round(float(parts[0]) + float(parts[1])/60, 2)
            elif len(parts) == 1: return float(parts[0])
            return 0
            
        df['Sleep_Hours'] = df['Sleep_Duration'].apply(clean_duration)
        quality_map = {'Eccellente': 9, 'Buono': 8, 'Discreto': 6, 'Scarso': 4}
        df['Feel_Score'] = df['Sleep_Quality'].map(quality_map).fillna(5)
        
        return df[['Date', 'Sleep_Hours', 'Feel_Score']]
    except Exception as e:
        st.error(f"Errore file Sonno: {e}")
        return pd.DataFrame()

# --- 3. PARSING GARMIN ATTIVITÀ (CON LOGICA SMART LOAD) ---
def parse_garmin_activities(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # Pulizia base
        df['Data'] = pd.to_datetime(df['Data'])
        df['Date_Day'] = df['Data'].dt.date
        
        # Helper Numerici
        def clean_num(x):
            if pd.isna(x): return 0.0
            return float(str(x).replace(',', ''))

        def clean_time(x):
            try:
                parts = str(x).split(':')
                if len(parts) == 3: return int(parts[0])*60 + int(parts[1]) + float(parts[2])/60
                elif len(parts) == 2: return int(parts[0]) + float(parts[1])/60
                return 0.0
            except: return 0.0

        # Colonne necessarie
        df['Mins'] = df['Tempo'].apply(clean_time)
        df['TE'] = df['TE aerobico'].apply(clean_num)
        df['TSS'] = df['Training Stress Score®'].apply(clean_num)
        df['Type'] = df['Tipo di attività'].astype(str)
        df['Dist_km'] = df['Distanza'].apply(clean_num)

        # --- ALGORITMO "SMART LOAD" ---
        def calculate_load(row):
            # 1. Se TSS è affidabile (> 10), usalo
            if row['TSS'] > 10:
                return row['TSS']
            
            # 2. Altrimenti: Calcolo Eco-Load (Durata * TE)
            # Se TE è mancante (es. nuoto manuale), assumiamo 2.0 (basso) o 3.0 (medio)
            te_val = row['TE'] if row['TE'] > 0 else 2.5 
            base_load = row['Mins'] * te_val
            
            # 3. Fattore Correttivo Attività (CNS Multiplier)
            activity_type = row['Type'].lower()
            multiplier = 1.0
            
            if any(x in activity_type for x in ['forza', 'palestra', 'pesi', 'crossfit', 'strength']):
                multiplier = 1.5 # Boost per stress nervoso
            elif any(x in activity_type for x in ['yoga', 'pilates']):
                multiplier = 0.5 # Riduzione per recupero attivo
                
            return base_load * multiplier

        df['Load_Score'] = df.apply(calculate_load, axis=1)

        # Aggregazione Giornaliera
        daily_grp = df.groupby('Date_Day').agg({
            'Load_Score': 'sum',    # Somma tutto il carico del giorno
            'Dist_km': 'sum',
            'Mins': 'sum'
        }).reset_index()
        
        daily_grp = daily_grp.rename(columns={'Date_Day': 'Date'})
        daily_grp['Date'] = pd.to_datetime(daily_grp['Date'])
        
        return daily_grp

    except Exception as e:
        st.error(f"Errore file Attività: {e}")
        return pd.DataFrame()

# --- GESTIONE DATABASE ---
def load_db():
    cols = ['Date', 'rMSSD', 'RHR', 'Sleep', 'Feel', 'Status', 'Daily_Load', 'Daily_Dist', 'Daily_TrainTime']
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE, parse_dates=['Date'])
        for col in cols:
            if col not in df.columns: df[col] = np.nan
        return df
    else:
        return pd.DataFrame(columns=cols)

def recalculate_status(df):
    df = df.sort_values('Date').reset_index(drop=True)
    new_statuses = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Saltiamo se manca HRV
        if pd.isna(row['rMSSD']):
            new_statuses.append("⚪ DATI MANCANTI")
            continue

        # Baseline storica (7gg)
        history = df.iloc[:i].tail(7)
        if len(history) < 3:
            new_statuses.append("⚪ START UP")
            continue
            
        base_rmssd = history['rMSSD'].mean()
        
        # Logica Semaforo Semplificata
        rmssd = row['rMSSD']
        # Se HRV crolla > 15% sotto media -> Rosso
        if rmssd < base_rmssd * 0.85:
            new_statuses.append("🔴 RIPOSO (Stress Acuto)")
        # Se HRV scende leggermente o Feel è basso -> Giallo
        elif rmssd < base_rmssd * 0.95 or (pd.notna(row['Feel']) and row['Feel'] < 6):
            new_statuses.append("🟡 CAUTELA")
        # Altrimenti -> Verde
        else:
            new_statuses.append("🟢 GO")
            
    df['Status'] = new_statuses
    return df

# --- UPLOAD E UPDATE ---
def update_db_generic(new_df, merge_cols):
    current_db = load_db()
    
    # Prepara key di merge (Data Giorno)
    current_db['Date'] = pd.to_datetime(current_db['Date'])
    current_db['Date_Day'] = current_db['Date'].dt.date
    
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df['Date_Day'] = new_df['Date'].dt.date
    
    cnt_new, cnt_upd = 0, 0
    
    for _, row in new_df.iterrows():
        mask = current_db['Date_Day'] == row['Date_Day']
        
        if current_db[mask].empty:
            # Crea nuova riga
            new_entry = {k: np.nan for k in current_db.columns if k not in ['Date', 'Date_Day']}
            new_entry['Date'] = row['Date']
            for c in merge_cols: 
                if c in row: new_entry[c] = row[c]
            
            current_db = pd.concat([current_db, pd.DataFrame([new_entry])], ignore_index=True)
            current_db['Date_Day'] = current_db['Date'].dt.date # Refresh key
            cnt_new += 1
        else:
            # Aggiorna
            idx = current_db[mask].index[0]
            for c in merge_cols:
                if c in row and pd.notna(row[c]):
                    current_db.at[idx, c] = row[c]
            cnt_upd += 1
            
    if 'Date_Day' in current_db.columns: current_db.drop(columns=['Date_Day'], inplace=True)
    
    final = recalculate_status(current_db)
    final.to_csv(DB_FILE, index=False)
    return cnt_new, cnt_upd

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 1. HRV")
    f_hrv = st.file_uploader("File TXT", type=['txt'], accept_multiple_files=True)
    if f_hrv and st.button("Carica HRV"):
        data = []
        for f in f_hrv:
            dt = extract_date_from_filename(f.name)
            if dt:
                r, h = parse_rr_file(f.getvalue())
                if r: data.append({'Date': dt, 'rMSSD': r, 'RHR': h})
        if data:
            n, u = update_db_generic(pd.DataFrame(data), ['rMSSD', 'RHR'])
            st.success(f"HRV: {n} nuovi, {u} agg.")
            st.rerun()

    st.header("🌙 2. Sonno")
    f_sleep = st.file_uploader("Riposo.csv", type=['csv'])
    if f_sleep and st.button("Carica Sonno"):
        df_s = parse_garmin_sleep(f_sleep)
        if not df_s.empty:
            n, u = update_db_generic(df_s, ['Sleep', 'Feel'])
            st.success(f"Sonno: {n} nuovi, {u} agg.")
            st.rerun()

    st.header("🏋️ 3. Attività")
    f_act = st.file_uploader("Activities.csv", type=['csv'])
    if f_act and st.button("Carica Attività"):
        df_a = parse_garmin_activities(f_act)
        if not df_a.empty:
            # Mapping nomi
            df_a = df_a.rename(columns={'Mins': 'Daily_TrainTime', 'Load_Score': 'Daily_Load', 'Dist_km': 'Daily_Dist'})
            n, u = update_db_generic(df_a, ['Daily_Load', 'Daily_TrainTime', 'Daily_Dist'])
            st.success(f"Attività: {n} nuovi, {u} agg.")
            st.rerun()

# --- DASHBOARD ---
df = load_db()

if not df.empty:
    df = df.sort_values('Date')
    last = df.iloc[-1]
    
    st.subheader(f"📊 Report: {last['Date'].strftime('%d/%m/%Y')}")
    
    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("rMSSD (Fisiologia)", f"{last['rMSSD']} ms", delta_color="normal")
    k2.metric("Load (Carico)", f"{int(last['Daily_Load'])}" if pd.notna(last['Daily_Load']) else "--", help="Punteggio Calcolato: Durata x Intensità (con boost Forza)")
    k3.metric("Sonno", f"{last['Sleep']} h")
    k4.metric("Status", last['Status'])

    st.divider()

    # --- GRAFICI CORRELAZIONE ---
    t1, t2, t3 = st.tabs(["⚡ Carico vs HRV", "🌙 Recupero", "📝 Dati"])
    
    with t1:
        st.markdown("#### Impatto dell'Allenamento sulla Fisiologia")
        st.caption("Barre Viola = Quanto ti sei allenato (Load). Linea Blu = Il tuo HRV (rMSSD).")
        st.caption("Cerca questo pattern: Barra Alta oggi -> Linea che scende domani.")
        
        # Grafico Combo personalizzato
        chart_data = df.set_index('Date')[['Daily_Load', 'rMSSD']].copy()
        
        # Visualizzazione con due assi Y sarebbe l'ideale, qui usiamo st.bar_chart e st.line_chart separati ma vicini
        # o normalizzati. Per semplicità Streamlit:
        c1, c2 = st.columns([3, 1])
        with c1:
             st.bar_chart(chart_data['Daily_Load'], color="#800080") # Viola
        with c2:
             st.line_chart(chart_data['rMSSD'], color="#0000FF")   # Blu

    with t2:
        c_s1, c_s2 = st.columns(2)
        with c_s1: 
            st.markdown("**Quantità Sonno (h)**")
            st.bar_chart(df.set_index('Date')['Sleep'], color="#2E8B57")
        with c_s2: 
            st.markdown("**Qualità Percepita (1-10)**")
            st.line_chart(df.set_index('Date')['Feel'], color="#FFA500")

    with t3:
        st.dataframe(df.sort_values('Date', ascending=False))

else:
    st.info("Inizia caricando i file dalla barra laterale.")
