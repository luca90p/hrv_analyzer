import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="HRV Engineer Dashboard", layout="wide", page_icon="🫀")

st.title("🫀 HRV Engineer Dashboard")
st.markdown("### Monitoraggio Ingegneristico: Carico Reale & Cronobiologia")

DB_FILE = 'hrv_database.csv'

# --- 1. PARSING HRV ---
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

# --- 2. PARSING SONNO ---
def parse_garmin_sleep(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        date_col = df.columns[0]
        df = df.rename(columns={date_col: 'Date', 'Durata': 'Sleep_Duration', 'Qualità': 'Sleep_Quality'})
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
    except: return pd.DataFrame()

# --- 3. PARSING ATTIVITÀ (LOGICA AVANZATA) ---
def parse_garmin_activities(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
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

        df['Mins'] = df['Tempo'].apply(clean_time)
        df['TE'] = df['TE aerobico'].apply(clean_num)
        df['TSS'] = df['Training Stress Score®'].apply(clean_num)
        df['Type'] = df['Tipo di attività'].astype(str)
        df['Dist_km'] = df['Distanza'].apply(clean_num)
        # Estrazione Ora per Cronobiologia
        df['Hour'] = df['Data'].dt.hour

        # --- ALGORITMO "SMART LOAD 2.0" ---
        def calculate_load(row):
            # A. Base Load (TSS Normalizzato)
            base_load = 0
            
            if row['TSS'] > 10:
                # Se abbiamo il TSS vero (Bici), usiamo quello
                base_load = row['TSS']
            else:
                # Se manca TSS (Corsa/Palestra), stimiamo: (Durata * TE) / 3
                # Il fattore /3 normalizza la scala a quella del TSS (approx)
                te_val = row['TE'] if row['TE'] > 0 else 2.5
                base_load = (row['Mins'] * te_val) / 3.0
            
            # B. Moltiplicatore Attività (CNS Impact)
            activity_mult = 1.0
            act_type = row['Type'].lower()
            if any(x in act_type for x in ['forza', 'palestra', 'pesi', 'crossfit', 'strength']):
                activity_mult = 1.5 # Boost neurale
            
            # C. Moltiplicatore Orario (Late Night Penalty)
            time_mult = 1.0
            hour = row['Hour']
            if hour >= 21:     # Dopo le 21:00
                time_mult = 1.20 # +20% Stress
            elif hour >= 18:   # Dopo le 18:00
                time_mult = 1.10 # +10% Stress
                
            return base_load * activity_mult * time_mult

        df['Load_Score'] = df.apply(calculate_load, axis=1)

        # Aggregazione Giornaliera
        daily_grp = df.groupby('Date_Day').agg({
            'Load_Score': 'sum',
            'Dist_km': 'sum',
            'Mins': 'sum'
        }).reset_index()
        
        daily_grp = daily_grp.rename(columns={'Date_Day': 'Date'})
        daily_grp['Date'] = pd.to_datetime(daily_grp['Date'])
        
        return daily_grp

    except Exception as e:
        st.error(f"Errore file Attività: {e}")
        return pd.DataFrame()

# --- GESTIONE DB ---
def load_db():
    cols = ['Date', 'rMSSD', 'RHR', 'Sleep', 'Feel', 'Status', 'Daily_Load', 'Daily_Dist', 'Daily_TrainTime']
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE, parse_dates=['Date'])
        for col in cols: 
            if col not in df.columns: df[col] = np.nan
        return df
    else: return pd.DataFrame(columns=cols)

def recalculate_status(df):
    df = df.sort_values('Date').reset_index(drop=True)
    new_statuses = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row['rMSSD']): 
            new_statuses.append("⚪ NO DATA")
            continue
            
        history = df.iloc[:i].tail(7)
        if len(history) < 3: 
            new_statuses.append("⚪ START")
            continue
            
        base_rmssd = history['rMSSD'].mean()
        rmssd = row['rMSSD']
        
        # Logica Semaforo
        if rmssd < base_rmssd * 0.85: 
            new_statuses.append("🔴 RIPOSO")
        elif rmssd < base_rmssd * 0.95 or (pd.notna(row['Feel']) and row['Feel'] < 6):
            new_statuses.append("🟡 CAUTELA")
        else:
            new_statuses.append("🟢 GO")
            
    df['Status'] = new_statuses
    return df

def update_db_generic(new_df, merge_cols):
    current_db = load_db()
    current_db['Date'] = pd.to_datetime(current_db['Date'])
    current_db['Date_Day'] = current_db['Date'].dt.date
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df['Date_Day'] = new_df['Date'].dt.date
    
    cnt_new, cnt_upd = 0, 0
    for _, row in new_df.iterrows():
        mask = current_db['Date_Day'] == row['Date_Day']
        if current_db[mask].empty:
            new_entry = {k: np.nan for k in current_db.columns if k not in ['Date', 'Date_Day']}
            new_entry['Date'] = row['Date']
            for c in merge_cols: 
                if c in row: new_entry[c] = row[c]
            current_db = pd.concat([current_db, pd.DataFrame([new_entry])], ignore_index=True)
            current_db['Date_Day'] = current_db['Date'].dt.date
            cnt_new += 1
        else:
            idx = current_db[mask].index[0]
            for c in merge_cols:
                if c in row and pd.notna(row[c]): current_db.at[idx, c] = row[c]
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
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("rMSSD", f"{last['rMSSD']} ms", delta_color="normal")
    
    # Tooltip spiegazione carico
    load_val = int(last['Daily_Load']) if pd.notna(last['Daily_Load']) else "--"
    k2.metric("Impact Load", load_val, help="TSS normalizzato + penalità oraria serale")
    
    k3.metric("Sonno", f"{last['Sleep']} h")
    k4.metric("Status", last['Status'])

    st.divider()

    t1, t2, t3 = st.tabs(["⚡ Carico vs HRV", "🌙 Recupero", "📝 Dati"])
    
    with t1:
        st.markdown("#### Stress vs Recupero")
        st.caption("Barre: Carico Impattante (tiene conto di orario e tipo sport). Linea: HRV.")
        
        # Grafico Carico vs HRV
        chart_df = df.set_index('Date').copy()
        
        # Normalizziamo per visualizzazione su stesso asse (Opzionale, ma aiuta)
        # Qui usiamo due grafici sovrapposti con assi indipendenti di Streamlit (limitati)
        # O colonne affiancate
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("**Carico Giornaliero (Barre)**")
            st.bar_chart(chart_df['Daily_Load'], color="#800080")
        with c2:
            st.markdown("**Andamento rMSSD**")
            st.line_chart(chart_df['rMSSD'], color="#0000FF")

    with t2:
        c_s1, c_s2 = st.columns(2)
        with c_s1: st.bar_chart(df.set_index('Date')['Sleep'], color="#2E8B57")
        with c_s2: st.line_chart(df.set_index('Date')['Feel'], color="#FFA500")

    with t3:
        st.dataframe(df.sort_values('Date', ascending=False))

else:
    st.info("Carica i file dalla sidebar per iniziare.")
