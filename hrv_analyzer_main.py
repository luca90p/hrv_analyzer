import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
from datetime import datetime, timedelta

# Tenta l'import di scipy per l'analisi in frequenza
try:
    from scipy import interpolate, signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="HRV Engineer Dashboard", layout="wide", page_icon="🫀")

st.title("🫀 HRV Engineer Dashboard")
st.markdown("### Monitoraggio Ingegneristico: Carico, Recupero & Analisi Spettrale")

DB_FILE = 'hrv_database.csv'

# --- 1. PARSING HRV AVANZATO ---
def parse_rr_file_advanced(file_content):
    try:
        content = file_content.decode("utf-8").splitlines()
        rr_intervals = []
        for line in content:
            line = line.strip()
            if line.isdigit():
                val = int(line)
                if 300 < val < 2000: rr_intervals.append(val)
        
        if len(rr_intervals) < 10: return None
        
        rr = np.array(rr_intervals)
        diffs = np.diff(rr)
        
        # --- TIME DOMAIN ---
        rmssd = np.sqrt(np.mean(diffs**2))
        ln_rmssd = np.log(rmssd) if rmssd > 0 else 0
        sdnn = np.std(rr, ddof=1)
        mean_rr = np.mean(rr)
        rhr = 60000 / mean_rr
        pnn50 = (np.sum(np.abs(diffs) > 50) / len(diffs)) * 100
        
        # --- FREQUENCY DOMAIN ---
        lf_power, hf_power, total_power, lf_hf = 0, 0, 0, 0
        
        if SCIPY_AVAILABLE and len(rr) > 30:
            try:
                # 1. Asse tempi cumulativo
                t_rr = np.cumsum(rr) / 1000.0
                t_rr = t_rr - t_rr[0]
                
                # 2. Interpolazione 4Hz
                fs = 4.0
                steps = np.arange(0, t_rr[-1], 1/fs)
                f_interp = interpolate.interp1d(t_rr, rr, kind='cubic', fill_value="extrapolate")
                rr_interp = f_interp(steps)
                
                # 3. Detrending
                rr_detrend = signal.detrend(rr_interp)
                
                # 4. Welch Periodogram
                freqs, psd = signal.welch(rr_detrend, fs=fs, nperseg=min(len(rr_detrend), 256))
                
                # 5. Integrazione Bande
                lf_band = (freqs >= 0.04) & (freqs < 0.15)
                hf_band = (freqs >= 0.15) & (freqs < 0.40)
                
                lf_power = np.trapz(psd[lf_band], freqs[lf_band])
                hf_power = np.trapz(psd[hf_band], freqs[hf_band])
                total_power = np.trapz(psd[(freqs >= 0) & (freqs < 0.4)], freqs[(freqs >= 0) & (freqs < 0.4)])
                
                lf_hf = lf_power / hf_power if hf_power > 0 else 0
                
            except Exception: pass
        
        return {
            'rMSSD': round(rmssd, 2),
            'ln_rMSSD': round(ln_rmssd, 2),
            'SDNN': round(sdnn, 2),
            'PNN50': round(pnn50, 1),
            'RHR': round(rhr, 1),
            'LF': round(lf_power, 0),
            'HF': round(hf_power, 0),
            'TotalPower': round(total_power, 0),
            'LF_HF': round(lf_hf, 2)
        }
    except Exception: return None

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

# --- 3. PARSING ATTIVITÀ ---
def parse_garmin_activities(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df['Data'] = pd.to_datetime(df['Data'])
        df['Date_Day'] = df['Data'].dt.date
        
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
        df['Type_Raw'] = df['Tipo di attività'].astype(str)
        df['Dist_km'] = df['Distanza'].apply(clean_num)
        df['Hour'] = df['Data'].dt.hour

        def get_category(s):
            s = s.lower()
            if 'corsa' in s: return 'Load_Corsa'
            if 'ciclismo' in s or 'bici' in s: return 'Load_Bici'
            return 'Load_Altro'

        df['Category'] = df['Type_Raw'].apply(get_category)

        def calculate_load(row):
            base_load = 0
            if row['TSS'] > 10: base_load = row['TSS']
            else:
                te_val = row['TE'] if row['TE'] > 0 else 2.5
                base_load = (row['Mins'] * te_val) / 3.0
            
            act_type = row['Type_Raw'].lower()
            activity_mult = 1.0
            if any(x in act_type for x in ['forza', 'palestra', 'pesi', 'crossfit']):
                activity_mult = 1.5 
            
            time_mult = 1.0
            if row['Hour'] >= 21: time_mult = 1.20
            elif row['Hour'] >= 18: time_mult = 1.10
                
            return base_load * activity_mult * time_mult

        df['Load_Score'] = df.apply(calculate_load, axis=1)

        pivot_df = df.pivot_table(index='Date_Day', columns='Category', values='Load_Score', aggfunc='sum', fill_value=0).reset_index()
        agg_total = df.groupby('Date_Day').agg({'Dist_km': 'sum', 'Mins': 'sum'}).reset_index()
        final_daily = pd.merge(pivot_df, agg_total, on='Date_Day')
        final_daily = final_daily.rename(columns={'Date_Day': 'Date'})
        final_daily['Date'] = pd.to_datetime(final_daily['Date'])
        
        return final_daily
    except Exception: return pd.DataFrame()

# --- GESTIONE DB ---
def load_db():
    cols = ['Date', 'rMSSD', 'RHR', 'Sleep', 'Feel', 'Status', 
            'Daily_Load', 'Load_Corsa', 'Load_Bici', 'Load_Altro', 
            'Daily_Dist', 'Daily_TrainTime',
            'ln_rMSSD', 'SDNN', 'PNN50', 'LF', 'HF', 'TotalPower', 'LF_HF']
    
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE, parse_dates=['Date'])
        for col in cols: 
            if col not in df.columns: 
                df[col] = 0.0 if 'Load_' in col else np.nan
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
        
        if rmssd < base_rmssd * 0.85: new_statuses.append("🔴 RIPOSO")
        elif rmssd < base_rmssd * 0.95 or (pd.notna(row['Feel']) and row['Feel'] < 6):
            new_statuses.append("🟡 CAUTELA")
        else: new_statuses.append("🟢 GO")
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
            for c in ['Load_Corsa', 'Load_Bici', 'Load_Altro']: new_entry[c] = 0.0
            for c in merge_cols: 
                if c in row: new_entry[c] = row[c]
            current_db = pd.concat([current_db, pd.DataFrame([new_entry])], ignore_index=True)
            current_db['Date_Day'] = current_db['Date'].dt.date
            cnt_new += 1
        else:
            idx = current_db[mask].index[0]
            for c in merge_cols:
                if c in row and pd.notna(row[c]):
                    if 'Load_' in c:
                        if row[c] > 0: current_db.at[idx, c] = row[c]
                    else: current_db.at[idx, c] = row[c]
            cnt_upd += 1
            
    cols_load = [c for c in ['Load_Corsa', 'Load_Bici', 'Load_Altro'] if c in current_db.columns]
    current_db['Daily_Load'] = current_db[cols_load].sum(axis=1)

    if 'Date_Day' in current_db.columns: current_db.drop(columns=['Date_Day'], inplace=True)
    final = recalculate_status(current_db)
    final.to_csv(DB_FILE, index=False)
    return cnt_new, cnt_upd

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 1. HRV")
    f_hrv = st.file_uploader("File TXT HRV", type=['txt'], accept_multiple_files=True)
    if f_hrv and st.button("Carica HRV"):
        data = []
        for f in f_hrv:
            dt = extract_date_from_filename(f.name)
            if dt:
                metrics = parse_rr_file_advanced(f.getvalue())
                if metrics:
                    entry = metrics
                    entry['Date'] = dt
                    data.append(entry)
        if data:
            cols = ['rMSSD', 'RHR', 'ln_rMSSD', 'SDNN', 'PNN50', 'LF', 'HF', 'TotalPower', 'LF_HF']
            n, u = update_db_generic(pd.DataFrame(data), cols)
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
    f_act = st.file_uploader("Activities.csv", type=['csv'], accept_multiple_files=True)
    if f_act and st.button("Carica Attività"):
        master_df = pd.DataFrame()
        for f in f_act:
            df_temp = parse_garmin_activities(f)
            if not df_temp.empty: master_df = pd.concat([master_df, df_temp])
        
        if not master_df.empty:
            master_df = master_df.groupby('Date').sum().reset_index()
            for c in ['Load_Corsa', 'Load_Bici', 'Load_Altro']:
                if c not in master_df.columns: master_df[c] = 0.0
            n, u = update_db_generic(master_df, ['Load_Corsa', 'Load_Bici', 'Load_Altro', 'Daily_Dist', 'Daily_TrainTime'])
            st.success(f"Attività: {n} nuovi, {u} agg.")
            st.rerun()

    # --- NUOVO: GESTIONE DATABASE ---
    st.markdown("---")
    st.header("⚙️ Gestione DB")
    
    # Inizializza stato conferma
    if 'confirm_reset' not in st.session_state:
        st.session_state.confirm_reset = False

    if st.button("🗑️ Pulisci Database"):
        st.session_state.confirm_reset = True

    if st.session_state.confirm_reset:
        st.warning("⚠️ Azione irreversibile. Sicuro?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ Conferma"):
                if os.path.exists(DB_FILE):
                    os.remove(DB_FILE)
                    st.success("Database cancellato.")
                st.session_state.confirm_reset = False
                st.rerun()
        with col_no:
            if st.button("❌ Annulla"):
                st.session_state.confirm_reset = False
                st.rerun()

# --- DASHBOARD ---
df = load_db()

if not df.empty:
    df = df.sort_values('Date')
    df['rMSSD_7d'] = df['rMSSD'].rolling(window=7, min_periods=1).mean()
    df['rMSSD_Response'] = df['rMSSD'].shift(-1)
    df['CV_7d'] = (df['rMSSD'].rolling(window=7, min_periods=3).std() / df['rMSSD'].rolling(window=7, min_periods=3).mean()) * 100
    
    last = df.iloc[-1]
    
    st.subheader(f"📊 Report: {last['Date'].strftime('%d/%m/%Y')}")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("rMSSD (Oggi)", f"{last['rMSSD']} ms", f"{last['rMSSD'] - last['rMSSD_7d']:.1f} vs Avg")
    k2.metric("Load (Oggi)", int(last['Daily_Load']) if pd.notna(last['Daily_Load']) else "--")
    k3.metric("Sonno (Stanotte)", f"{last['Sleep']} h")
    k4.metric("Status", last['Status'])

    st.divider()

    col_mode, col_info = st.columns([1, 3])
    with col_mode:
        view_mode = st.radio("Interpretazione:", ["🔋 Readiness (Default)", "📉 Response (Effect)"], index=0)
    with col_info:
        if view_mode == "🔋 Readiness (Default)":
            st.info("**Readiness**: Come stai stamattina prima di allenarti.")
            y_axis_hrv = 'rMSSD'
            color_hrv = 'black'
        else:
            st.info("**Response**: Come ha reagito il tuo corpo (HRV di domani) al carico di oggi.")
            y_axis_hrv = 'rMSSD_Response'
            color_hrv = '#d62728'

    t1, t2, t3, t4 = st.tabs(["⚡ HRV & Carico", "🌙 Recupero", "📝 Dati", "🔬 Laboratorio HRV"])
    
    with t1:
        chart_data = df.copy()
        chart_data = chart_data.rename(columns={'Load_Corsa': 'Corsa', 'Load_Bici': 'Bici', 'Load_Altro': 'Altro'})
        base = alt.Chart(chart_data).encode(x=alt.X('Date:T', axis=alt.Axis(format='%d/%m', title='')))
        
        melted_load = chart_data.melt(id_vars=['Date'], value_vars=['Corsa', 'Bici', 'Altro'], var_name='Sport', value_name='Load')
        bars = alt.Chart(melted_load).mark_bar().encode(
            x='Date:T', y=alt.Y('Load:Q', title='Impact Load'),
            color=alt.Color('Sport:N', scale=alt.Scale(domain=['Corsa', 'Bici', 'Altro'], range=['#d62728', '#1f77b4', '#7f7f7f'])),
            tooltip=['Date:T', 'Sport', 'Load']
        )
        line_hrv = base.mark_line(color=color_hrv, strokeWidth=3, point=True).encode(
            y=alt.Y(f'{y_axis_hrv}:Q', title='rMSSD (ms)', scale=alt.Scale(zero=False)),
            tooltip=['Date', f'{y_axis_hrv}']
        )
        upper = alt.layer(bars, line_hrv).resolve_scale(y='independent').properties(height=350)
        
        bars_sleep = base.mark_bar(color='#2ca02c', opacity=0.5).encode(
            y=alt.Y('Sleep:Q', title='Ore', scale=alt.Scale(domain=[4, 12])), tooltip=['Date', 'Sleep']
        )
        line_feel = base.mark_line(color='#ff7f0e', point=True).encode(
            y=alt.Y('Feel:Q', title='Feel (1-10)', scale=alt.Scale(domain=[1, 10])), tooltip=['Date', 'Feel']
        )
        lower = alt.layer(bars_sleep, line_feel).resolve_scale(y='independent').properties(height=150)
        
        st.altair_chart(alt.vconcat(upper, lower).resolve_scale(x='shared'), use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1: st.bar_chart(df.set_index('Date')['Sleep'], color="#2E8B57")
        with c2: st.line_chart(df.set_index('Date')['Feel'], color="#FFA500")

    with t3:
        st.dataframe(df.sort_values('Date', ascending=False))

    with t4:
        st.markdown("### 🔬 Analisi Statistica Avanzata")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("ln(rMSSD)", f"{last['ln_rMSSD']}", help="Logaritmo di rMSSD. Indice 'puro' HRV.")
        r1c2.metric("PNN50 %", f"{last['PNN50']}%", help="% battiti >50ms diff. Parasimpatico.")
        r1c3.metric("SDNN", f"{last['SDNN']} ms", help="Variabilità totale (Stress + Recupero).")
        r1c4.metric("CV (7gg)", f"{last['CV_7d']:.1f}%", help="Stabilità (<5% = Stabile).")

        st.divider()
        if 'LF' in df.columns and pd.notna(last['LF']) and last['TotalPower'] > 0:
            st.markdown("#### 📡 Frequency Domain")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Total Power", f"{int(last['TotalPower'])}", help="Energia totale.")
            f2.metric("LF", f"{int(last['LF'])}", help="Simpatico/Pressione.")
            f3.metric("HF", f"{int(last['HF'])}", help="Parasimpatico.")
            f4.metric("LF/HF", f"{last['LF_HF']}", help="Balance.")
        else:
            st.info("Dati spettrali non disponibili.")
            
        st.markdown("#### 📉 Coefficiente di Variazione")
        cv_chart = alt.Chart(df).mark_line(color='purple', point=True).encode(
            x='Date:T', y=alt.Y('CV_7d:Q', title='CV %'), tooltip=['Date', 'CV_7d']
        ).properties(height=250)
        rule = alt.Chart(pd.DataFrame({'y': [10]})).mark_rule(color='red', strokeDash=[3,3]).encode(y='y')
        st.altair_chart(cv_chart + rule, use_container_width=True)

else:
    st.info("Carica i file dalla sidebar.")
