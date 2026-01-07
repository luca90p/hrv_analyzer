import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
from datetime import datetime, timedelta

# Import Scipy (Analisi Spettro)
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

# --- 1. PARSING HRV ---
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
        
        # Time Domain
        rmssd = np.sqrt(np.mean(diffs**2))
        ln_rmssd = np.log(rmssd) if rmssd > 0 else 0
        sdnn = np.std(rr, ddof=1)
        mean_rr = np.mean(rr)
        rhr = 60000 / mean_rr
        pnn50 = (np.sum(np.abs(diffs) > 50) / len(diffs)) * 100
        
        # Frequency Domain
        lf_power, hf_power, total_power, lf_hf = 0, 0, 0, 0
        
        if SCIPY_AVAILABLE and len(rr) > 30:
            try:
                t_rr = np.cumsum(rr) / 1000.0
                t_rr = t_rr - t_rr[0]
                fs = 4.0 
                steps = np.arange(0, t_rr[-1], 1/fs)
                f_interp = interpolate.interp1d(t_rr, rr, kind='cubic', fill_value="extrapolate")
                rr_interp = f_interp(steps)
                rr_detrend = signal.detrend(rr_interp)
                freqs, psd = signal.welch(rr_detrend, fs=fs, nperseg=min(len(rr_detrend), 256))
                
                lf_band = (freqs >= 0.04) & (freqs < 0.15)
                hf_band = (freqs >= 0.15) & (freqs < 0.40)
                
                lf_power = np.trapz(psd[lf_band], freqs[lf_band])
                hf_power = np.trapz(psd[hf_band], freqs[hf_band])
                total_power = np.trapz(psd[(freqs >= 0) & (freqs < 0.4)], freqs[(freqs >= 0) & (freqs < 0.4)])
                lf_hf = lf_power / hf_power if hf_power > 0 else 0
            except: pass
        
        return {
            'rMSSD': round(rmssd, 2), 'ln_rMSSD': round(ln_rmssd, 2),
            'SDNN': round(sdnn, 2), 'PNN50': round(pnn50, 1),
            'RHR': round(rhr, 1), 'LF': round(lf_power, 0),
            'HF': round(hf_power, 0), 'TotalPower': round(total_power, 0),
            'LF_HF': round(lf_hf, 2)
        }
    except: return None

# --- 2. PARSING SONNO (CORRETTO) ---
def parse_garmin_sleep(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # Garmin esporta spesso la prima colonna con nomi strani, la rinominiamo in Date
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        
        # Cerchiamo la colonna durata in modo flessibile
        col_durata = None
        for c in df.columns:
            if 'durata' in c.lower() or 'tempo' in c.lower():
                col_durata = c
                break
        
        if not col_durata:
            st.error("Colonna durata non trovata nel file.")
            return pd.DataFrame()

        df['Date'] = pd.to_datetime(df['Date'])
        
        def clean_duration(val):
            if pd.isna(val): return 0.0
            val = str(val).lower().replace('h', '').replace('min', '').replace('hrs', '')
            parts = val.split()
            if len(parts) == 2: return round(float(parts[0]) + float(parts[1])/60, 2)
            elif len(parts) == 1: 
                # Gestione caso "7.5" o "7:30"
                if ':' in str(val):
                    p = str(val).split(':')
                    return round(float(p[0]) + float(p[1])/60, 2)
                return float(parts[0])
            return 0.0
            
        df['Sleep'] = df[col_durata].apply(clean_duration)
        
        # Mappa qualità
        # Cerchiamo colonna qualità
        col_qualita = 'Qualità'
        for c in df.columns: 
            if 'qualit' in c.lower(): col_qualita = c; break

        quality_map = {'Eccellente': 9, 'Buono': 8, 'Discreto': 6, 'Scarso': 4}
        if col_qualita in df.columns:
            df['Feel'] = df[col_qualita].map(quality_map).fillna(5)
        else:
            df['Feel'] = 5 
            
        # Restituisce le colonne con i nomi corretti per il DB
        return df[['Date', 'Sleep', 'Feel']]
    except Exception as e:
        st.error(f"Errore lettura Sonno: {e}")
        return pd.DataFrame()

# --- 3. PARSING ATTIVITÀ (CORRETTO) ---
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
            mult = 1.0
            if any(x in act_type for x in ['forza', 'palestra', 'pesi', 'crossfit']): mult = 1.5 
            
            t_mult = 1.0
            if row['Hour'] >= 21: t_mult = 1.20
            elif row['Hour'] >= 18: t_mult = 1.10
            return base_load * mult * t_mult

        df['Load_Score'] = df.apply(calculate_load, axis=1)
        
        # Pivot per Load
        piv = df.pivot_table(index='Date_Day', columns='Category', values='Load_Score', aggfunc='sum', fill_value=0).reset_index()
        # Totali
        agg = df.groupby('Date_Day').agg({'Dist_km': 'sum', 'Mins': 'sum'}).reset_index()
        
        final = pd.merge(piv, agg, on='Date_Day')
        final = final.rename(columns={'Date_Day': 'Date'})
        final['Date'] = pd.to_datetime(final['Date'])
        
        # RINOMINA PER MATCH DB
        final = final.rename(columns={'Dist_km': 'Daily_Dist', 'Mins': 'Daily_TrainTime'})
        
        return final
    except: return pd.DataFrame()

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
    stats = []
    for i in range(len(df)):
        r = df.iloc[i]
        if pd.isna(r['rMSSD']): 
            stats.append("⚪ NO DATA")
            continue
        hist = df.iloc[:i].tail(7)
        if len(hist) < 3: 
            stats.append("⚪ START")
            continue
        base = hist['rMSSD'].mean()
        curr = r['rMSSD']
        
        if curr < base * 0.85: stats.append("🔴 RIPOSO")
        elif curr < base * 0.95 or (pd.notna(r['Feel']) and r['Feel'] < 6): stats.append("🟡 CAUTELA")
        else: stats.append("🟢 GO")
    df['Status'] = stats
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
                # Controlla se la colonna esiste nel nuovo dato e non è nulla
                if c in row and pd.notna(row[c]):
                    if 'Load_' in c:
                        # Per i Load sommiamo o sovrascriviamo solo se > 0
                        if row[c] > 0: current_db.at[idx, c] = row[c]
                    else:
                        # Per il resto (es. Sleep, Feel) sovrascriviamo
                        current_db.at[idx, c] = row[c]
            cnt_upd += 1
            
    cols_load = [c for c in ['Load_Corsa', 'Load_Bici', 'Load_Altro'] if c in current_db.columns]
    current_db['Daily_Load'] = current_db[cols_load].sum(axis=1)

    if 'Date_Day' in current_db.columns: current_db.drop(columns=['Date_Day'], inplace=True)
    final = recalculate_status(current_db)
    final.to_csv(DB_FILE, index=False)
    return cnt_new, cnt_upd

# --- SIDEBAR ---
with st.sidebar:
    if not SCIPY_AVAILABLE: st.error("⚠️ Scipy non installato. Analisi LF/HF impossibile.")
    
    st.header("📂 1. HRV")
    f_hrv = st.file_uploader("File TXT HRV", type=['txt'], accept_multiple_files=True)
    if f_hrv and st.button("Carica HRV"):
        data = []
        for f in f_hrv:
            dt = extract_date_from_filename(f.name)
            if dt:
                m = parse_rr_file_advanced(f.getvalue())
                if m:
                    m['Date'] = dt
                    data.append(m)
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

    st.markdown("---")
    if 'confirm_reset' not in st.session_state: st.session_state.confirm_reset = False
    if st.button("🗑️ Pulisci DB"): st.session_state.confirm_reset = True
    if st.session_state.confirm_reset:
        st.warning("Sicuro?")
        c1, c2 = st.columns(2)
        if c1.button("Sì"):
            if os.path.exists(DB_FILE): os.remove(DB_FILE)
            st.session_state.confirm_reset = False
            st.rerun()
        if c2.button("No"):
            st.session_state.confirm_reset = False
            st.rerun()

# --- DASHBOARD ---
df = load_db()

if not df.empty:
    df = df.sort_values('Date')
    
    # 1. CALCOLI PER VISUALIZZAZIONE
    df['rMSSD_7d'] = df['rMSSD'].rolling(window=7, min_periods=1).mean()
    df['std_7d'] = df['rMSSD'].rolling(window=7, min_periods=3).std()

    # NEW: SDNN Trend (Rolling) per evitare regression bug
    if 'SDNN' in df.columns:
        df['SDNN_7d'] = df['SDNN'].rolling(window=7, min_periods=1).mean()
    else:
        df['SDNN_7d'] = np.nan
        df['SDNN'] = np.nan
    
    # Banda di Normalità (Soft Background)
    df['zone_min'] = df['rMSSD_7d'] - (0.75 * df['std_7d'])
    df['zone_max'] = df['rMSSD_7d'] + (0.75 * df['std_7d'])
    
    # Shift per Response
    df['rMSSD_Response'] = df['rMSSD'].shift(-1)
    df['CV_7d'] = (df['std_7d'] / df['rMSSD_7d']) * 100
    
    # Colore Status per i Punti (Mapping Esplicito per Altair)
    def get_status_color(status):
        if 'GO' in status: return '#00E676'      # Verde Neon
        if 'CAUTELA' in status: return '#FFEA00' # Giallo Acceso
        return '#FF1744'                         # Rosso Acceso
    
    df['Status_Color'] = df['Status'].apply(get_status_color)
    
    last = df.iloc[-1]

    # --- INIZIO MODIFICA: FILTRO INTELLIGENTE DATA ---
    # Troviamo la prima data in cui c'è l'HRV
    first_hrv = df[df['rMSSD'] > 0]['Date'].min()
    # Troviamo la prima data in cui c'è il Sonno
    first_sleep = df[df['Sleep'] > 0]['Date'].min()
    
    # Calcoliamo la data di partenza (Intersezione)
    # Se esistono entrambi, prendiamo la data più recente tra le due partenze
    if pd.notna(first_hrv) and pd.notna(first_sleep):
        start_date = max(first_hrv, first_sleep)
    elif pd.notna(first_hrv):
        start_date = first_hrv
    elif pd.notna(first_sleep):
        start_date = first_sleep
    else:
        start_date = df['Date'].min()
        
    # Creiamo il dataset filtrato SOLO per la visualizzazione
    df_viz = df[df['Date'] >= start_date].copy()
    # --- FINE MODIFICA ---
    
    # --- HEADER & KPI ---
    st.subheader(f"📊 Report: {last['Date'].strftime('%d %B %Y')}")
    
    # Banner Status Pulito
    status_bg = get_status_color(last['Status'])
    st.markdown(f"""
    <div style="
        padding: 15px; 
        border-radius: 12px; 
        background: linear-gradient(90deg, {status_bg}20 0%, rgba(255,255,255,0) 100%);
        border-left: 6px solid {status_bg}; 
        margin-bottom: 25px;">
        <h2 style="margin:0; color: {status_bg}; text-shadow: 0px 0px 1px rgba(0,0,0,0.2);">{last['Status']}</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        k1, k2, k3, k4 = st.columns(4)
        diff = last['rMSSD'] - last['rMSSD_7d']
        arrow = "⬆️" if diff > 1 else "⬇️" if diff < -1 else "➡️"
        
        k1.metric("rMSSD", f"{int(last['rMSSD'])} ms", f"{diff:.1f} vs Baseline {arrow}")
        k2.metric("Strain (Load)", int(last['Daily_Load']) if pd.notna(last['Daily_Load']) else "--")
        k3.metric("Sonno", f"{last['Sleep']} h" if pd.notna(last['Sleep']) else "--")
        k4.metric("Stabilità (CV)", f"{last['CV_7d']:.1f}%", help="<5% Ottimo")

    st.divider()

    # --- CONTROLLI ---
    c_mode, c_legend = st.columns([1, 4])
    with c_mode:
        view_mode = st.radio("Modo:", ["Readiness (AM)", "Response (Effect)"], horizontal=True, label_visibility="collapsed")
    
    if view_mode == "Readiness (AM)":
        y_target = 'rMSSD'
        line_col_hex = '#263238' # Antracite
    else:
        y_target = 'rMSSD_Response'
        line_col_hex = '#D32F2F' # Rosso Scuro (Effect)

    # --- TABS GRAFICI ---
    t1, t2, t3, t4 = st.tabs(["⚡ Performance", "🌙 Sleep", "📝 Data", "🔬 Lab"])
    
    with t1:
        # Prep Data
        chart_data = df_viz.copy()
        chart_data = chart_data.rename(columns={'Load_Corsa': 'Corsa', 'Load_Bici': 'Bici', 'Load_Altro': 'Altro'})
        
        base = alt.Chart(chart_data).encode(x=alt.X('Date:T', axis=alt.Axis(format='%d/%m', title=None, grid=False, domain=False)))

        # 1. BANDA NORMALITÀ (Sfondo Tenue)
        band = base.mark_area(opacity=0.4, color='#ECEFF1').encode(
            y=alt.Y('zone_min:Q', title='rMSSD (ms)'),
            y2='zone_max:Q'
        )

        # 2. CARICO (Barre Pastello - Sfondo)
        melted_load = chart_data.melt(id_vars=['Date'], value_vars=['Corsa', 'Bici', 'Altro'], var_name='Sport', value_name='Load')
        bars = alt.Chart(melted_load).mark_bar(width=12, opacity=0.8, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x='Date:T',
            y=alt.Y('Load:Q', axis=alt.Axis(title='Strain Load', grid=False, labels=False, ticks=False)),
            color=alt.Color('Sport:N', legend=alt.Legend(orient='top', title=None), 
                            scale=alt.Scale(
                                domain=['Corsa', 'Bici', 'Altro'],
                                range=['#EF9A9A', '#90CAF9', '#B0BEC5'] # Rosso Pastello, Blu Pastello, Grigio
                            )),
            tooltip=['Date', 'Sport', 'Load']
        )

        # 3. LINEA HRV (Netta e Scura)
        line = base.mark_line(color=line_col_hex, strokeWidth=2.5, interpolate='monotone').encode(
            y=f'{y_target}:Q'
        )

        # 4. PUNTI STATUS (Brillanti e Bordati)
        points = base.mark_circle(size=100, opacity=1, stroke='white', strokeWidth=2).encode(
            y=f'{y_target}:Q',
            color=alt.Color('Status_Color:N', scale=None), # Usa i colori esadecimali diretti
            tooltip=['Date', f'{y_target}', 'Status']
        )

        # Assemblaggio
        final_chart = (band + bars + line + points).resolve_scale(y='independent').properties(height=420)
        st.altair_chart(final_chart, use_container_width=True)
        st.caption("🟦 Fascia Grigia: Zona di Normalità. 📊 Barre Pastello: Carico Allenante. ● Punti: Semaforo Recupero.")

    with t2:
        # GRAFICO SONNO (Area Sfumata)
        st.markdown("#### Qualità del Sonno")
        
        base_s = alt.Chart(df_viz).encode(x=alt.X('Date:T', axis=alt.Axis(format='%d/%m', title=None, grid=False)))
        
        area = base_s.mark_area(
            line={'color':'#66BB6A'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='rgba(102, 187, 106, 0.5)', offset=0),
                       alt.GradientStop(color='rgba(102, 187, 106, 0.05)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            y=alt.Y('Sleep:Q', scale=alt.Scale(domain=[4, 11]), axis=alt.Axis(title='Ore', grid=True, gridDash=[2,2])),
            tooltip=['Date', 'Sleep']
        )
        
        # Linea Feel (Giallo Brillante)
        line_f = base_s.mark_line(color='#FFD600', strokeWidth=3, strokeDash=[4,2]).encode(
            y=alt.Y('Feel:Q', scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(title='Feel')),
            tooltip=['Date', 'Feel']
        )
        
        st.altair_chart((area + line_f).resolve_scale(y='independent').properties(height=280), use_container_width=True)

    with t3:
        st.dataframe(
            df[['Date', 'rMSSD', 'Status', 'Daily_Load', 'Sleep', 'Feel']].sort_values('Date', ascending=False),
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                "rMSSD": st.column_config.NumberColumn("rMSSD", format="%d ms"),
                "Daily_Load": st.column_config.ProgressColumn("Carico", format="%d", min_value=0, max_value=400),
                "Status": st.column_config.TextColumn("Stato"),
            }
        )

    with t4:
        st.markdown("## 🔬 Laboratorio Analisi HRV")
        
        # --- SEZIONE 1: DOMINIO DEL TEMPO (Resilienza) ---
        st.markdown("### 1. Resilienza Totale (SDNN)")
        with st.expander("📘 Guida: rMSSD vs SDNN"):
            st.markdown("""
            * **rMSSD (Il Meccanico):** Indica l'attività parasimpatica a *breve termine*. È quanto velocemente il tuo corpo riesce a "frenare" il cuore beat-to-beat. È l'indice del recupero immediato.
            * **SDNN (Il Serbatoio):** È la deviazione standard di *tutti* i battiti. Rappresenta la **capacità totale** del tuo sistema nervoso di rispondere agli stress. 
                * *SDNN Alto:* Grande riserva di energia adattiva (Sei resiliente).
                * *SDNN Basso cronico:* Rischio burnout o sovrallenamento strutturale.
            """)
        
        c1, c2 = st.columns(2)
        c1.metric("SDNN (Oggi)", f"{last['SDNN']} ms", help="Target > 50-100ms a seconda dell'età")
        c2.metric("PNN50", f"{last['PNN50']}%", help="% Battiti che differiscono >50ms. Indice puro di tono vagale.")
        
        # Grafico SDNN con linea di tendenza
        base_sdnn = alt.Chart(df_viz).encode(x='Date:T')
        line_sdnn = base_sdnn.mark_line(color='#AB47BC', strokeWidth=3).encode(
            y=alt.Y('SDNN:Q', title='SDNN (ms)', scale=alt.Scale(zero=False)),
            tooltip=['Date', 'SDNN']
        )
        trend_sdnn = base_sdnn.mark_line(
            color='white', 
            opacity=0.5, 
            strokeDash=[5,5]
        ).encode(
            y=alt.Y('SDNN_7d:Q')
        )
        
        st.altair_chart((line_sdnn + trend_sdnn).properties(height=250), use_container_width=True)

        st.divider()

        # --- SEZIONE 2: DOMINIO DELLA FREQUENZA (Bilanciamento) ---
        st.markdown("### 2. Spettro di Potenza (LF vs HF)")
        with st.expander("📘 Guida: Gas vs Freno (LF/HF)"):
            st.markdown("""
            L'analisi spettrale scompone il segnale cardiaco in frequenze:
            * **LF (Low Frequency - 0.04-0.15Hz):** [Immagine del sistema nervoso simpatico] Associato al sistema **Simpatico** (Lotta o Fuga) e alla regolazione della pressione sanguigna. Se alto, sei "attivato".
            * **HF (High Frequency - 0.15-0.4Hz):** [Immagine del sistema nervoso parasimpatico] Associato al sistema **Parasimpatico** (Riposo e Respiro). Se alto, stai recuperando.
            * **Ratio LF/HF:** Indica il bilanciamento. 
                * *Target:* Basso (es. 1.0 - 2.0) a riposo.
                * *Alto (>3.0):* Stress predominante.
            """)
            
        if 'LF' in df.columns and pd.notna(last['LF']):
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Power", f"{int(last['TotalPower'])} ms²", help="Energia totale del sistema.")
            m2.metric("LF (Stress/BP)", f"{int(last['LF'])}", delta_color="inverse") # Se sale troppo è male
            m3.metric("HF (Recupero)", f"{int(last['HF'])}")
            
            # --- CODICE GRAFICO CORRETTO ---
            # Filtriamo solo le righe dove LF e HF sono > 0 per evitare errori grafici
            valid_spectra = df_viz[(df_viz['LF'] > 0) & (df_viz['HF'] > 0)].copy()
            
            if not valid_spectra.empty:
                spectra_data = valid_spectra[['Date', 'LF', 'HF']].melt('Date', var_name='Band', value_name='Power')
                
                area_spectra = alt.Chart(spectra_data).mark_area(opacity=0.6).encode(
                    x=alt.X('Date:T', axis=alt.Axis(format='%d/%m')),
                    y=alt.Y('Power:Q', stack='normalize', title='Dominanza % (Normalizzata)'),
                    color=alt.Color('Band:N', 
                                    scale=alt.Scale(domain=['LF', 'HF'], range=['#FF7043', '#42A5F5']),
                                    legend=alt.Legend(title="Banda Frequenza")),
                    tooltip=[
                        alt.Tooltip('Date:T', format='%d %B'), 
                        alt.Tooltip('Band', title='Tipo'), 
                        alt.Tooltip('Power', format='.0f', title='Potenza (ms²)')
                    ]
                ).properties(height=300)
                
                st.altair_chart(area_spectra, use_container_width=True)
                st.caption("🟥 Arancione: Simpatico (LF) | 🟦 Blu: Parasimpatico (HF). A riposo vorremmo vedere più Blu.")
            else:
                st.warning("Ci sono dati HRV, ma il calcolo spettrale ha prodotto zeri. Verifica che i file .txt abbiano abbastanza battiti (>300).")

        st.divider()

        # --- SEZIONE 3: ACCOPPIAMENTO CARDIACO ---
        st.markdown("### 3. Correlazione rMSSD vs RHR")
        with st.expander("📘 Guida: Saturazione Parasimpatica"):
            st.markdown("""
            Normalmente, **rMSSD e RHR sono inversi**:
            * rMSSD sale ⬆️ -> RHR scende ⬇️ (Buon segno: recupero).
            * rMSSD scende ⬇️ -> RHR sale ⬆️ (Cattivo segno: stress).
            
            **⚠️ Il Pericolo (Saturazione):**
            Se vedi **rMSSD basso ⬇️ E RHR basso ⬇️**, attenzione! Potrebbe essere "Saturazione Parasimpatica" o "Exhaustion". Il corpo è così stanco che non riesce nemmeno ad alzare i battiti. È un campanello d'allarme per l'overtraining.
            """)
        
        # Grafico a doppio asse sincronizzato
        base_corr = alt.Chart(df_viz).encode(x='Date:T')
        
        line_rmssd = base_corr.mark_line(color='#263238').encode(
            y=alt.Y('rMSSD:Q', title='rMSSD (ms)')
        )
        
        line_rhr = base_corr.mark_line(color='#d62728', strokeDash=[5,5]).encode(
            y=alt.Y('RHR:Q', title='RHR (bpm)', scale=alt.Scale(zero=False))
        )
        
        st.altair_chart(alt.layer(line_rmssd, line_rhr).resolve_scale(y='independent').properties(height=300), use_container_width=True)
        st.caption("⚫ Linea Nera: rMSSD | 🔴 Linea Rossa Tratteggiata: Battiti a Riposo (RHR).")

else:
    st.info("👋 Database vuoto. Carica i file dalla sidebar.")
