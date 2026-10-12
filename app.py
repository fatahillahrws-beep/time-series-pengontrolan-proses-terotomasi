"""
app.py — Dashboard Streamlit: Perbandingan 9 Metode Prediksi Multivariat NPK Phonska IV
Jalankan:  streamlit run app.py
"""
import os, io, glob, json, inspect
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import engine as eng

# =============================================================================
# KONFIGURASI DEFAULT  (ubah di sini, atau lewat panel "Konfigurasi" di sidebar)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = dict(
    APP_TITLE='NPK Phonska IV — Forecasting Intelligence',
    # Lokasi dataset di repo GitHub (dicari berurutan; file pertama yang ditemukan dipakai)
    DATA_CANDIDATES=['data/NPK_Phonska_IV_Imputasi_ML.xlsx', 'data/*Phonska*.xlsx', 'data/*.xlsx',
                     'NPK_Phonska_IV_Imputasi_ML.xlsx', '*Phonska*.xlsx'],
    TIME_COL='Time',
    VARS=['H2O', 'N', 'P2O5', 'K2O'],
    WINDOWS=[3, 6, 12],
    TRAIN_RATIO=0.8,
    SEED=42,
    SELECT_BY='RMSE_avg',            # 'RMSE_avg' atau 'MAPE_avg'
    ALPHA=0.00273, LAMBDA_MEWMA=0.20, LAMBDA_MIN=0.05, LAMBDA_MAX=0.30, GAMMA=0.50,
)
# Preset kecepatan: "Standar" = persis notebook asli; "Cepat" = untuk demo/uji coba
PRESETS = {
    'Standar (sesuai notebook)': dict(rf_iter=30, dl_epochs=100, dl_patience=10, dl_units=[32, 64], dl_lrs=[0.001, 0.01],
                                      n_cal=1000, max_rl=500, svr_C=[1, 10, 100], svr_gamma=[0.001, 0.01, 0.1], svr_eps=[0.01, 0.1]),
    'Cepat (demo)': dict(rf_iter=8, dl_epochs=30, dl_patience=5, dl_units=[32], dl_lrs=[0.001],
                         n_cal=300, max_rl=300, svr_C=[1, 10], svr_gamma=[0.01, 0.1], svr_eps=[0.1]),
}

METHOD_COLOR = {'VAR': '#6C5CE7', 'SVR': '#00B894', 'Random Forest': '#E17055', 'XGBoost': '#0984E3',
                'LightGBM': '#F5B700', 'LSTM': '#E84393', 'RNN': '#636E72', 'GRU': '#00CEC9', 'TCN': '#D63031'}
VAR_COLOR = ['#2563EB', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#14B8A6']
INK, GOOD, WARN = '#1F2D3D', '#10B981', '#EF4444'
SCALE_GOOD_LOW = [[0, '#0F9D76'], [0.5, '#F7E08A'], [1, '#E4572E']]   # rendah = bagus (hijau)

st.set_page_config(page_title=CONFIG['APP_TITLE'], page_icon='🌾', layout='wide', initial_sidebar_state='expanded')

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
.hero { background: linear-gradient(120deg, #0B3954 0%, #087E8B 55%, #2FBF9B 100%); border-radius: 22px;
        padding: 30px 36px; color: white; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(8,126,139,.25); position: relative; overflow: hidden; }
.hero:after { content: ''; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
              background: rgba(255,255,255,.08); }
.hero h1 { margin: 0 0 6px 0; font-size: 2.0rem; font-weight: 800; letter-spacing: -.5px; color: white; }
.hero p { margin: 0; opacity: .92; font-size: 1.0rem; max-width: 900px; }
.chip { display: inline-block; padding: 3px 12px; border-radius: 999px; background: rgba(255,255,255,.18); font-size: .78rem;
        margin: 10px 6px 0 0; font-weight: 600; }
.kpi { background: white; border: 1px solid #E6ECF2; border-radius: 16px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(20,40,80,.05);
       height: 100%; }
.kpi-l { font-size: .74rem; text-transform: uppercase; letter-spacing: .08em; color: #6B7A8C; font-weight: 600; }
.kpi-v { font-size: 1.65rem; font-weight: 800; color: #0B3954; line-height: 1.25; margin-top: 2px; }
.kpi-s { font-size: .8rem; color: #6B7A8C; margin-top: 2px; }
.interp { background: #F0FAF7; border-left: 5px solid #10B981; border-radius: 10px; padding: 12px 16px; margin: 6px 0 26px 0;
          color: #1F2D3D; font-size: .93rem; line-height: 1.6; }
.interp b.t { color: #087E8B; }
.note { background: #FFF8E6; border-left: 5px solid #F5B700; border-radius: 10px; padding: 12px 16px; margin: 6px 0 18px 0; font-size: .92rem; }
.sec { font-size: 1.18rem; font-weight: 700; color: #0B3954; margin: 22px 0 2px 0; }
.sub { color: #6B7A8C; font-size: .9rem; margin-bottom: 8px; }
.step { background: white; border: 1px solid #E6ECF2; border-radius: 14px; padding: 14px 16px; height: 100%; }
.step .n { width: 30px; height: 30px; border-radius: 50%; background: #087E8B; color: white; font-weight: 700; display: inline-flex;
           align-items: center; justify-content: center; margin-bottom: 6px; }
.step b { color: #0B3954; } .step span { color: #5A6B7D; font-size: .88rem; }
div[data-testid="stTabs"] button { font-weight: 600; }
section[data-testid="stSidebar"] { background: #F7FAFC; border-right: 1px solid #E6ECF2; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER UI
# =============================================================================
def _stretch(fn):
    return {'use_container_width': True} if 'use_container_width' in inspect.signature(fn).parameters else {'width': 'stretch'}


PLOT_CFG = {'displaylogo': False, 'toImageButtonOptions': {'scale': 2}}


def plot(fig, key):
    st.plotly_chart(fig, key=key, config=PLOT_CFG, **_stretch(st.plotly_chart))


def table(df, cfg=None, height=None):
    kw = dict(hide_index=True, column_config=cfg or {}, **_stretch(st.dataframe))
    if height: kw['height'] = height
    st.dataframe(df, **kw)


def section(title, sub=''):
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="sub">{sub}</div>' if sub else ''), unsafe_allow_html=True)


def interp(*lines):
    st.markdown('<div class="interp"><b class="t">💡 Interpretasi</b><br>' + '<br>'.join(lines) + '</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub=''):
    col.markdown(f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>',
                 unsafe_allow_html=True)


def style(fig, title=None, height=460, legend=True):
    fig.update_layout(template='plotly_white', height=height, font=dict(family='Inter, Segoe UI, sans-serif', size=13, color=INK),
                      title=dict(text=title, x=0.01, font=dict(size=17)) if title else None,
                      margin=dict(l=10, r=10, t=64 if title else 24, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      hoverlabel=dict(bgcolor='white', font_size=12), showlegend=legend,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right'))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(120,130,150,.15)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(120,130,150,.15)', zeroline=False)
    return fig


def hex_rgba(h, a):
    h = h.lstrip('#'); return f'rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})'


def play_layout(fig, names, labels, duration=60, redraw=False, transition=0, easing='linear'):
    """Tambah tombol Play/Pause + slider ke figure beranimasi."""
    fr = dict(duration=duration, redraw=redraw)
    tr = dict(duration=transition, easing=easing)
    fig.update_layout(
        margin=dict(b=120),
        updatemenus=[dict(type='buttons', direction='left', x=0.0, y=-0.14, xanchor='left', yanchor='top', showactive=False,
                          bgcolor='#EAF6F4', bordercolor='#087E8B', font=dict(color='#087E8B'),
                          buttons=[dict(label='▶ Play', method='animate', args=[None, dict(frame=fr, transition=tr, fromcurrent=True, mode='immediate')]),
                                   dict(label='⏸ Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode='immediate')])])],
        sliders=[dict(active=0, x=0.17, y=-0.14, len=0.82, xanchor='left', yanchor='top', pad=dict(t=0, b=0),
                      currentvalue=dict(prefix='▸ ', font=dict(size=12, color='#087E8B')), ticklen=0, minorticklen=0,
                      font=dict(size=1, color='rgba(0,0,0,0)'), bgcolor='#EAF6F4', activebgcolor='#087E8B',
                      steps=[dict(method='animate', label=l, args=[[n], dict(mode='immediate', frame=dict(duration=0, redraw=redraw), transition=dict(duration=0))])
                             for n, l in zip(names, labels)])])
    return fig


def frame_points(n, max_frames):
    step = max(1, int(np.ceil(n / max_frames)))
    ks = list(range(step, n + 1, step))
    if not ks or ks[-1] != n: ks.append(n)
    return ks


def tstr(x):
    return pd.to_datetime(pd.Series(x)).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()


def animated_lines(x, series, colors, title=None, ytitle='', widths=None, dashes=None, max_frames=90, duration=55,
                   shade=None, height=520, fill_first=False):
    """Garis yang 'tumbuh' dari kiri ke kanan; titik kepala bergerak; slider waktu."""
    xl = tstr(x); n = len(xl); names = list(series); m = len(names)
    ks = frame_points(n, max_frames); k0 = ks[0]
    widths, dashes = widths or {}, dashes or {}
    data = [go.Scatter(x=xl[:k0], y=series[nm][:k0], mode='lines', name=nm, legendgroup=nm,
                       line=dict(color=colors[nm], width=widths.get(nm, 2), dash=dashes.get(nm, 'solid')),
                       fill='tozeroy' if (fill_first and i == 0) else None,
                       fillcolor=hex_rgba(colors[nm], .10) if (fill_first and i == 0) else None) for i, nm in enumerate(names)]
    data += [go.Scatter(x=[xl[k0 - 1]], y=[series[nm][k0 - 1]], mode='markers', legendgroup=nm, showlegend=False, hoverinfo='skip',
                        marker=dict(color=colors[nm], size=10, line=dict(color='white', width=2))) for nm in names]
    frames = [go.Frame(name=str(k), traces=list(range(2 * m)),
                       data=[go.Scatter(x=xl[:k], y=series[nm][:k]) for nm in names] +
                            [go.Scatter(x=[xl[k - 1]], y=[series[nm][k - 1]]) for nm in names]) for k in ks]
    fig = go.Figure(data=data, frames=frames)
    allv = np.concatenate([np.asarray(v, float) for v in series.values()]); pad = (allv.max() - allv.min()) * .08 or 1
    fig.update_xaxes(type='date', range=[xl[0], xl[-1]])
    fig.update_yaxes(range=[allv.min() - pad, allv.max() + pad], title=ytitle)
    if shade: fig.add_vrect(x0=shade[0], x1=xl[-1], fillcolor='#F59E0B', opacity=.08, line_width=0, annotation_text=shade[1],
                            annotation_position='top left')
    style(fig, title, height)
    return play_layout(fig, [str(k) for k in ks], [xl[k - 1][:16] for k in ks], duration)


def animated_control(x, s, sig, ucl, title, color, max_frames=90, duration=40, height=430):
    xl = tstr(x); n = len(xl); ks = frame_points(n, max_frames); k0 = ks[0]
    s = np.asarray(s, float); sig = np.asarray(sig, bool)
    xa = np.array(xl)
    data = [go.Scatter(x=xl[:k0], y=s[:k0], mode='lines', name='Statistik', line=dict(color=color, width=2),
                       fill='tozeroy', fillcolor=hex_rgba(color, .08)),
            go.Scatter(x=xa[:k0][sig[:k0]], y=s[:k0][sig[:k0]], mode='markers', name='Out-of-control',
                       marker=dict(color=WARN, size=9, line=dict(color='white', width=1.5))),
            go.Scatter(x=[xl[0], xl[-1]], y=[ucl, ucl], mode='lines', name=f'UCL = {ucl:.2f}', line=dict(color=WARN, dash='dash', width=1.6))]
    frames = [go.Frame(name=str(k), traces=[0, 1],
                       data=[go.Scatter(x=xl[:k], y=s[:k]), go.Scatter(x=xa[:k][sig[:k]], y=s[:k][sig[:k]])]) for k in ks]
    fig = go.Figure(data=data, frames=frames)
    fig.update_xaxes(type='date', range=[xl[0], xl[-1]]); fig.update_yaxes(range=[0, max(s.max(), ucl) * 1.12], title='Statistik')
    style(fig, title, height)
    return play_layout(fig, [str(k) for k in ks], [xl[k - 1][:16] for k in ks], duration)


def fmt(x, d=3): return f'{x:,.{d}f}'
def ac1(e): return float(np.corrcoef(e[:-1], e[1:])[0, 1])


# =============================================================================
# DATA
# =============================================================================
def find_data_files():
    found = []
    for pat in CONFIG['DATA_CANDIDATES']:
        for p in sorted(glob.glob(os.path.join(BASE_DIR, pat))):
            n = os.path.basename(p)
            if n.startswith('~$') or 'Hasil' in n or 'Perbandingan' in n: continue
            if p not in found: found.append(p)
    return found


@st.cache_data(show_spinner=False)
def load_df(raw: bytes, name: str, time_col: str):
    bio = io.BytesIO(raw)
    df = pd.read_csv(bio) if name.lower().endswith('.csv') else pd.read_excel(bio)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.sort_values(time_col).reset_index(drop=True)


# =============================================================================
# SIDEBAR: DATA + KONFIGURASI
# =============================================================================
def sidebar():
    with st.sidebar:
        st.markdown('### 🌾 NPK Phonska IV')
        st.caption('Dashboard perbandingan 9 metode prediksi multivariat + control chart')
        st.markdown('#### ⚙️ Konfigurasi')
        mode = st.selectbox('Preset kecepatan', list(PRESETS), key='preset',
                            help='"Standar" memakai pengaturan notebook asli (±20–30 menit). "Cepat" untuk uji coba (jauh lebih singkat).')
        P = PRESETS[mode]; k = lambda n: f'{n}|{mode}'

        files = find_data_files(); raw = name = None
        with st.expander('📂 Sumber data', expanded=not files):
            if files:
                choice = st.selectbox('Dataset di repo', files, format_func=lambda p: os.path.relpath(p, BASE_DIR))
                with open(choice, 'rb') as f: raw, name = f.read(), os.path.basename(choice)
                st.caption('✅ Dataset terbaca otomatis dari repo.')
            up = st.file_uploader('…atau unggah file lain (.xlsx / .csv)', type=['xlsx', 'csv'])
            if up is not None: raw, name = up.getvalue(), up.name
            if raw is None: st.warning('Dataset tidak ditemukan. Taruh file di folder `data/` pada repo atau unggah di sini.')
        if raw is None: return None, None, None, False
        try:
            df = load_df(raw, name, CONFIG['TIME_COL'])
        except Exception as e:
            st.error(f'Gagal membaca data: {e}'); return None, None, None, False
        num_cols = [c for c in df.columns if c != CONFIG['TIME_COL'] and pd.api.types.is_numeric_dtype(df[c])]

        with st.form('cfg_form'):
            with st.expander('📐 Data & pembagian', expanded=True):
                vars_ = st.multiselect('Variabel', num_cols, default=[v for v in CONFIG['VARS'] if v in num_cols] or num_cols[:4])
                windows = st.multiselect('Sliding window', [2, 3, 4, 5, 6, 8, 9, 12, 18, 24], default=CONFIG['WINDOWS'])
                ratio = st.slider('Proporsi data latih', 0.60, 0.90, CONFIG['TRAIN_RATIO'], 0.05)
                seed = st.number_input('Random seed', 0, 99999, CONFIG['SEED'])
                select_by = st.selectbox('Kriteria window/konfigurasi terbaik', ['RMSE_avg', 'MAPE_avg'],
                                         index=['RMSE_avg', 'MAPE_avg'].index(CONFIG['SELECT_BY']))
            with st.expander('🧪 Metode yang dijalankan', expanded=False):
                methods = st.multiselect('Metode', eng.METHOD_ORDER, default=eng.METHOD_ORDER)
            with st.expander('🔧 Hyperparameter & tuning', expanded=False):
                rf_iter = st.slider('Random Forest — iterasi RandomizedSearch', 3, 60, P['rf_iter'], key=k('rf'))
                svr_C = st.multiselect('SVR — C', [0.1, 1, 10, 100, 1000], default=P['svr_C'], key=k('C'))
                svr_g = st.multiselect('SVR — gamma', [0.0001, 0.001, 0.01, 0.1, 1], default=P['svr_gamma'], key=k('g'))
                svr_e = st.multiselect('SVR — epsilon', [0.01, 0.05, 0.1, 0.2], default=P['svr_eps'], key=k('e'))
                dl_units = st.multiselect('Deep learning — units', [16, 32, 64, 128], default=P['dl_units'], key=k('u'))
                dl_lrs = st.multiselect('Deep learning — learning rate', [0.0005, 0.001, 0.005, 0.01], default=P['dl_lrs'], key=k('lr'))
                dl_drop = st.slider('Deep learning — dropout', 0.0, 0.5, 0.2, 0.05)
                dl_ep = st.slider('Deep learning — maks epoch', 10, 300, P['dl_epochs'], 10, key=k('ep'))
                dl_pat = st.slider('Deep learning — patience (early stopping)', 3, 30, P['dl_patience'], key=k('pat'))
                var_max = st.slider('VAR — lag maksimum untuk AIC/BIC', 5, 40, 30)
            with st.expander('🚦 Control chart', expanded=False):
                alpha = st.number_input('α (laju alarm palsu)', 0.0001, 0.05, CONFIG['ALPHA'], format='%.5f')
                lam = st.slider('λ MEWMA', 0.05, 0.60, CONFIG['LAMBDA_MEWMA'], 0.01)
                lmin = st.slider('λ min (Adaptive-DHWMA)', 0.01, 0.30, CONFIG['LAMBDA_MIN'], 0.01)
                lmax = st.slider('λ max (Adaptive-DHWMA)', 0.05, 0.80, CONFIG['LAMBDA_MAX'], 0.01)
                gam = st.slider('γ adaptif', 0.05, 2.0, CONFIG['GAMMA'], 0.05)
                n_cal = st.select_slider('Replikasi Monte Carlo', [200, 300, 500, 1000, 2000], value=P['n_cal'], key=k('ncal'))
                max_rl = st.select_slider('Panjang maksimum run (MAX_RL)', [200, 300, 500, 800], value=P['max_rl'], key=k('mrl'))
            submitted = st.form_submit_button('🚀 Jalankan Analisis', type='primary')

        cfg = dict(vars=vars_, time_col=CONFIG['TIME_COL'], windows=sorted(windows), train_ratio=float(ratio), seed=int(seed),
                   select_by=select_by, run={m: (m in methods) for m in eng.METHOD_ORDER}, rf_iter=int(rf_iter),
                   svr_C=svr_C, svr_gamma=svr_g, svr_eps=svr_e, dl_units=dl_units, dl_lrs=dl_lrs, dl_dropout=float(dl_drop),
                   dl_epochs=int(dl_ep), dl_patience=int(dl_pat), var_maxlags=int(var_max), alpha=float(alpha),
                   lambda_mewma=float(lam), lambda_min=float(lmin), lambda_max=float(lmax), gamma=float(gam),
                   n_cal=int(n_cal), max_rl=int(max_rl))
        st.caption('Ubah pengaturan lalu tekan **Jalankan Analisis**. Hasil dengan konfigurasi yang sama disimpan di cache.')
        return df, name, cfg, submitted


def validate(cfg):
    errs = []
    if len(cfg['vars']) < 2: errs.append('Pilih minimal 2 variabel (analisis multivariat).')
    if not cfg['windows']: errs.append('Pilih minimal 1 sliding window.')
    if not any(cfg['run'].values()): errs.append('Pilih minimal 1 metode.')
    if cfg['lambda_min'] >= cfg['lambda_max']: errs.append('λ min harus lebih kecil dari λ max.')
    if cfg['run']['SVR'] and not (cfg['svr_C'] and cfg['svr_gamma'] and cfg['svr_eps']): errs.append('Grid SVR tidak boleh kosong.')
    dl = [m for m in ['LSTM', 'RNN', 'GRU', 'TCN'] if cfg['run'][m]]
    if dl and not (cfg['dl_units'] and cfg['dl_lrs']): errs.append('Grid deep learning (units & learning rate) tidak boleh kosong.')
    return errs


# =============================================================================
# TAB 1 — BERANDA
# =============================================================================
def tab_home(df, cfg, res):
    if res is None:
        note('👈 Dataset sudah terbaca. Atur <b>Konfigurasi</b> di sidebar lalu tekan <b>🚀 Jalankan Analisis</b> untuk melatih semua metode. '
             'Sambil menunggu, tab <b>Eksplorasi Data</b> dan <b>Metode & Dampak</b> sudah bisa dipakai.')
    else:
        B = res['BEST']; top = B.iloc[0]
        c = st.columns(5)
        kpi(c[0], 'Metode terbaik', top.Metode, f'window {top.Window} • {top.Kategori}')
        kpi(c[1], 'RMSE rata-rata', fmt(top.RMSE_avg, 4), 'skala asli, 4 variabel')
        kpi(c[2], 'MAPE rata-rata', f'{top.MAPE_avg:.3f}%', 'data uji')
        kpi(c[3], 'Data latih / uji', f"{res['N_TRAIN']} / {res['N'] - res['N_TRAIN']}", f"total {res['N']} observasi")
        kpi(c[4], 'Waktu komputasi', f"{res['elapsed']/60:.1f} mnt", f"{len(B)} metode × {len(res['cfg']['windows'])} window")
        st.write('')
        for w in res['warnings']: st.warning(w)
    section('Alur analisis', 'Dari data mentah hingga pemantauan kualitas prediksi')
    steps = [('Data & Split', 'Deret 4 jam dibagi kronologis (mis. 80:20). Data uji identik untuk semua metode.'),
             ('Sliding window', 'Input = w observasi sebelumnya, target = 4 variabel pada waktu t (one-step-ahead).'),
             ('Latih 9 metode', 'VAR, SVR, RF, XGBoost, LightGBM, LSTM, RNN, GRU, TCN + tuning sesuai kategori.'),
             ('Evaluasi', 'RMSE & MAPE pada skala asli, rata-rata 4 variabel; peringkat gabungan.'),
             ('Control chart', 'Residual uji dimonitor dengan MEWMA & Adaptive-DHWMA (UCL Monte Carlo).')]
    cols = st.columns(5)
    for i, (c, (a, b)) in enumerate(zip(cols, steps), 1):
        c.markdown(f'<div class="step"><div class="n">{i}</div><br><b>{a}</b><br><span>{b}</span></div>', unsafe_allow_html=True)
    st.write('')
    section('Konfigurasi aktif')
    show = cfg if res is None else res['cfg']
    with st.expander('Lihat seluruh parameter yang dipakai'):
        st.json(show)
    if res is not None:
        interp(f"Dengan konfigurasi ini, <b>{top.Metode}</b> (window {top.Window}) menghasilkan galat terkecil secara gabungan. "
               f"Ringkasan lengkap peringkat, grafik animasi, dan penjelasan tiap metode ada di tab-tab berikutnya.")


# =============================================================================
# TAB 2 — EKSPLORASI DATA
# =============================================================================
def tab_eda(df, cfg):
    from statsmodels.tsa.stattools import acf, adfuller
    V, tc = cfg['vars'], CONFIG['TIME_COL']
    n = len(df); nt = int(cfg['train_ratio'] * n); t_split = str(df[tc].iloc[nt])
    c = st.columns(4)
    kpi(c[0], 'Observasi', f'{n:,}', 'interval 4 jam'); kpi(c[1], 'Periode', f"{df[tc].min():%d %b %Y}", f"s.d. {df[tc].max():%d %b %Y}")
    kpi(c[2], 'Variabel', str(len(V)), ', '.join(V)); kpi(c[3], 'Missing value', str(int(df[V].isna().sum().sum())), 'setelah imputasi')
    st.write('')

    section('Cuplikan & statistik deskriptif')
    c1, c2 = st.columns([3, 2])
    with c1: table(df[[tc] + V].head(200), {tc: st.column_config.DatetimeColumn('Waktu', format='YYYY-MM-DD HH:mm'),
                                           **{v: st.column_config.NumberColumn(v, format='%.3f') for v in V}}, height=300)
    with c2:
        d = df[V].describe().T.reset_index().rename(columns={'index': 'Variabel'})
        table(d, {k: st.column_config.NumberColumn(k, format='%.3f') for k in d.columns if k != 'Variabel'}, height=300)
    sd_shift = {v: (df[v].iloc[nt:].mean() - df[v].iloc[:nt].mean()) / df[v].iloc[:nt].std() for v in V}
    big = max(sd_shift, key=lambda k: abs(sd_shift[k]))
    interp(f'Tabel kiri memperlihatkan 200 observasi pertama, tabel kanan merangkum sebaran tiap variabel (rata-rata, simpangan baku, kuartil). '
           f'Perbedaan skala antarvariabel itu sebabnya RMSE dihitung per variabel lalu dirata-ratakan, dan RMSE ternormalisasi (dibagi SD data latih) disediakan sebagai pembanding bebas-skala. '
           f'Pergeseran rata-rata data uji terhadap data latih terbesar ada pada <b>{big}</b>: {sd_shift[big]:+.2f} SD.')

    section('🎬 Putar ulang deret waktu', 'Tekan ▶ Play atau geser slider; area oranye = data uji')
    v = st.selectbox('Variabel', V, key='eda_v')
    fig = animated_lines(df[tc], {v: df[v].values}, {v: VAR_COLOR[V.index(v) % 6]}, ytitle=v, widths={v: 2.2},
                         shade=(t_split, 'Data uji'), fill_first=True, duration=45)
    plot(fig, 'eda_anim')
    interp(f'Garis tumbuh mengikuti waktu sehingga perubahan level, lonjakan, dan pola berulang pada <b>{v}</b> mudah terlihat. '
           f'Wilayah oranye adalah {n-nt} observasi terakhir yang <i>tidak pernah dilihat</i> model saat latihan; bila perilaku di area ini berbeda dari area latih, model akan lebih sulit memprediksinya.')

    section('Semua variabel pada skala yang sama (z-score)', 'Gunakan range slider di bawah grafik untuk memperbesar periode tertentu')
    Z = (df[V] - df[V].iloc[:nt].mean()) / df[V].iloc[:nt].std()
    fig = go.Figure([go.Scatter(x=df[tc], y=Z[x], name=x, mode='lines', line=dict(width=1.4, color=VAR_COLOR[i % 6])) for i, x in enumerate(V)])
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=.07),
                     rangeselector=dict(buttons=[dict(count=7, label='7 hr', step='day', stepmode='backward'),
                                                 dict(count=1, label='1 bln', step='month', stepmode='backward'), dict(step='all', label='Semua')]))
    fig.add_vline(x=t_split, line_dash='dot', line_color='#F59E0B')
    plot(style(fig, height=470), 'eda_z')
    interp('Setelah distandardisasi (memakai mean & SD data latih), variabel yang bergerak serempak tampak sejajar dan yang bergerak berlawanan saling menjauh. '
           'Garis titik-titik oranye adalah batas latih/uji. Kesamaan pola naik-turun antarvariabel menjadi alasan memakai model <b>multivariat</b>.')

    c1, c2 = st.columns(2)
    with c1:
        section('Korelasi antarvariabel')
        C = df[V].corr().values
        fig = go.Figure(go.Heatmap(z=C, x=V, y=V, zmin=-1, zmax=1, colorscale='RdBu_r', text=np.round(C, 2), texttemplate='%{text}',
                                   colorbar=dict(thickness=10)))
        fig.update_yaxes(autorange='reversed'); plot(style(fig, height=400, legend=False), 'eda_corr')
        iu = np.triu_indices(len(V), 1); k = np.argmax(np.abs(C[iu]))
        interp(f'Pasangan paling berkorelasi: <b>{V[iu[0][k]]}–{V[iu[1][k]]}</b> (r = {C[iu][k]:+.2f}). '
               f'Korelasi kuat berarti informasi satu variabel membantu memprediksi variabel lain; korelasi lemah berarti hubungan lebih banyak ditentukan lag variabel itu sendiri.')
    with c2:
        section('Autokorelasi (ACF) — memori deret')
        vv = st.selectbox('Variabel ACF', V, key='acf_v')
        a = acf(df[vv].iloc[:nt], nlags=40, fft=True); ci = 1.96 / np.sqrt(nt)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, 41)), y=a[1:], marker=dict(color=np.where(np.abs(a[1:]) > ci, '#087E8B', '#B8C4D0')), name='ACF'))
        fig.add_hrect(y0=-ci, y1=ci, fillcolor='#F59E0B', opacity=.12, line_width=0)
        fig.update_xaxes(title='Lag (×4 jam)'); fig.update_yaxes(title='ACF')
        plot(style(fig, height=345, legend=False), 'eda_acf')
        nsig = int((np.abs(a[1:]) > ci).sum())
        interp(f'ACF lag-1 <b>{vv}</b> = {a[1]:.2f}; {nsig} dari 40 lag melewati batas signifikansi (pita oranye). '
               + ('Autokorelasi tinggi → nilai masa lalu sangat informatif, sliding window pendek pun sudah efektif.' if a[1] > .7 else 'Autokorelasi sedang/rendah → deret relatif “bising”, galat prediksi akan lebih besar.'))

    section('Peta panas harian × jam', 'Pola siklus harian pada pengukuran tiap 4 jam')
    vh = st.selectbox('Variabel peta panas', V, key='cal_v')
    t = df[[tc, vh]].copy(); t['Tanggal'] = t[tc].dt.date; t['Jam'] = t[tc].dt.strftime('%H:%M')
    pv = t.pivot_table(index='Jam', columns='Tanggal', values=vh, aggfunc='mean')
    fig = go.Figure(go.Heatmap(z=pv.values, x=[str(x) for x in pv.columns], y=pv.index, colorscale='Viridis', colorbar=dict(title=vh, thickness=10)))
    plot(style(fig, height=330, legend=False), 'eda_cal')
    hm = pv.mean(1)
    interp(f'Setiap kolom satu hari, setiap baris satu slot jam. Rata-rata <b>{vh}</b> tertinggi pada slot <b>{hm.idxmax()}</b> ({hm.max():.2f}) dan terendah pada <b>{hm.idxmin()}</b> ({hm.min():.2f}). '
           f'Warna yang seragam sepanjang baris menandakan perilaku harian stabil; blok warna berbeda menandakan pergeseran proses/perubahan kondisi.')

    section('Sebaran data latih vs uji')
    fig = make_subplots(rows=1, cols=len(V), subplot_titles=V)
    for i, x in enumerate(V, 1):
        fig.add_trace(go.Violin(y=df[x].iloc[:nt], name='Latih', side='negative', line_color='#087E8B', fillcolor=hex_rgba('#087E8B', .35),
                                meanline_visible=True, showlegend=(i == 1), legendgroup='a'), row=1, col=i)
        fig.add_trace(go.Violin(y=df[x].iloc[nt:], name='Uji', side='positive', line_color='#F59E0B', fillcolor=hex_rgba('#F59E0B', .35),
                                meanline_visible=True, showlegend=(i == 1), legendgroup='b'), row=1, col=i)
    fig.update_layout(violinmode='overlay')
    plot(style(fig, height=380), 'eda_violin')
    interp('Violin kiri (teal) = data latih, kanan (oranye) = data uji, garis = rata-rata. Bila bentuk dan posisinya berbeda jauh, terjadi <i>distribution shift</i>: '
           'metode berbasis pohon (RF, XGBoost, LightGBM) tidak dapat mengekstrapolasi di luar rentang yang pernah dilihat, sedangkan model linear/VAR dan jaringan saraf lebih fleksibel.')

    section('Uji stasioneritas (ADF) pada data latih')
    k_adf = int(np.trunc((nt - 1) ** (1 / 3))); rows = []
    for x in V:
        st_, p_ = adfuller(df[x].iloc[:nt], maxlag=k_adf, regression='ct', autolag=None)[:2]
        rows.append({'Variabel': x, 'Statistik ADF': st_, 'p-value': p_, 'Keputusan': 'Stasioner' if p_ < .05 else 'Tidak stasioner'})
    A = pd.DataFrame(rows)
    table(A, {'Statistik ADF': st.column_config.NumberColumn(format='%.3f'), 'p-value': st.column_config.NumberColumn(format='%.4f')})
    ns = A[A['p-value'] >= .05].Variabel.tolist()
    interp('Hipotesis nol ADF: deret memiliki akar unit (tidak stasioner); p < 0,05 → stasioner. ' +
           ('Semua variabel stasioner, sehingga asumsi utama VAR terpenuhi tanpa differencing.' if not ns else
            f'Variabel <b>{", ".join(ns)}</b> belum stasioner; hasil VAR perlu dibaca hati-hati (kode R asli berhenti pada kondisi ini).'))


# =============================================================================
# TAB 3 — PERBANDINGAN METODE
# =============================================================================
def need_res(res):
    if res is None:
        note('Belum ada hasil. Tekan <b>🚀 Jalankan Analisis</b> di sidebar untuk melatih model.'); return True
    return False


def tab_compare(res):
    if need_res(res): return
    B, ALL, cfg = res['BEST'], res['ALL'], res['cfg']; W = cfg['windows']; V = res['VARS']
    top = B.iloc[0]; sel = cfg['select_by']

    section('🏆 Papan peringkat', 'Window terbaik dipilih per metode berdasarkan ' + sel)
    L = B[['No', 'Metode', 'Kategori', 'Window', 'RMSE_avg', 'MAPE_avg', 'RMSE_norm', 'Skor', 'Peringkat_RMSE', 'Peringkat_MAPE', 'Peringkat_Gabungan', 'Waktu_s']].copy()
    L['Metode'] = [('🏆 ' if i == 0 else '') + m for i, m in enumerate(L.Metode)]
    table(L, {'RMSE_avg': st.column_config.NumberColumn('RMSE', format='%.4f'), 'MAPE_avg': st.column_config.NumberColumn('MAPE (%)', format='%.3f'),
              'RMSE_norm': st.column_config.NumberColumn('RMSE norm.', format='%.3f', help='RMSE / SD data latih; <1 berarti lebih baik daripada menebak rata-rata'),
              'Skor': st.column_config.ProgressColumn('Skor (100 = terbaik)', min_value=0, max_value=100, format='%.1f'),
              'Peringkat_RMSE': 'Rank RMSE', 'Peringkat_MAPE': 'Rank MAPE', 'Peringkat_Gabungan': 'Rank gabungan',
              'Waktu_s': st.column_config.NumberColumn('Waktu (dtk)', format='%.1f')})
    cat = B.groupby('Kategori').RMSE_avg.min().idxmin(); r2 = B.iloc[1] if len(B) > 1 else None
    interp(f"Secara gabungan (rata-rata peringkat RMSE & MAPE), <b>{top.Metode}</b> (window {top.Window}) unggul dengan RMSE {fmt(top.RMSE_avg,4)} dan MAPE {top.MAPE_avg:.3f}%."
           + (f" Selisih RMSE dengan peringkat 2 (<b>{r2.Metode}</b>) sebesar {100*(r2.RMSE_avg/top.RMSE_avg-1):.1f}%." if r2 is not None else '')
           + f" Kategori dengan RMSE terkecil: <b>{cat}</b>. "
           + (f"RMSE ternormalisasi {top.RMSE_norm:.2f} (&lt;1) berarti model lebih baik daripada sekadar menebak rata-rata data latih." if top.RMSE_norm < 1
              else f"RMSE ternormalisasi {top.RMSE_norm:.2f} (≥1) berarti model belum lebih baik daripada menebak rata-rata data latih."))

    metric = st.radio('Metrik grafik', ['RMSE_avg', 'MAPE_avg', 'RMSE_norm'], horizontal=True, key='race_metric',
                      format_func=lambda x: {'RMSE_avg': 'RMSE rata-rata', 'MAPE_avg': 'MAPE rata-rata (%)', 'RMSE_norm': 'RMSE ternormalisasi'}[x])
    A = ALL[ALL.Window.isin(W)]
    section('🎬 Balapan metode per window', 'Tekan ▶ — urutan batang berganti mengikuti window; batang terpendek = galat terkecil')
    xmax = A[metric].max() * 1.18
    def bars(w):
        d = A[A.Window == w].sort_values(metric)
        return go.Bar(y=d.Metode, x=d[metric], orientation='h', text=[f'{x:.3f}' for x in d[metric]], textposition='outside',
                      marker=dict(color=[METHOD_COLOR[m] for m in d.Metode], line=dict(color='white', width=1)))
    fig = go.Figure(data=[bars(W[0])], frames=[go.Frame(name=str(w), data=[bars(w)], traces=[0]) for w in W])
    fig.update_xaxes(range=[0, xmax]); fig.update_yaxes(autorange='reversed')
    style(fig, height=470, legend=False)
    plot(play_layout(fig, [str(w) for w in W], [f'Window = {w}' for w in W], duration=1100, redraw=True, transition=700, easing='cubic-in-out'), 'race')
    wm = A.groupby('Window')[metric].mean()
    interp(f'Rata-rata {metric} seluruh metode paling kecil pada <b>window {wm.idxmin()}</b> ({wm.min():.3f}) dan paling besar pada window {wm.idxmax()} ({wm.max():.3f}). '
           'Window terlalu kecil membuat model kekurangan konteks, window terlalu besar menambah dimensi input (lebih rentan overfitting pada metode berbasis fitur lag).')

    c1, c2 = st.columns(2)
    with c1:
        section('🎬 Peta kompromi RMSE vs MAPE', 'Gelembung besar = skor lebih baik')
        def bub(w):
            d = A[A.Window == w]; sc = 100 * d.RMSE_avg.min() / d.RMSE_avg
            return go.Scatter(x=d.RMSE_avg, y=d.MAPE_avg, mode='markers+text', text=d.Metode, textposition='top center',
                              marker=dict(size=14 + .38 * sc, color=[METHOD_COLOR[m] for m in d.Metode], opacity=.85, line=dict(color='white', width=2)),
                              hovertemplate='%{text}<br>RMSE %{x:.4f}<br>MAPE %{y:.3f}%<extra></extra>')
        fig = go.Figure(data=[bub(W[0])], frames=[go.Frame(name=str(w), data=[bub(w)], traces=[0]) for w in W])
        fig.update_xaxes(title='RMSE rata-rata', range=[A.RMSE_avg.min() * .9, A.RMSE_avg.max() * 1.1])
        fig.update_yaxes(title='MAPE rata-rata (%)', range=[A.MAPE_avg.min() * .85, A.MAPE_avg.max() * 1.12])
        style(fig, height=500, legend=False)
        plot(play_layout(fig, [str(w) for w in W], [f'Window = {w}' for w in W], duration=1100, redraw=True, transition=700, easing='cubic-in-out'), 'bubble')
        interp('Titik di <b>kiri-bawah</b> = RMSE dan MAPE sama-sama kecil (ideal). Bila sebuah metode rendah di RMSE tetapi tinggi di MAPE, ia akurat pada nilai besar namun relatif kurang akurat pada nilai kecil (MAPE sensitif terhadap nilai aktual yang kecil).')
    with c2:
        section('Peta panas metode × window')
        piv = ALL[ALL.Window.isin(W)].pivot_table(index='Metode', columns='Window', values=metric).reindex(B.Metode)
        fig = go.Figure(go.Heatmap(z=piv.values, x=[f'w = {c}' for c in piv.columns], y=piv.index, colorscale=SCALE_GOOD_LOW,
                                   text=np.round(piv.values, 3), texttemplate='%{text}', colorbar=dict(thickness=10)))
        fig.update_yaxes(autorange='reversed'); plot(style(fig, height=500, legend=False), 'heat_mw')
        bi = np.unravel_index(np.nanargmin(piv.values), piv.shape)
        interp(f'Sel hijau = galat kecil, merah = galat besar. Kombinasi terbaik di seluruh tabel: <b>{piv.index[bi[0]]}</b> pada <b>window {piv.columns[bi[1]]}</b> ({np.nanmin(piv.values):.3f}). '
               'Baris yang warnanya berubah drastis antar-kolom adalah metode yang sensitif terhadap pilihan window.')

    c1, c2 = st.columns(2)
    with c1:
        section('Radar skor relatif per variabel')
        pick = st.multiselect('Metode', B.Metode.tolist(), default=B.Metode.head(4).tolist(), key='radar_m')
        fig = go.Figure()
        for m in pick:
            r = B[B.Metode == m].iloc[0]
            sc = [100 * B[f'RMSE_{v}'].min() / r[f'RMSE_{v}'] for v in V]
            fig.add_trace(go.Scatterpolar(r=sc + sc[:1], theta=V + V[:1], name=m, fill='toself', line=dict(color=METHOD_COLOR[m], width=2.5),
                                          fillcolor=hex_rgba(METHOD_COLOR[m], .12)))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix='')))
        plot(style(fig, height=430), 'radar')
        interp('Setiap sumbu = satu variabel; nilai 100 berarti metode itu terbaik untuk variabel tersebut. Bentuk yang melebar ke segala arah menandakan metode konsisten; bentuk yang “penyok” di satu sumbu berarti metode itu lemah pada variabel tersebut.')
    with c2:
        section('Bump chart — peringkat per variabel')
        rk = pd.DataFrame({v: B[f'RMSE_{v}'].rank(method='min') for v in V}); rk.index = B.Metode
        fig = go.Figure()
        for m in rk.index:
            fig.add_trace(go.Scatter(x=V, y=rk.loc[m], mode='lines+markers+text', name=m, text=[m if i == len(V) - 1 else '' for i in range(len(V))],
                                     textposition='middle right', line=dict(color=METHOD_COLOR[m], width=4), marker=dict(size=14, line=dict(color='white', width=2))))
        fig.update_yaxes(autorange='reversed', dtick=1, title='Peringkat (1 = terbaik)'); fig.update_xaxes(range=[-.2, len(V) - .35])
        plot(style(fig, height=430, legend=False), 'bump')
        pv_ = res['PER_VAR']
        interp('Garis yang naik-turun menunjukkan metode terbaik berbeda untuk tiap variabel. ' +
               ' '.join(f"{r.Variabel}: <b>{r.Terbaik_RMSE.split(' (')[0]}</b>." for r in pv_.itertuples()))

    c1, c2 = st.columns(2)
    with c1:
        section('Treemap kategori → metode', 'Ukuran = skor, warna = RMSE')
        fig = px.treemap(B, path=[px.Constant('Semua'), 'Kategori', 'Metode'], values='Skor', color='RMSE_avg', color_continuous_scale=SCALE_GOOD_LOW)
        fig.update_traces(textinfo='label+value', texttemplate='<b>%{label}</b><br>%{value:.0f}')
        plot(style(fig, height=400, legend=False), 'tree')
        g = B.groupby('Kategori').agg(rmse=('RMSE_avg', 'min'), n=('Metode', 'count'))
        interp('Kotak lebih besar = skor lebih tinggi. ' + ' '.join(f"{k}: RMSE terbaik {r.rmse:.3f} ({int(r.n)} metode)." for k, r in g.iterrows()))
    with c2:
        section('Pengaruh sliding window per metode')
        pr = res['PIV_RMSE']
        fig = go.Figure()
        for m in pr.index:
            fig.add_trace(go.Scatter(x=[str(c) for c in pr.columns], y=pr.loc[m], mode='lines+markers', name=m,
                                     line=dict(color=METHOD_COLOR[m], width=2.5), marker=dict(size=9)))
        fig.update_xaxes(title='Sliding window'); fig.update_yaxes(title='RMSE rata-rata')
        plot(style(fig, height=400), 'winline')
        sens = (pr.max(1) - pr.min(1)) / pr.min(1) * 100
        interp(f'Metode paling sensitif terhadap window: <b>{sens.idxmax()}</b> (selisih terbaik–terburuk {sens.max():.1f}%); paling stabil: <b>{sens.idxmin()}</b> ({sens.min():.1f}%). '
               'Garis hampir datar = pilihan window tidak kritis; garis curam = window perlu dituning.')

    section('Metode terbaik per variabel')
    pv_ = res['PER_VAR']
    table(pv_, {'RMSE': st.column_config.NumberColumn(format='%.4f'), 'MAPE (%)': st.column_config.NumberColumn(format='%.3f')})
    interp('Tidak selalu ada satu metode yang menang di semua variabel. Jika pemenang berbeda-beda, pertimbangkan model terpisah per variabel atau ensemble untuk produksi.')

    c1, c2 = st.columns(2)
    with c1:
        section('RMSE & MAPE per variabel (konfigurasi terbaik)')
        long = B.melt(id_vars=['Metode'], value_vars=[f'RMSE_{v}' for v in V], var_name='Var', value_name='RMSE'); long['Var'] = long.Var.str[5:]
        fig = px.bar(long, x='Var', y='RMSE', color='Metode', barmode='group', color_discrete_map=METHOD_COLOR)
        fig.update_xaxes(title=''); plot(style(fig, height=400), 'bar_var')
        interp('Skala RMSE berbeda antarvariabel (mengikuti satuan masing-masing), jadi bandingkan tinggi batang <i>antarmetode dalam satu variabel</i>, bukan antarvariabel.')
    with c2:
        section('Waktu komputasi')
        d = B.sort_values('Waktu_s')
        fig = go.Figure(go.Bar(y=d.Metode, x=d.Waktu_s, orientation='h', marker=dict(color=[METHOD_COLOR[m] for m in d.Metode]),
                               text=[f'{t:.1f}s' for t in d.Waktu_s], textposition='outside'))
        fig.update_xaxes(title='detik (termasuk tuning)'); plot(style(fig, height=400, legend=False), 'time')
        fast = d.iloc[0]; slow = d.iloc[-1]
        interp(f'Tercepat: <b>{fast.Metode}</b> ({fast.Waktu_s:.1f} dtk); terlama: <b>{slow.Metode}</b> ({slow.Waktu_s:.1f} dtk). '
               'Waktu mencakup tuning (grid/random search) sehingga metode tanpa tuning (XGBoost, LightGBM) relatif ringan.')


# =============================================================================
# TAB 4 — PREDIKSI & RESIDUAL
# =============================================================================
def tab_pred(res):
    if need_res(res): return
    B, V, Y, T, P = res['BEST'], res['VARS'], res['Y_TEST'], res['T_TEST'], res['PREDS']
    c1, c2 = st.columns([1, 3])
    v = c1.selectbox('Variabel', V, key='pv')
    meths = c2.multiselect('Metode (window terbaik masing-masing)', B.Metode.tolist(), default=B.Metode.head(3).tolist(), key='pm')
    if not meths: note('Pilih minimal satu metode.'); return
    j = V.index(v); win = {m: int(B.loc[B.Metode == m, 'Window'].iloc[0]) for m in meths}
    lab = {m: f'{m} (w={win[m]})' for m in meths}
    yp = {m: P[(m, win[m])][:, j] for m in meths}; E = {m: Y[:, j] - yp[m] for m in meths}
    colors = {'Aktual': '#1F2D3D', **{lab[m]: METHOD_COLOR[m] for m in meths}}

    section('🎬 Aktual vs prediksi — putar ulang data uji', 'Titik bulat mengikuti waktu; garis hitam tebal = aktual')
    series = {'Aktual': Y[:, j], **{lab[m]: yp[m] for m in meths}}
    fig = animated_lines(T, series, colors, ytitle=v, widths={'Aktual': 3.2}, dashes={lab[m]: 'solid' for m in meths}, duration=45)
    plot(fig, 'pred_anim')
    best_m = min(meths, key=lambda m: np.sqrt(np.mean(E[m] ** 2)))
    interp(f'Semakin menempel garis prediksi pada garis hitam, semakin akurat. Untuk <b>{v}</b>, metode terdekat adalah <b>{lab[best_m]}</b> '
           f'(RMSE {np.sqrt(np.mean(E[best_m]**2)):.4f}). Perhatikan juga titik balik (puncak/lembah): prediksi satu-langkah-ke-depan sering terlambat satu periode karena sangat bergantung pada nilai terakhir.')

    section('🎬 RMSE kumulatif — siapa yang paling stabil sepanjang waktu uji?')
    cum = {lab[m]: np.sqrt(np.cumsum(E[m] ** 2) / np.arange(1, len(E[m]) + 1)) for m in meths}
    plot(animated_lines(T, cum, {k: colors[k] for k in cum}, ytitle=f'RMSE kumulatif {v}', widths={k: 2.6 for k in cum}, duration=45, height=470), 'cum')
    fin = {k: x[-1] for k, x in cum.items()}; jump = {k: x.max() / x[-1] for k, x in cum.items()}
    interp(f'Nilai akhir menyamai RMSE keseluruhan. Kurva yang melonjak mendadak menandakan periode dengan galat besar (kejadian ekstrem/perubahan proses). '
           f'Terendah di akhir: <b>{min(fin, key=fin.get)}</b> ({min(fin.values()):.4f}).')

    c1, c2 = st.columns(2)
    with c1:
        section('Aktual vs prediksi (scatter)')
        fig = go.Figure(); lo, hi = Y[:, j].min(), Y[:, j].max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', line=dict(color='#9AA8B8', dash='dash'), name='Prediksi sempurna'))
        for m in meths:
            r2 = 1 - np.sum(E[m] ** 2) / np.sum((Y[:, j] - Y[:, j].mean()) ** 2)
            fig.add_trace(go.Scatter(x=Y[:, j], y=yp[m], mode='markers', name=f'{m} (R²={r2:.3f})', marker=dict(color=METHOD_COLOR[m], size=6, opacity=.55)))
        fig.update_xaxes(title='Aktual'); fig.update_yaxes(title='Prediksi')
        plot(style(fig, height=430), 'scat')
        interp('Titik yang rapat di sekitar garis putus-putus = prediksi akurat. Pola melengkung berarti bias sistematis (menaksir terlalu rendah di nilai tinggi atau sebaliknya); R² mendekati 1 berarti hampir seluruh variasi data uji dijelaskan.')
    with c2:
        section('Sebaran residual')
        fig = go.Figure()
        for m in meths:
            fig.add_trace(go.Violin(y=E[m], name=m, box_visible=True, meanline_visible=True, line_color=METHOD_COLOR[m],
                                    fillcolor=hex_rgba(METHOD_COLOR[m], .3), points=False))
        fig.add_hline(y=0, line_color='#9AA8B8', line_dash='dash'); fig.update_yaxes(title='Residual (aktual − prediksi)')
        plot(style(fig, height=430, legend=False), 'viol')
        bias = {m: E[m].mean() for m in meths}; a1 = {m: ac1(E[m]) for m in meths}
        interp(f'Violin yang sempit dan berpusat di nol = galat kecil dan tidak bias. Bias terbesar: <b>{max(bias, key=lambda k: abs(bias[k]))}</b> ({bias[max(bias, key=lambda k: abs(bias[k]))]:+.4f}). '
               f'Autokorelasi residual lag-1 berkisar {min(a1.values()):.2f}–{max(a1.values()):.2f}; nilai tinggi berarti masih ada pola yang belum ditangkap model (dan akan memicu alarm pada control chart).')

    section('Peta panas galat absolut sepanjang waktu uji', 'Merah = galat besar; baris = metode')
    Z = np.vstack([np.abs(E[m]) for m in meths])
    fig = go.Figure(go.Heatmap(z=Z, x=tstr(T), y=[lab[m] for m in meths], colorscale=SCALE_GOOD_LOW, colorbar=dict(title='|galat|', thickness=10)))
    fig.update_xaxes(type='date'); plot(style(fig, height=120 + 70 * len(meths), legend=False), 'errheat')
    peak = int(np.argmax(Z.max(0)))
    interp(f'Pita merah yang muncul serempak di beberapa baris menandakan periode yang sulit bagi <i>semua</i> metode (mis. sekitar {pd.Timestamp(T[peak]):%d %b %Y %H:%M}, galat maks {Z.max():.3f}); '
           'artinya penyebabnya kemungkinan perubahan proses/gangguan, bukan kelemahan satu model.')

    section('Tabel prediksi data uji')
    tb = pd.DataFrame({'Waktu': pd.to_datetime(T), f'{v} aktual': Y[:, j], **{f'{lab[m]}': yp[m] for m in meths}})
    table(tb, {c: st.column_config.NumberColumn(format='%.4f') for c in tb.columns if c != 'Waktu'} | {'Waktu': st.column_config.DatetimeColumn(format='YYYY-MM-DD HH:mm')}, height=300)


# =============================================================================
# TAB 5 — CONTROL CHART
# =============================================================================
def tab_cc(res):
    if need_res(res): return
    B, CC, CAL, S, UCL, T, cfg = res['BEST'], res['CC'], res['CAL'], res['CC_SERIES'], res['UCL'], res['T_TEST'], res['cfg']
    chart_names = ['MEWMA', 'Adaptive-DHWMA']
    section('Kalibrasi batas kendali (UCL)', f"Monte Carlo {cfg['n_cal']} replikasi × {cfg['max_rl']} observasi, target ARL₀ = 1/α = {1/cfg['alpha']:.1f}")
    c = st.columns(4)
    kpi(c[0], 'UCL MEWMA', fmt(UCL['MEWMA'], 3), f"λ = {cfg['lambda_mewma']}"); kpi(c[1], 'UCL Adaptive-DHWMA', fmt(UCL['Adaptive-DHWMA'], 3), f"λ {cfg['lambda_min']}–{cfg['lambda_max']}, γ={cfg['gamma']}")
    kpi(c[2], 'ARL₀ target', f"{1/cfg['alpha']:.1f}", f"α = {cfg['alpha']}"); kpi(c[3], 'ARL₀ hasil kalibrasi', f"{CAL.ARL0_hasil_kalibrasi.mean():.1f}", 'rata-rata kedua chart')
    st.write('')
    table(CAL, {k: st.column_config.NumberColumn(format='%.3f') for k in ['UCL', 'ARL0_target', 'ARL0_hasil_kalibrasi']})
    interp(f'UCL dicari agar proses terkendali menghasilkan alarm palsu rata-rata sekali per ±{1/cfg["alpha"]:.0f} pengamatan. ARL₀ hasil simulasi ({CAL.ARL0_hasil_kalibrasi.min():.0f}–{CAL.ARL0_hasil_kalibrasi.max():.0f}) '
           'yang dekat dengan target menandakan kalibrasi valid. Satu UCL dipakai untuk semua metode agar perbandingannya adil.')

    section('🎬 Control chart beranimasi', 'Residual data uji distandardisasi multivariat (Cholesky) lalu dipantau')
    m = st.selectbox('Metode', B.Metode.tolist(), key='cc_m')
    cols = st.columns(2)
    for col, ch, colr in zip(cols, chart_names, ['#2563EB', '#8B5CF6']):
        s, sig = S[(m, ch)]
        with col:
            plot(animated_control(T, s, sig, UCL[ch], f'{ch} — {m} | OOC = {int(sig.sum())}', colr), f'cc_{ch}')
    r = CC[CC.Metode == m].set_index('Chart')
    interp(f"Untuk <b>{m}</b>: MEWMA menandai {int(r.loc['MEWMA','Out_of_Control'])} titik ({r.loc['MEWMA','Persen_OOC']:.1f}%), "
           f"Adaptive-DHWMA {int(r.loc['Adaptive-DHWMA','Out_of_Control'])} titik ({r.loc['Adaptive-DHWMA','Persen_OOC']:.1f}%). Titik merah di atas garis UCL = residual keluar kendali "
           f"(galat prediksi bergeser dari perilaku biasanya). Secara nominal hanya ±{100*cfg['alpha']:.2f}% titik yang boleh menjadi alarm palsu; "
           'nilai jauh di atas itu mengindikasikan galat berpola/berautokorelasi, perubahan proses, atau model yang belum menangkap dinamika data.')

    c1, c2 = st.columns(2)
    with c1:
        section('Persentase out-of-control per metode')
        fig = px.bar(CC, x='Metode', y='Persen_OOC', color='Chart', barmode='group', text=CC.Persen_OOC.round(1),
                     color_discrete_map={'MEWMA': '#2563EB', 'Adaptive-DHWMA': '#8B5CF6'}, category_orders={'Metode': B.Metode.tolist()})
        fig.add_hline(y=100 * cfg['alpha'], line_dash='dot', line_color=WARN, annotation_text='alarm palsu nominal')
        fig.update_yaxes(title='% observasi OOC'); fig.update_xaxes(title='')
        plot(style(fig, height=420), 'ooc_bar')
        g = CC.groupby('Metode').Persen_OOC.mean()
        interp(f'Rata-rata OOC terendah: <b>{g.idxmin()}</b> ({g.min():.1f}%); tertinggi: <b>{g.idxmax()}</b> ({g.max():.1f}%). '
               'Metode dengan OOC rendah memiliki galat yang stabil dan mendekati acak (white noise), tanda model sudah menyerap sebagian besar struktur data.')
    with c2:
        section('Peta panas OOC (%)')
        H = CC.pivot_table(index='Metode', columns='Chart', values='Persen_OOC').reindex(B.Metode)
        fig = go.Figure(go.Heatmap(z=H.values, x=H.columns, y=H.index, colorscale=SCALE_GOOD_LOW, text=np.round(H.values, 1), texttemplate='%{text}%',
                                   colorbar=dict(thickness=10)))
        fig.update_yaxes(autorange='reversed'); plot(style(fig, height=420, legend=False), 'ooc_heat')
        d = (H['Adaptive-DHWMA'] - H['MEWMA'])
        interp(f'Selisih Adaptive-DHWMA − MEWMA rata-rata {d.mean():+.1f} poin persen. ' +
               ('Adaptive-DHWMA lebih sensitif (lebih banyak sinyal): cocok untuk mendeteksi pergeseran kecil, dengan konsekuensi perlu verifikasi lebih sering.' if d.mean() > 0
                else 'MEWMA memberi sinyal lebih banyak; Adaptive-DHWMA lebih tenang karena λ-nya menyesuaikan besar galat.'))

    section('Barcode sinyal — kapan setiap metode memicu alarm?')
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=chart_names, vertical_spacing=.12)
    for i, ch in enumerate(chart_names, 1):
        for mm in B.Metode:
            s, sig = S[(mm, ch)]
            fig.add_trace(go.Scatter(x=pd.to_datetime(T[sig]), y=[mm] * int(sig.sum()), mode='markers', name=mm, legendgroup=mm, showlegend=(i == 1),
                                     marker=dict(symbol='line-ns', size=14, line=dict(width=2, color=METHOD_COLOR[mm]))), row=i, col=1)
        fig.update_yaxes(categoryorder='array', categoryarray=B.Metode.tolist()[::-1], row=i, col=1)
    fig.update_xaxes(range=[pd.to_datetime(T[0]), pd.to_datetime(T[-1])])
    plot(style(fig, height=640), 'barcode')
    interp('Setiap garis vertikal adalah satu alarm. Garis yang muncul serentak pada banyak baris menandai kejadian yang sama-sama tidak terduga oleh banyak model (kemungkinan gangguan proses nyata); '
           'alarm yang hanya muncul pada satu metode lebih mungkin kelemahan model tersebut.')

    section('Tabel ringkasan control chart')
    table(CC, {'UCL': st.column_config.NumberColumn(format='%.3f'), 'Persen_OOC': st.column_config.NumberColumn('% OOC', format='%.2f')})
    first = CC[CC.Sinyal_Pertama != '-'].sort_values('Sinyal_Pertama').head(1)
    interp('Kolom <b>Episode_Sinyal</b> menghitung rangkaian alarm berurutan sebagai satu kejadian, sehingga lebih informatif daripada jumlah titik bila alarm datang bergerombol. ' +
           (f"Sinyal paling awal: {first.iloc[0].Metode} / {first.iloc[0].Chart} pada {first.iloc[0].Sinyal_Pertama}." if len(first) else 'Tidak ada sinyal pada semua metode.'))


# =============================================================================
# TAB 6 — METODE & DAMPAK
# =============================================================================
INFO = {
 'VAR': dict(kat='Statistik', cara='Setiap variabel diregresikan secara linear terhadap lag dirinya dan lag variabel lain (Vector AutoRegression). Diperiksa dengan uji ADF (stasioneritas), pemilihan lag AIC/BIC, dan uji stabilitas akar.',
             tuning='Lag 3/6/12 sesuai sliding window + lag optimal AIC (maks 30). Prediksi rolling satu-langkah memakai data aktual sebelumnya.',
             plus='Interpretabel, sangat cepat, baseline kuat bila hubungan linear dan deret stasioner.', minus='Tidak menangkap non-linearitas; sensitif terhadap non-stasioneritas dan lonjakan.',
             dampak='Mengukur seberapa jauh dinamika 4 variabel dapat dijelaskan oleh hubungan linear saja. Jika VAR kompetitif, struktur data cenderung linear dan model kompleks tidak banyak menambah nilai.'),
 'SVR': dict(kat='Machine Learning', cara='Support Vector Regression kernel RBF: mencari fungsi non-linear yang menyimpang ≤ ε dari data dengan margin maksimum. Satu model per variabel, fitur lag & target distandardisasi.',
             tuning='Grid C × gamma × epsilon dengan validasi kronologis 3 lipatan (masing-masing 15% akhir data latih).',
             plus='Kuat pada data berukuran sedang, tahan terhadap outlier berkat ε-insensitive loss.', minus='Skala fitur wajib distandardisasi; pencarian grid mahal bila data/fitur besar.',
             dampak='Mengukur manfaat non-linearitas halus. Pemilihan C/gamma menunjukkan seberapa “lentur” hubungan antar-lag pada data ini.'),
 'Random Forest': dict(kat='Machine Learning', cara='Kumpulan banyak pohon keputusan yang dilatih pada sampel acak lalu dirata-ratakan (bagging). Memodelkan 4 output sekaligus.',
             tuning='RandomizedSearchCV (n_estimators, max_depth, min_samples_split/leaf, max_features) + TimeSeriesSplit 4 lipatan.',
             plus='Tahan overfitting, menangkap interaksi non-linear tanpa asumsi distribusi.', minus='Tidak bisa mengekstrapolasi di luar rentang nilai latih; prediksi cenderung “mendatar” pada tren baru.',
             dampak='Bila data uji bergeser keluar dari rentang data latih, RF akan menahan prediksi pada nilai yang pernah dilihat sehingga bias muncul di wilayah tersebut.'),
 'XGBoost': dict(kat='Machine Learning', cara='Gradient boosting: pohon-pohon dangkal dibangun berurutan, masing-masing memperbaiki galat pendahulunya, dengan regularisasi L1/L2.',
             tuning='Tanpa tuning — hyperparameter tetap per variabel (lihat tabel di bawah).', plus='Akurasi tinggi, regularisasi kuat, cepat pada data tabular.',
             minus='Seperti RF, tidak mengekstrapolasi; hasil bergantung pada hyperparameter tetap yang diberikan.',
             dampak='Menjadi pembanding “siap pakai” (tanpa tuning). Selisihnya terhadap RF/SVR yang dituning menunjukkan nilai tambah proses tuning.'),
 'LightGBM': dict(kat='Machine Learning', cara='Boosting berbasis histogram dengan pertumbuhan pohon leaf-wise sehingga sangat efisien.',
             tuning='Tanpa tuning — hyperparameter tetap per variabel; num_leaves & min_child_samples memakai default.', plus='Sangat cepat dan hemat memori, akurasi setara XGBoost.',
             minus='Leaf-wise dapat overfitting pada data kecil bila tidak dibatasi.', dampak='Dibandingkan dengan XGBoost, memperlihatkan apakah strategi pertumbuhan pohon berpengaruh pada data ini.'),
 'LSTM': dict(kat='Deep Learning', cara='Jaringan saraf rekuren dengan gerbang input/forget/output yang menyimpan memori jangka panjang pada sekuens window.',
             tuning='Grid units × learning rate; data MinMax; early stopping & ReduceLROnPlateau; 10% akhir latih untuk validasi.', plus='Menangkap ketergantungan temporal non-linear.',
             minus='Butuh data lebih banyak, sensitif inisialisasi/seed, waktu latih lebih lama.', dampak='Pada window pendek (3–12 langkah) keunggulan memori panjang LSTM biasanya terbatas; hasil menunjukkan apakah ada dependensi non-linear yang bermakna.'),
 'RNN': dict(kat='Deep Learning', cara='SimpleRNN: state tersembunyi diperbarui tiap langkah waktu tanpa gerbang.', tuning='Protokol sama dengan LSTM.',
             plus='Sederhana dan cepat dibanding LSTM/GRU.', minus='Rentan vanishing gradient pada sekuens panjang.',
             dampak='Berfungsi sebagai pembanding minimal untuk jaringan rekuren; jika RNN setara LSTM/GRU, gerbang tambahan tidak diperlukan pada window ini.'),
 'GRU': dict(kat='Deep Learning', cara='Varian LSTM dengan dua gerbang (update & reset): lebih ringkas namun tetap menjaga memori.', tuning='Protokol sama dengan LSTM.',
             plus='Parameter lebih sedikit, latih lebih cepat, sering setara LSTM.', minus='Tetap bergantung pada seed dan jumlah data.',
             dampak='Kompromi efisiensi–akurasi; sering menjadi pilihan praktis untuk deret proses industri berukuran menengah.'),
 'TCN': dict(kat='Deep Learning', cara='Temporal Convolutional Network: konvolusi kausal berdilasi dengan koneksi residual sehingga menangkap pola lokal pada berbagai skala waktu.',
             tuning='Protokol sama dengan LSTM; jumlah blok ditentukan agar receptive field ≥ window.', plus='Paralel, stabil dilatih, receptive field fleksibel.',
             minus='Receptive field terbatas oleh jumlah blok/dilasi.', dampak='Menguji apakah pola lokal berskala-ganda lebih informatif daripada memori berurutan ala RNN.'),
}


def tab_methods(res):
    section('Metode prediksi & dampaknya pada dataset', 'Ketuk panel untuk membaca cara kerja, tuning, serta dampak pada data NPK Phonska IV')
    for m in eng.METHOD_ORDER:
        i = INFO[m]
        badge = ''
        if res is not None and (res['BEST'].Metode == m).any():
            r = res['BEST'][res['BEST'].Metode == m].iloc[0]
            badge = f" — peringkat #{int(r.No)} • w={int(r.Window)} • RMSE {r.RMSE_avg:.3f}"
        with st.expander(f'{m}  ·  {i["kat"]}{badge}'):
            st.markdown(f"**Cara kerja.** {i['cara']}\n\n**Tuning.** {i['tuning']}\n\n**Kelebihan.** {i['plus']}\n\n**Keterbatasan.** {i['minus']}")
            msg = f"<b>Dampak pada dataset ini:</b> {i['dampak']}"
            if res is not None and (res['BEST'].Metode == m).any():
                r = res['BEST'][res['BEST'].Metode == m].iloc[0]; top = res['BEST'].iloc[0]
                msg += f"<br><b>Hasil nyata:</b> RMSE rata-rata {r.RMSE_avg:.4f}, MAPE {r.MAPE_avg:.3f}% pada window {int(r.Window)}; " + \
                       ('metode ini <b>terbaik</b>.' if r.Metode == top.Metode else f"{100*(r.RMSE_avg/top.RMSE_avg-1):.1f}% di atas RMSE metode terbaik ({top.Metode}).")
            interp(msg)
    section('Hyperparameter tetap XGBoost & LightGBM')
    table(eng.hp_table(), {c: st.column_config.NumberColumn(format='%.6g') for c in eng.hp_table().columns if c not in ('Model', 'Variabel')})
    interp('Parameter ini dipakai apa adanya tanpa pencarian ulang. <b>learning_rate</b> kecil + <b>n_estimators</b> besar = pembelajaran bertahap yang stabil; <b>max_depth</b> membatasi kompleksitas pohon; '
           '<b>subsample</b> & <b>colsample_bytree</b> < 1 menambah keacakan untuk mengurangi overfitting; <b>reg_lambda</b> adalah penalti L2.')

    section('Metrik & protokol evaluasi')
    c = st.columns(3)
    c[0].markdown('**RMSE** — akar rata-rata galat kuadrat (satuan asli). Menghukum galat besar lebih berat; dirata-ratakan lintas 4 variabel.')
    c[1].markdown('**MAPE** — rata-rata galat relatif absolut dalam %. Mudah dibandingkan antarvariabel, tetapi membesar bila nilai aktual mendekati nol.')
    c[2].markdown('**RMSE ternormalisasi** — RMSE ÷ SD data latih, dirata-ratakan. Bebas skala; <1 berarti lebih baik daripada menebak rata-rata.')
    interp('<b>Protokol adil:</b> (1) data uji identik untuk semua metode dan window, (2) prediksi one-step-ahead dengan input data aktual, (3) rumus metrik sama pada skala asli. '
           '<b>Dampak sliding window:</b> menentukan banyaknya riwayat yang dilihat model—terlalu pendek kehilangan konteks, terlalu panjang menambah dimensi dan risiko overfitting.')

    section('Control chart: MEWMA & Adaptive-DHWMA')
    st.markdown("""
- **MEWMA** menghaluskan vektor residual dengan λ tetap lalu menghitung statistik T²; sensitif terhadap pergeseran rata-rata kecil yang menetap.
- **Adaptive-DHWMA** memakai pemulusan ganda (Henderson/Holt-type) dengan λₜ = λmin + (λmax − λmin)(1 − e^(−γ·dₜ)); makin besar galat, makin cepat chart bereaksi.
- **UCL** dikalibrasi lewat Monte Carlo agar ARL₀ ≈ 1/α, sehingga kedua chart punya laju alarm palsu yang sama.""")
    interp('<b>Dampak pada dataset:</b> control chart tidak memonitor data mentah, melainkan <i>kualitas prediksi</i>. Banyaknya sinyal pada residual suatu model menandakan galatnya berpola atau proses berubah; '
           'dua hal itu perlu dibedakan dengan melihat apakah sinyalnya muncul serentak pada banyak metode (perubahan proses) atau hanya satu (kelemahan model).')


# =============================================================================
# TAB 7 — EKSPOR
# =============================================================================
def tab_export(res, data_name):
    if need_res(res): return
    section('Unduh hasil')
    c = st.columns(3)
    with c[0]:
        st.markdown('**📊 Laporan Excel lengkap** (peringkat, per-variabel, per-window, control chart, lampiran)')
        if st.button('Siapkan file Excel', key='mk_xlsx'):
            with st.spinner('Membuat Excel …'):
                st.session_state['xlsx'] = eng.build_excel(res, data_name)
        if 'xlsx' in st.session_state:
            st.download_button('⬇️ Unduh Perbandingan_9_Metode_NPK.xlsx', st.session_state['xlsx'], 'Perbandingan_9_Metode_NPK.xlsx',
                               'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    with c[1]:
        st.markdown('**📈 Prediksi data uji (CSV)**')
        pr = pd.DataFrame({'Time': pd.to_datetime(res['T_TEST'])})
        for j, v in enumerate(res['VARS']):
            pr[f'{v}_Aktual'] = res['Y_TEST'][:, j]
            for (m, w) in res['BEST_KEYS']: pr[f'{v}_{m}_w{w}'] = res['PREDS'][(m, w)][:, j]
        st.download_button('⬇️ Unduh prediksi.csv', pr.to_csv(index=False).encode(), 'prediksi_uji.csv', 'text/csv')
    with c[2]:
        st.markdown('**⚙️ Konfigurasi (JSON)**')
        st.download_button('⬇️ Unduh konfigurasi.json', json.dumps(res['cfg'], indent=2, default=str).encode(), 'konfigurasi.json', 'application/json')
    interp('Excel memuat tabel perbandingan berformat (sel hijau = terbaik) dan seluruh lampiran tuning. File CSV berisi prediksi tiap metode pada data uji untuk analisis lanjutan, '
           'dan JSON merekam konfigurasi agar eksperimen dapat direproduksi.')
    section('Seluruh hasil (semua metode × window)')
    table(res['ALL'].drop(columns=['Catatan']), {c: st.column_config.NumberColumn(format='%.4f') for c in res['ALL'].columns if c.startswith(('RMSE', 'MAPE'))}, height=420)


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.markdown(f"""
    <div class="hero"><h1>🌾 {CONFIG['APP_TITLE']}</h1>
    <p>Perbandingan 9 metode prediksi multivariat — VAR, SVR, Random Forest, XGBoost, LightGBM, LSTM, RNN, GRU, TCN — dengan pemantauan galat memakai control chart MEWMA & Adaptive-DHWMA.</p>
    <span class="chip">Interval 4 jam</span><span class="chip">One-step-ahead</span><span class="chip">RMSE & MAPE</span><span class="chip">Grafik interaktif & animasi</span></div>
    """, unsafe_allow_html=True)

    df, name, cfg, submitted = sidebar()
    if df is None:
        st.info('Menunggu dataset. Letakkan file Excel di folder `data/` pada repo atau unggah lewat sidebar.'); return
    errs = validate(cfg)
    if submitted:
        if errs:
            for e in errs: st.error(e)
        else:
            key = eng.make_key(df, cfg); cached = eng.load_cache(key)
            if cached is not None:
                st.session_state['res'] = cached; st.session_state.pop('xlsx', None); st.toast('Hasil dimuat dari cache ✅')
            else:
                with st.status('Menjalankan analisis …', expanded=True) as status:
                    bar = st.progress(0.0); txt = st.empty()
                    def cb(msg, f): bar.progress(float(min(max(f, 0), 1))); txt.write(f'⏳ {msg}')
                    try:
                        res = eng.run_analysis(df, cfg, cb)
                        eng.save_cache(key, res); st.session_state['res'] = res; st.session_state.pop('xlsx', None)
                        bar.progress(1.0); status.update(label=f"Selesai dalam {res['elapsed']/60:.1f} menit", state='complete', expanded=False)
                    except Exception as e:
                        status.update(label='Gagal', state='error'); st.exception(e)
    res = st.session_state.get('res')
    if res is not None and res['cfg'] != cfg and not submitted:
        st.caption('ℹ️ Tab hasil menampilkan analisis terakhir yang dijalankan; tekan **Jalankan Analisis** untuk menerapkan perubahan konfigurasi.')

    tabs = st.tabs(['🏠 Beranda', '📊 Eksplorasi Data', '🏆 Perbandingan Metode', '🎯 Prediksi & Residual', '🚦 Control Chart', '📚 Metode & Dampak', '📥 Ekspor'])
    with tabs[0]: tab_home(df, cfg, res)
    with tabs[1]: tab_eda(df, cfg if not errs else {**cfg, 'vars': CONFIG['VARS']})
    with tabs[2]: tab_compare(res)
    with tabs[3]: tab_pred(res)
    with tabs[4]: tab_cc(res)
    with tabs[5]: tab_methods(res)
    with tabs[6]: tab_export(res, name)


main()
