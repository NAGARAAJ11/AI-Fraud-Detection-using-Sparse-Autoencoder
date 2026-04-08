# ==============================
# NeuralGuard Fraud Detection Dashboard
# Dark Theme — Cyberpunk Fintech Aesthetic
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NeuralGuard — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — Cyberpunk dark fintech aesthetic
# Fonts: Orbitron (display) + Syne (body) — sharp, technical, memorable
# Colors: deep navy bg + electric cyan accent + hot coral fraud alert
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Syne:wght@300;400;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #070b14;
    color: #cbd5e1;
}
.stApp { background-color: #070b14; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #070b14 100%);
    border-right: 1px solid rgba(0,255,200,0.1);
}
section[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; letter-spacing: 0.05em; }
h1 { color: #00ffc8 !important; font-size: 1.7rem !important; font-weight: 800; }
h2 { color: #00d4aa !important; font-size: 1.2rem !important; }
h3 { color: #7fffd4 !important; font-size: 1rem !important; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0b1120 0%, #0d1929 100%);
    border: 1px solid rgba(0,255,200,0.15);
    border-top: 2px solid #00ffc8;
    border-radius: 12px;
    padding: 1.2rem 1rem 1rem;
    position: relative;
    overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 100px; height: 100px;
    background: radial-gradient(circle, rgba(0,255,200,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
div[data-testid="stMetricValue"] {
    color: #00ffc8 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700;
}
div[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Custom hero banner ── */
.ng-hero {
    background: linear-gradient(135deg, #0b1120 0%, #0d1a2e 60%, #0b1428 100%);
    border: 1px solid rgba(0,255,200,0.12);
    border-left: 4px solid #00ffc8;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.ng-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,255,200,0.04) 0%, transparent 70%);
    border-radius: 50%;
}
.ng-hero h1 {
    font-size: 2rem !important;
    margin: 0 0 0.4rem !important;
    background: linear-gradient(90deg, #00ffc8, #00b8d9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ng-hero p { color: #64748b; font-size: 0.95rem; margin: 0; }

/* ── Tag pills ── */
.ng-tag {
    display: inline-block;
    background: rgba(0,255,200,0.08);
    border: 1px solid rgba(0,255,200,0.25);
    color: #00ffc8;
    border-radius: 20px;
    padding: 0.18rem 0.75rem;
    font-size: 0.72rem;
    font-family: 'Orbitron', monospace;
    margin: 0.2rem 0.15rem;
    letter-spacing: 0.04em;
}
.ng-tag.fraud {
    background: rgba(255,60,90,0.08);
    border-color: rgba(255,60,90,0.3);
    color: #ff3c5a;
}

/* ── Section card ── */
.ng-card {
    background: linear-gradient(135deg, #0b1120, #0d1929);
    border: 1px solid rgba(0,255,200,0.1);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin: 0.6rem 0;
}

/* ── Alert fraud card ── */
.ng-alert {
    background: linear-gradient(135deg, rgba(255,60,90,0.08), rgba(255,60,90,0.03));
    border: 1px solid rgba(255,60,90,0.25);
    border-left: 3px solid #ff3c5a;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #fca5a5;
}

/* ── Divider ── */
.ng-divider {
    border: none;
    border-top: 1px solid rgba(0,255,200,0.08);
    margin: 2rem 0;
}

/* ── Table ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stDataFrame thead th {
    background: #0b1120 !important;
    color: #00ffc8 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00c9a7, #0099cc);
    color: #070b14;
    font-family: 'Orbitron', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.8rem;
    transition: all 0.2s;
    box-shadow: 0 0 20px rgba(0,201,167,0.2);
}
.stButton > button:hover {
    box-shadow: 0 0 35px rgba(0,201,167,0.4);
    transform: translateY(-1px);
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1a2744, #0d1929);
    color: #00ffc8;
    border: 1px solid rgba(0,255,200,0.3);
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    border-radius: 8px;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    border-color: #00ffc8;
    box-shadow: 0 0 20px rgba(0,255,200,0.15);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0b1120;
    border: 2px dashed rgba(0,255,200,0.2);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,255,200,0.5);
}

/* ── Selectbox, slider labels ── */
.stSelectbox label, .stSlider label, .stFileUploader label {
    color: #64748b !important;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Success / error alerts ── */
.stSuccess { border-radius: 10px; background: rgba(0,201,167,0.1); border-color: #00c9a7; }
.stError   { border-radius: 10px; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1120;
    border-radius: 10px;
    padding: 0.3rem;
    gap: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace;
    font-size: 0.72rem;
    color: #64748b;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    letter-spacing: 0.04em;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,255,200,0.1) !important;
    color: #00ffc8 !important;
}

/* ── Sidebar radio ── */
.stRadio label { color: #94a3b8 !important; font-size: 0.85rem; }
.stRadio [data-baseweb="radio"] { accent-color: #00ffc8; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b1120; }
::-webkit-scrollbar-thumb { background: rgba(0,255,200,0.2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG  = "#070b14"
CARD_BG  = "#0b1120"
GRID     = "#0f1c30"
CYAN     = "#00ffc8"
CYAN2    = "#00b8d9"
CORAL    = "#ff3c5a"
VIOLET   = "#7c6af7"
GOLD     = "#f5a623"
TEXT     = "#cbd5e1"
SUBTEXT  = "#475569"

def _base_layout(**kwargs):
    # Merge caller overrides into base defaults to avoid duplicate key errors
    base_yaxis = dict(gridcolor=GRID, zerolinecolor=GRID, color=SUBTEXT)
    base_xaxis = dict(gridcolor=GRID, zerolinecolor=GRID, color=SUBTEXT)
    if "yaxis" in kwargs:
        base_yaxis = {**base_yaxis, **kwargs.pop("yaxis")}
    if "xaxis" in kwargs:
        base_xaxis = {**base_xaxis, **kwargs.pop("xaxis")}
    return dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Syne, sans-serif", color=TEXT, size=12),
        margin=dict(l=50, r=30, t=55, b=50),
        xaxis=base_xaxis,
        yaxis=base_yaxis,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID,
                    font=dict(color=TEXT)),
        **kwargs
    )

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 1.8rem;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">🛡️</div>
        <div style="font-family:'Orbitron',monospace; font-size:1.1rem;
                    color:#00ffc8; font-weight:800; letter-spacing:0.1em;">
            NEURALGUARD
        </div>
        <div style="font-size:0.72rem; color:#334155; margin-top:0.4rem;
                    letter-spacing:0.12em; text-transform:uppercase;">
            Fraud Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.8rem 1rem; background:#0b1120;
                border:1px solid rgba(0,255,200,0.1); border-radius:10px;
                margin-bottom:1rem;">
        <div style="font-size:0.7rem; color:#475569; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:0.6rem;">Status</div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <div style="width:8px;height:8px;border-radius:50%;
                        background:#00ffc8;box-shadow:0 0 8px #00ffc8;"></div>
            <span style="font-size:0.82rem;color:#94a3b8;">System Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    threshold_pct = st.slider("Detection Threshold %ile", 90, 99, 95, 1)

    st.markdown("<hr style='border-color:rgba(0,255,200,0.08);margin:1rem 0;'>",
                unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="ng-hero">
    <h1>🛡️ NEURALGUARD</h1>
    <p>Real-time credit card fraud detection powered by a Sparse Autoencoder
    with L1 sparsity regularisation. Upload transaction data to begin analysis.</p>
    <div style="margin-top:1rem;">
        <span class="ng-tag">Sparse Autoencoder</span>
        <span class="ng-tag">Anomaly Detection</span>
        <span class="ng-tag">KL Divergence</span>
        <span class="ng-tag fraud">Live Fraud Scoring</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & SCALER
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    try:
        model  = load_model("model.h5", compile=False)
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_artifacts()

if model is None:
    st.markdown("""
    <div class="ng-alert">
        ⚠️ <b>model.h5</b> or <b>scaler.pkl</b> not found.<br>
        Run the training pipeline first, then reload this dashboard.
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# FILE UPLOAD
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### 📂 Upload Transaction Data")
uploaded_file = st.file_uploader(
    "Drop your CSV here — creditcard_2023.csv or any transaction file",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown("""
    <div class="ng-card" style="text-align:center; padding:3rem;">
        <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">📊</div>
        <div style="font-family:'Orbitron',monospace; font-size:0.9rem;
                    color:#334155; letter-spacing:0.08em;">
            AWAITING DATA UPLOAD
        </div>
        <div style="font-size:0.82rem; color:#1e293b; margin-top:0.5rem;">
            Upload a CSV file with transaction features to begin fraud analysis
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING & PREDICTION
# ──────────────────────────────────────────────────────────────────────────────

with st.spinner("🔍 Analysing transactions…"):
    data = pd.read_csv(uploaded_file).dropna()

    has_labels = "Class" in data.columns
    X = data.drop("Class", axis=1) if has_labels else data
    y_true = data["Class"].values.astype(int) if has_labels else None

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

    if model is not None:
        reconstructions = model.predict(X_scaled, verbose=0)
    else:
        # Fallback: simulate for demo
        reconstructions = X_scaled + np.random.randn(*X_scaled.shape) * 0.1

    mse       = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, threshold_pct)
    y_pred    = (mse > threshold).astype(int)

    results = data.copy()
    results["Anomaly_Score"]   = mse
    results["Fraud_Prediction"] = y_pred
    results["Risk_Level"]      = pd.cut(
        mse, bins=[0, np.percentile(mse,70), np.percentile(mse,90), mse.max()+1],
        labels=["🟢 Low","🟡 Medium","🔴 High"]
    )

st.success(f"✅ Analysis complete — {len(data):,} transactions processed")

# ──────────────────────────────────────────────────────────────────────────────
# KPI METRICS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("<hr class='ng-divider'>", unsafe_allow_html=True)
st.markdown("### 📊 Detection Summary")

total     = len(results)
fraud_ct  = int(y_pred.sum())
normal_ct = total - fraud_ct
fraud_pct = 100 * fraud_ct / total
avg_score = float(mse.mean())

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("🔴 Fraud Flagged",    f"{fraud_ct:,}")
c3.metric("✅ Normal Passed",    f"{normal_ct:,}")
c4.metric("Fraud Rate",          f"{fraud_pct:.2f}%")
c5.metric("Avg Anomaly Score",   f"{avg_score:.5f}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN VISUALISATION TABS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("<hr class='ng-divider'>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Distribution", "🔲 Error Analysis",
    "📉 ROC & PR", "🗃️ Results Table", "⚔️ Model Comparison"
])

# ── Tab 1: Distributions ─────────────────────────────────────────────────────
with tab1:
    c_pie, c_bar = st.columns(2)

    with c_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["Normal", "Fraud"],
            values=[normal_ct, fraud_ct],
            hole=0.62,
            marker=dict(
                colors=[CYAN, CORAL],
                line=dict(color=DARK_BG, width=3)
            ),
            textfont=dict(family="Syne", size=13, color=TEXT),
        ))
        fig_pie.add_annotation(
            text=f"<b>{fraud_pct:.1f}%</b><br><span style='font-size:10px'>Fraud</span>",
            x=0.5, y=0.5, font=dict(size=20, color=CORAL, family="Orbitron"),
            showarrow=False
        )
        fig_pie.update_layout(**_base_layout(
            title=dict(text="Transaction Distribution", font=dict(color=CYAN, family="Orbitron", size=13)),
            height=360, showlegend=True
        ))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_bar:
        # Anomaly score distribution by risk tier
        bins = np.histogram(mse, bins=40)
        fig_hist = go.Figure(go.Bar(
            x=bins[1][:-1], y=bins[0],
            marker=dict(
                color=bins[1][:-1],
                colorscale=[[0,CYAN],[0.6,VIOLET],[1,CORAL]],
                line=dict(width=0)
            ),
            opacity=0.85,
            name="Anomaly Score"
        ))
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color=GOLD, line_width=2,
                           annotation_text=f"Threshold ({threshold_pct}th pct)",
                           annotation_font_color=GOLD)
        fig_hist.update_layout(**_base_layout(
            title=dict(text="Anomaly Score Distribution", font=dict(color=CYAN, family="Orbitron", size=13)),
            height=360,
            xaxis_title="Reconstruction Error (MSE)",
            yaxis_title="Count"
        ))
        st.plotly_chart(fig_hist, use_container_width=True)

# ── Tab 2: Error Analysis ─────────────────────────────────────────────────────
with tab2:
    if has_labels and y_true is not None:
        c_err, c_cm = st.columns(2)

        with c_err:
            mse_n = mse[y_true == 0]
            mse_f = mse[y_true == 1]
            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(x=mse_n, name="Normal",
                                            marker_color=CYAN, opacity=0.7, nbinsx=50))
            fig_err.add_trace(go.Histogram(x=mse_f, name="Fraud",
                                            marker_color=CORAL, opacity=0.7, nbinsx=50))
            fig_err.add_vline(x=threshold, line_dash="dash", line_color=GOLD,
                               annotation_text="Threshold", annotation_font_color=GOLD)
            fig_err.update_layout(**_base_layout(
                title=dict(text="Reconstruction Error — Normal vs Fraud",
                           font=dict(color=CYAN, family="Orbitron", size=13)),
                barmode="overlay", height=360,
                xaxis_title="Reconstruction Error (MSE)", yaxis_title="Count"
            ))
            st.plotly_chart(fig_err, use_container_width=True)

        with c_cm:
            cm = confusion_matrix(y_true, y_pred)
            tn,fp,fn,tp = cm.ravel() if cm.size==4 else (0,0,0,0)
            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Predicted Normal","Predicted Fraud"],
                y=["Actual Normal","Actual Fraud"],
                text=[[str(v) for v in row] for row in cm],
                texttemplate="<b>%{text}</b>",
                textfont=dict(size=22, family="Orbitron"),
                colorscale=[[0,CARD_BG],[0.5,"#0d2b4a"],[1,CYAN]],
                showscale=False
            ))
            fig_cm.update_layout(**_base_layout(
                title=dict(text="Confusion Matrix",
                           font=dict(color=CYAN, family="Orbitron", size=13)),
                height=360
            ))
            st.plotly_chart(fig_cm, use_container_width=True)

        # Stats row
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        s1,s2,s3,s4,s5 = st.columns(5)
        s1.metric("True Positives",  f"{tp:,}")
        s2.metric("False Positives", f"{fp:,}")
        s3.metric("False Negatives", f"{fn:,}")
        s4.metric("Precision",       f"{precision:.4f}")
        s5.metric("Recall",          f"{recall:.4f}")
    else:
        st.info("Upload a file with a 'Class' column to see error analysis and confusion matrix.")

# ── Tab 3: ROC & PR ───────────────────────────────────────────────────────────
with tab3:
    if has_labels and y_true is not None:
        from sklearn.metrics import average_precision_score, precision_recall_curve
        fpr, tpr, _ = roc_curve(y_true, mse)
        auc = roc_auc_score(y_true, mse)
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, mse)
        auc_pr = average_precision_score(y_true, mse)

        c_roc, c_pr = st.columns(2)

        with c_roc:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"SAE (AUC={auc:.3f})",
                line=dict(color=CYAN, width=2.5),
                fill="tozeroy", fillcolor="rgba(0,255,200,0.05)"
            ))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                          line=dict(color=SUBTEXT, width=1.5, dash="dash")))
            fig_roc.update_layout(**_base_layout(
                title=dict(text=f"ROC Curve  |  AUC = {auc:.4f}",
                           font=dict(color=CYAN, family="Orbitron", size=13)),
                height=380, xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            ))
            st.plotly_chart(fig_roc, use_container_width=True)

        with c_pr:
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=rec_arr, y=prec_arr, name=f"SAE (AUC-PR={auc_pr:.3f})",
                line=dict(color=VIOLET, width=2.5),
                fill="tozeroy", fillcolor="rgba(124,106,247,0.05)"
            ))
            fig_pr.update_layout(**_base_layout(
                title=dict(text=f"Precision-Recall  |  AUC-PR = {auc_pr:.4f}",
                           font=dict(color=VIOLET, family="Orbitron", size=13)),
                height=380, xaxis_title="Recall", yaxis_title="Precision"
            ))
            st.plotly_chart(fig_pr, use_container_width=True)

        # AUC banner
        st.markdown(f"""
        <div class="ng-card" style="text-align:center; padding:1rem;">
            <span style="font-family:'Orbitron',monospace; font-size:0.8rem;
                         color:#475569; letter-spacing:0.1em;">
                AUC-ROC
            </span>
            <span style="font-family:'Orbitron',monospace; font-size:2rem;
                         color:#00ffc8; margin:0 2rem; font-weight:800;">
                {auc:.4f}
            </span>
            <span style="font-family:'Orbitron',monospace; font-size:0.8rem;
                         color:#475569; letter-spacing:0.1em;">
                AUC-PR
            </span>
            <span style="font-family:'Orbitron',monospace; font-size:2rem;
                         color:#7c6af7; margin-left:2rem; font-weight:800;">
                {auc_pr:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Upload a file with 'Class' column to see ROC and Precision-Recall curves.")

# ── Tab 4: Results Table ──────────────────────────────────────────────────────
with tab4:
    st.markdown(f"""
    <div class="ng-card" style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-family:'Orbitron',monospace; color:#00ffc8; font-size:0.85rem;">
            PREDICTION RESULTS
        </span>
        <span style="font-size:0.8rem; color:#475569;">
            Showing top 100 of {total:,} transactions
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Highlight fraud rows
    display_cols = ["Anomaly_Score","Fraud_Prediction","Risk_Level"]
    if has_labels:
        display_cols = ["Class"] + display_cols

    show_df = results[display_cols + [c for c in results.columns
                                       if c not in display_cols]].head(100)

    st.dataframe(
        show_df.style
               .background_gradient(subset=["Anomaly_Score"],
                                    cmap="RdYlGn_r", vmin=0, vmax=threshold*2)
               .map(lambda v: "color: #ff3c5a; font-weight:bold"
                          if v == 1 else "", subset=["Fraud_Prediction"]),
        use_container_width=True,
        height=440
    )

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download Full Results CSV",
        data=csv_bytes,
        file_name="neuralguard_fraud_results.csv",
        mime="text/csv"
    )

# ── Tab 5: Model Comparison ─────────────────────────────────────────────────
with tab5:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    colors_cmp  = [CYAN, VIOLET, CORAL]
    metric_keys = ["AUC-ROC", "F1", "Precision", "Recall", "Accuracy"]

    def _calc_metrics(yt, yp, ys):
        try:
            return {
                "Accuracy":  round(accuracy_score(yt, yp), 4),
                "Precision": round(precision_score(yt, yp, zero_division=0), 4),
                "Recall":    round(recall_score(yt, yp, zero_division=0), 4),
                "F1":        round(f1_score(yt, yp, zero_division=0), 4),
                "AUC-ROC":   round(roc_auc_score(yt, ys), 4),
            }
        except Exception:
            return {k: 0.0 for k in ["Accuracy","Precision","Recall","F1","AUC-ROC"]}

    def _draw_comparison(metrics, roc_data):
        fig_cmp = go.Figure()
        for i, (name, vals) in enumerate(metrics.items()):
            fig_cmp.add_trace(go.Bar(
                name=name, x=metric_keys,
                y=[vals[k] for k in metric_keys],
                marker_color=colors_cmp[i], opacity=0.85,
                text=[f"{vals[k]:.3f}" for k in metric_keys],
                textposition="outside",
                textfont=dict(size=11, color=TEXT, family="Orbitron")
            ))
        fig_cmp.update_layout(**_base_layout(
            title=dict(text="MODEL PERFORMANCE COMPARISON",
                       font=dict(color=CYAN, family="Orbitron", size=13)),
            barmode="group", height=440,
            yaxis=dict(gridcolor=GRID, color=SUBTEXT, range=[0, 1.18]),
        ))
        st.plotly_chart(fig_cmp, use_container_width=True)

        if roc_data:
            fig_roc2 = go.Figure()
            for i, (name, (fpr_, tpr_, auc_)) in enumerate(roc_data.items()):
                fig_roc2.add_trace(go.Scatter(
                    x=fpr_, y=tpr_,
                    name=f"{name} (AUC={auc_:.3f})",
                    line=dict(color=colors_cmp[i], width=2.5)
                ))
            fig_roc2.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], name="Random",
                line=dict(color=SUBTEXT, dash="dash", width=1)
            ))
            fig_roc2.update_layout(**_base_layout(
                title=dict(text="ROC CURVES — ALL MODELS",
                           font=dict(color=CYAN, family="Orbitron", size=13)),
                height=400, xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            ))
            st.plotly_chart(fig_roc2, use_container_width=True)

        df_cmp = pd.DataFrame(metrics).T
        st.dataframe(
            df_cmp.style.highlight_max(axis=0, color="#0d2b4a"),
            use_container_width=True
        )
        best = max(metrics, key=lambda k: metrics[k]["AUC-ROC"])
        st.markdown(f"""
        <div class="ng-card" style="border-left:3px solid {CYAN};
                    text-align:center; padding:1.2rem; margin-top:1rem;">
            <span style="font-family:'Orbitron',monospace; font-size:0.75rem;
                         color:#475569; letter-spacing:0.1em;">BEST MODEL</span><br>
            <span style="font-family:'Orbitron',monospace; font-size:1.4rem;
                         color:#00ffc8; font-weight:800;">{best}</span>
            <span style="font-size:0.85rem; color:#475569; margin-left:1rem;">
                AUC-ROC = {metrics[best]["AUC-ROC"]:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Algorithm overview cards — always visible
    ca, cb, cc = st.columns(3)
    algo_cards = [
        ("🔷", "Sparse Autoencoder", CYAN,
         "Pre-trained neural net. Anomaly score = reconstruction MSE. "
         "L1 sparsity prevents memorisation of fraud patterns."),
        ("🌲", "Isolation Forest", VIOLET,
         "Random tree ensemble. Anomalies isolated in fewer splits — "
         "shorter path length = higher anomaly score."),
        ("🔴", "One-Class SVM", CORAL,
         "RBF-kernel boundary around normal data. "
         "Samples outside the margin are flagged as anomalies."),
    ]
    for col, (icon, title, color, desc) in zip([ca, cb, cc], algo_cards):
        with col:
            st.markdown(f"""
            <div class="ng-card" style="border-top:2px solid {color}; min-height:150px;">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-family:'Orbitron',monospace; font-size:0.78rem;
                             color:{color}; margin-bottom:0.5rem;">{title}</div>
                <div style="font-size:0.79rem; color:#475569;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='ng-divider'>", unsafe_allow_html=True)

    if not has_labels:
        st.markdown("""
        <div class="ng-card" style="border-left:3px solid #f5a623; margin-bottom:1rem;">
            <b style="color:#f5a623;">No Class column detected.</b>
            Running in score-only mode — shows normalised anomaly score
            distributions per model without evaluation metrics.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run Score Comparison (no labels)", type="primary", key="run_nolabel"):
            with st.spinner("Running all three models on your data…"):
                try:
                    iso2 = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
                    iso2.fit(X_scaled)
                    iso_sc = -iso2.decision_function(X_scaled)

                    svm2 = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
                    svm2.fit(X_scaled[:min(3000, len(X_scaled))])
                    svm_sc = -svm2.decision_function(X_scaled)

                    def _norm(s):
                        return (s - s.min()) / (s.max() - s.min() + 1e-9)

                    fig_sc = go.Figure()
                    for sc, name, color in [
                        (mse,    "SAE Reconstruction Error", CYAN),
                        (iso_sc, "Isolation Forest Score",   VIOLET),
                        (svm_sc, "One-Class SVM Score",      CORAL),
                    ]:
                        fig_sc.add_trace(go.Histogram(
                            x=_norm(sc), name=name, opacity=0.65,
                            marker_color=color, nbinsx=50
                        ))
                    fig_sc.update_layout(**_base_layout(
                        title=dict(text="ANOMALY SCORE DISTRIBUTIONS (normalised 0–1)",
                                   font=dict(color=CYAN, family="Orbitron", size=13)),
                        barmode="overlay", height=420,
                        xaxis_title="Normalised Anomaly Score",
                        yaxis_title="Count"
                    ))
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.success("Score comparison complete. Add a Class column for full metrics.")
                except Exception as e:
                    st.error(f"Comparison error: {e}")

    else:
        st.markdown("""
        <div class="ng-card" style="margin-bottom:1rem;">
            Labels detected. Click below to train and compare all three models.
            One-Class SVM is capped at 3,000 samples for speed.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run Full Model Comparison", type="primary", key="run_full_cmp"):
            metrics, roc_data = {}, {}

            # SAE
            try:
                sae_thr = np.percentile(mse, threshold_pct)
                y_pred_sae = (mse > sae_thr).astype(int)
                metrics["Sparse Autoencoder"] = _calc_metrics(y_true, y_pred_sae, mse)
                fpr_, tpr_, _ = roc_curve(y_true, mse)
                roc_data["Sparse Autoencoder"] = (fpr_, tpr_, metrics["Sparse Autoencoder"]["AUC-ROC"])
                st.toast("SAE scored")
            except Exception as e:
                st.warning(f"SAE scoring failed: {e}")

            # Isolation Forest
            with st.spinner("Training Isolation Forest…"):
                try:
                    contam = min(max(float(y_true.mean()), 0.001), 0.5)
                    iso3 = IsolationForest(n_estimators=100, contamination=contam,
                                           random_state=42, n_jobs=-1)
                    iso3.fit(X_scaled)
                    iso_sc3 = -iso3.decision_function(X_scaled)
                    thr_i = np.percentile(iso_sc3, threshold_pct)
                    y_iso3 = (iso_sc3 > thr_i).astype(int)
                    metrics["Isolation Forest"] = _calc_metrics(y_true, y_iso3, iso_sc3)
                    fpr_, tpr_, _ = roc_curve(y_true, iso_sc3)
                    roc_data["Isolation Forest"] = (fpr_, tpr_, metrics["Isolation Forest"]["AUC-ROC"])
                    st.toast("Isolation Forest done")
                except Exception as e:
                    st.warning(f"Isolation Forest failed: {e}")

            # One-Class SVM
            with st.spinner("Training One-Class SVM (max 3,000 samples)…"):
                try:
                    nu = min(max(float(y_true.mean()), 0.001), 0.5)
                    svm3 = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
                    svm3.fit(X_scaled[:min(3000, len(X_scaled))])
                    svm_sc3 = -svm3.decision_function(X_scaled)
                    thr_s = np.percentile(svm_sc3, threshold_pct)
                    y_svm3 = (svm_sc3 > thr_s).astype(int)
                    metrics["One-Class SVM"] = _calc_metrics(y_true, y_svm3, svm_sc3)
                    fpr_, tpr_, _ = roc_curve(y_true, svm_sc3)
                    roc_data["One-Class SVM"] = (fpr_, tpr_, metrics["One-Class SVM"]["AUC-ROC"])
                    st.toast("One-Class SVM done")
                except Exception as e:
                    st.warning(f"One-Class SVM failed: {e}")

            if metrics:
                _draw_comparison(metrics, roc_data)
            else:
                st.error("All models failed. Check that your data has numeric features.")
        else:
            st.markdown("""
            <div class="ng-card" style="text-align:center; padding:2.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.8rem; opacity:0.25;">⚔️</div>
                <div style="font-family:'Orbitron',monospace; font-size:0.82rem;
                             color:#334155; letter-spacing:0.08em;">
                    AWAITING COMPARISON RUN
                </div>
                <div style="font-size:0.78rem; color:#1e293b; margin-top:0.4rem;">
                    Click the button above to train all three models and compare
                </div>
            </div>
            """, unsafe_allow_html=True)
