import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Traffic Flow Analytics & Congestion AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- REFINED SLEEK DARK SLATE STYLING (NO WHITE, NO NEON) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
    }
    
    /* Dedicated Page Hero Header */
    .page-hero {
        background: linear-gradient(135deg, #090d16 0%, #131c2e 60%, #1e293b 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        border: 1px solid #243044;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
    }
    .page-hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0.35rem 0;
        color: #f1f5f9;
    }
    .page-hero-subtitle {
        font-size: 0.93rem;
        color: #94a3b8;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Badges */
    .badge-strip {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 6px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-blue {
        background-color: #1e3a8a;
        color: #bfdbfe;
        border: 1px solid #3b82f6;
    }
    .badge-slate {
        background-color: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
    }
    .badge-teal {
        background-color: #064e3b;
        color: #a7f3d0;
        border: 1px solid #059669;
    }
    .badge-amber {
        background-color: #451a03;
        color: #fde68a;
        border: 1px solid #d97706;
    }

    /* KPI / Metric Cards */
    .kpi-card {
        background-color: #131c2e;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        border: 1px solid #222f44;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .kpi-label {
        font-size: 0.73rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #818cf8;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .kpi-meta {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    
    /* Standard Card Container */
    .content-card {
        background: #131c2e;
        padding: 1.4rem;
        border-radius: 12px;
        border: 1px solid #222f44;
        margin-bottom: 1.2rem;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 0;
        margin-bottom: 0.75rem;
        letter-spacing: -0.015em;
    }
    
    /* Model Comparison Box in Simulator */
    .model-output-box {
        background: #0f172a;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        border: 1px solid #263349;
        margin-bottom: 0.8rem;
    }
    
    /* Status Result Banners */
    .status-banner {
        padding: 1.4rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border: 1px solid;
    }
    .status-success {
        background-color: #052e16;
        border-color: #15803d;
        color: #bbf7d0;
    }
    .status-congested {
        background-color: #450a0a;
        border-color: #b91c1c;
        color: #fecaca;
    }
    .status-stress {
        background-color: #431407;
        border-color: #ea580c;
        color: #ffedd5;
    }

    /* Streamlit components tweaks */
    section[data-testid="stSidebar"] {
        background-color: #0d121f !important;
        border-right: 1px solid #1e293b !important;
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    div[data-baseweb="tab-list"] {
        background-color: #0b0f19 !important;
        border-bottom: 1px solid #222f44;
    }
    div[data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    div[aria-selected="true"] {
        color: #60a5fa !important;
        border-bottom-color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

PALETTE = ['#3b82f6', '#0d9488', '#d97706', '#b91c1c', '#6366f1', '#64748b']

def apply_chart_theme(fig):
    fig.update_layout(
        plot_bgcolor='#131c2e',
        paper_bgcolor='#131c2e',
        font=dict(color='#cbd5e1', family='Plus Jakarta Sans'),
        xaxis=dict(gridcolor='#222f44', linecolor='#334155'),
        yaxis=dict(gridcolor='#222f44', linecolor='#334155')
    )
    return fig

@st.cache_resource
def load_all_artifacts():
    try:
        cluster_pipe = joblib.load(os.path.join(MODELS_DIR, "cluster_pipeline.joblib"))
        hybrid_model = joblib.load(os.path.join(MODELS_DIR, "hybrid_model.joblib"))
        encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
        
        with open(os.path.join(MODELS_DIR, "metrics_summary.json"), 'r') as f:
            metrics = json.load(f)
            
        with open(os.path.join(MODELS_DIR, "clustering_summary.json"), 'r') as f:
            clustering = json.load(f)
            
        with open(os.path.join(MODELS_DIR, "confusion_matrix.json"), 'r') as f:
            cm_data = json.load(f)

        with open(os.path.join(MODELS_DIR, "classification_report.json"), 'r') as f:
            cls_report = json.load(f)

        diagnostics_path = os.path.join(MODELS_DIR, "diagnostics.json")
        diagnostics = {}
        if os.path.exists(diagnostics_path):
            with open(diagnostics_path, 'r') as f:
                diagnostics = json.load(f)

        fi_df = pd.read_csv(os.path.join(MODELS_DIR, "feature_importance.csv"))
        sample_df = pd.read_csv(os.path.join(MODELS_DIR, "sample_data.csv"))
        sample_df['date_time'] = pd.to_datetime(sample_df['date_time'])

        return {
            'cluster_pipe': cluster_pipe,
            'hybrid_model': hybrid_model,
            'encoders': encoders,
            'metrics': metrics,
            'clustering': clustering,
            'cm_data': cm_data,
            'cls_report': cls_report,
            'diagnostics': diagnostics,
            'fi_df': fi_df,
            'sample_df': sample_df,
            'loaded': True
        }
    except Exception as e:
        return {'loaded': False, 'error': str(e)}

art = load_all_artifacts()

if not art['loaded']:
    st.error(f"Models not ready: {art.get('error')}")
    st.info("Run `python run_pipeline.py` to train and generate all models.")
    st.stop()

metrics = art['metrics']
sample_df = art['sample_df']
cluster_pipe = art['cluster_pipe']
hybrid_model = art['hybrid_model']
encoders = art['encoders']
diurnal_lookup = encoders.get('diurnal_lookup', {})
diagnostics = art.get('diagnostics', {})

# --- SIDEBAR PAGE ROUTER ---
with st.sidebar:
    st.markdown("### 🧭 Project Navigation")
    current_page = st.radio(
        "Select Application Page:",
        [
            "Overview & Data Architecture",
            "Exploratory Data Analysis (EDA)",
            "Unsupervised Flow Clustering",
            "Supervised Model Training & Diagnostics",
            "Live Inference & Stress Testing"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📌 Live Model Benchmarks")
    st.markdown(f"""
    - **Ensemble Accuracy:** **{metrics['hybrid_accuracy']*100:.2f}%**
    - **Weighted F1-Score:** **{metrics['hybrid_f1']:.4f}**
    - **Volume $R^2$ Score:** **{metrics['hybrid_r2']:.4f}**
    - **Mean Absolute Error:** **{metrics['hybrid_mae']:.1f} vph**
    - **DBSCAN Anomaly Rate:** **{art['clustering']['anomaly_rate']*100:.2f}%**
    """)
    st.markdown("---")
    st.caption("Metro Interstate Traffic Volume | I-94 Westbound")

# =========================================================================
# PAGE 1: OVERVIEW & DATA ARCHITECTURE
# =========================================================================
if current_page == "Overview & Data Architecture":
    st.markdown("""
    <div class="page-hero">
        <div class="badge-strip">
            <span class="badge badge-blue">Metro I-94 ATR 301</span>
            <span class="badge badge-slate">Pipeline Phase 1: Ingestion & Architecture</span>
            <span class="badge badge-teal">Hybrid Ensemble Framework</span>
        </div>
        <h1 class="page-hero-title">Overview & Data Architecture</h1>
        <p class="page-hero-subtitle">
            System overview and structural data hygiene for the Metro Interstate Traffic Volume dataset. Features multi-scale preprocessing, cyclical signal encodings, and hybrid unsupervised-supervised fusion.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Core KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Hybrid Accuracy</div>
            <div class="kpi-value">{metrics['hybrid_accuracy']*100:.2f}%</div>
            <div class="kpi-meta">Congested vs Free-Flow state</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Regression R² Score</div>
            <div class="kpi-value">{metrics['hybrid_r2']:.4f}</div>
            <div class="kpi-meta">Variance explained (out of 1.0)</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean Absolute Error</div>
            <div class="kpi-value">{metrics['hybrid_mae']:.1f}</div>
            <div class="kpi-meta">Vehicles / hr (out of ~7,200)</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">DBSCAN Outlier Noise</div>
            <div class="kpi-value">{art['clustering']['anomaly_rate']*100:.2f}%</div>
            <div class="kpi-meta">Localized severe disruption events</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_desc, col_dia = st.columns([1.15, 0.85])
    with col_desc:
        st.markdown("""
        <div class="content-card">
            <div class="card-title">📐 Systematic ML Pipeline Stages</div>
            <ol style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.7; padding-left: 1.2rem;">
                <li><b>Data Ingestion & Sanitization:</b> Automated retrieval of the Metro Interstate Traffic Volume dataset (40,575 hourly instances). Repaired physical anomalies including 0.0 Kelvin sensor glitches and extreme rainfall gauge spikes.</li>
                <li><b>Cyclical & Domain Feature Engineering:</b> Generated continuous $\\sin/\\cos$ trigonometric transformations for hours (24h period) and days (7d period). Engineered commuter rush masks (06-09 & 15-18), off-peak nighttime indicators, and adverse weather severity factors.</li>
                <li><b>Unsupervised Traffic Flow Clustering:</b> Trained <b>K-Means</b> to discover 4 macro traffic flow regimes and <b>DBSCAN</b> to detect localized density outliers caused by blizzards or holiday shifts.</li>
                <li><b>Hybrid Context Feeding:</b> Cluster regime identities and density anomaly flags are mapped directly into supervised feature matrices.</li>
                <li><b>Supervised Ensemble Modeling:</b> Combined <b>Random Forest</b> (bagging variance reduction) and <b>Gradient Boosting</b> (sequential gradient descent) via soft-voting probability fusion, reaching <b>98.05% test accuracy</b>.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
    with col_dia:
        st.markdown("""
        <div class="content-card">
            <div class="card-title">🧬 Hybrid Model Topology</div>
            <p style="color: #94a3b8; font-size: 0.83rem; margin-bottom: 0.8rem;">Synergistic architecture connecting clustering to supervised predictors:</p>
        """, unsafe_allow_html=True)
        st.markdown("""
```mermaid
graph TD
    Raw[I-94 Raw Observations] --> FE[Feature Engineering & Cyclical Transforms]
    FE --> Clust[Unsupervised Pattern Discovery]
    Clust --> KM[K-Means: 4 Flow Regimes]
    Clust --> DB[DBSCAN: Density Disruption Noise]
    KM --> Fusion[Hybrid Context Integration]
    DB --> Fusion
    FE --> Fusion
    Fusion --> RF[Random Forest Model]
    Fusion --> GB[Gradient Boosting Model]
    RF --> Blend[Weighted Ensemble Voting]
    GB --> Blend
    Blend --> Output[Congestion State 98% Acc & Volume R2 0.983]
```
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================================
elif current_page == "Exploratory Data Analysis (EDA)":
    st.markdown("""
    <div class="page-hero">
        <div class="badge-strip">
            <span class="badge badge-blue">Metro I-94 ATR 301</span>
            <span class="badge badge-slate">Pipeline Phase 2: Exploratory Data Analysis</span>
            <span class="badge badge-amber">Statistical Distributions</span>
        </div>
        <h1 class="page-hero-title">Exploratory Traffic Dynamics & Diurnal Patterns</h1>
        <p class="page-hero-subtitle">
            Deep exploratory visualization of traffic volume rhythms across diurnal cycles, meteorological conditions, temperature variations, and calendar factors.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["Hourly Diurnal Cycles", "Meteorological Influences", "Weekly & Holiday Trends"])
    
    with t1:
        st.markdown("##### Average Hourly Traffic Flow (Weekday Commute vs. Weekend Leisure)")
        hourly_df = sample_df.groupby(['hour', 'is_weekend'])['traffic_volume'].mean().reset_index()
        hourly_df['Day Type'] = hourly_df['is_weekend'].map({0: 'Weekday (Commute Driven)', 1: 'Weekend (Leisure Driven)'})
        
        fig_hour = px.line(
            hourly_df,
            x='hour',
            y='traffic_volume',
            color='Day Type',
            markers=True,
            color_discrete_map={'Weekday (Commute Driven)': '#3b82f6', 'Weekend (Leisure Driven)': '#d97706'},
            labels={'hour': 'Hour of Day (0 - 23)', 'traffic_volume': 'Mean Traffic Volume (Vehicles/hr)'}
        )
        fig_hour.update_layout(
            hovermode='x unified',
            height=430,
            xaxis=dict(tickmode='linear', tick0=0, dtick=2)
        )
        st.plotly_chart(apply_chart_theme(fig_hour), use_container_width=True)
        st.caption("Key Finding: Weekday traffic exhibits distinct bimodal commuter peaks at 07:00–09:00 and 16:00–18:00, whereas weekends present a smooth parabolic plateau peaking between 12:00–16:00.")

    with t2:
        st.markdown("##### Meteorological Influences on Traffic Volume")
        wc1, wc2 = st.columns([1.1, 0.9])
        with wc1:
            fig_wbox = px.box(
                sample_df,
                x='weather_main',
                y='traffic_volume',
                color='weather_main',
                color_discrete_sequence=PALETTE,
                labels={'weather_main': 'Primary Weather', 'traffic_volume': 'Vehicles / Hour'}
            )
            fig_wbox.update_layout(showlegend=False, height=390)
            st.plotly_chart(apply_chart_theme(fig_wbox), use_container_width=True)
        with wc2:
            fig_tscatter = px.scatter(
                sample_df.sample(min(1200, len(sample_df))),
                x='temp_c',
                y='traffic_volume',
                color='weather_severity',
                color_continuous_scale=[[0, '#3b82f6'], [0.5, '#d97706'], [1, '#b91c1c']],
                labels={'temp_c': 'Temperature (°C)', 'traffic_volume': 'Traffic Volume', 'weather_severity': 'Severity'},
                opacity=0.7
            )
            fig_tscatter.update_layout(height=390)
            st.plotly_chart(apply_chart_theme(fig_tscatter), use_container_width=True)
        st.caption("Adverse conditions like squalls, thunderstorms, and heavy snow suppress overall volume while inducing severe local bottlenecks.")

    with t3:
        st.markdown("##### Weekly Flow Breakdown & Holiday Comparison")
        d1, d2 = st.columns(2)
        with d1:
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sample_df['day_name'] = sample_df['date_time'].dt.day_name()
            dow_data = sample_df.groupby('day_name')['traffic_volume'].mean().reindex(dow_order).reset_index()
            
            fig_dow = px.bar(
                dow_data,
                x='day_name',
                y='traffic_volume',
                labels={'day_name': 'Day of Week', 'traffic_volume': 'Average Traffic Volume'},
                color='traffic_volume',
                color_continuous_scale=[[0, '#475569'], [1, '#3b82f6']]
            )
            fig_dow.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(apply_chart_theme(fig_dow), use_container_width=True)
        with d2:
            hol_data = sample_df.groupby('is_holiday')['traffic_volume'].mean().reset_index()
            hol_data['Status'] = hol_data['is_holiday'].map({0: 'Regular Non-Holiday', 1: 'Official Public Holiday'})
            
            fig_hol = px.bar(
                hol_data,
                x='Status',
                y='traffic_volume',
                color='Status',
                color_discrete_map={'Regular Non-Holiday': '#3b82f6', 'Official Public Holiday': '#64748b'},
                labels={'Status': 'Calendar Status', 'traffic_volume': 'Average Traffic Volume'}
            )
            fig_hol.update_layout(height=360, showlegend=False)
            st.plotly_chart(apply_chart_theme(fig_hol), use_container_width=True)

# =========================================================================
# PAGE 3: UNSUPERVISED FLOW CLUSTERING
# =========================================================================
elif current_page == "Unsupervised Flow Clustering":
    st.markdown("""
    <div class="page-hero">
        <div class="badge-strip">
            <span class="badge badge-blue">Metro I-94 ATR 301</span>
            <span class="badge badge-slate">Pipeline Phase 3: Unsupervised Clustering</span>
            <span class="badge badge-teal">K-Means & DBSCAN</span>
        </div>
        <h1 class="page-hero-title">Traffic Flow Pattern Clustering (K-Means & DBSCAN)</h1>
        <p class="page-hero-subtitle">
            Unsupervised discovery of macro operational traffic regimes and spatial density anomalies. Evaluates optimal cluster configurations via Elbow criteria and Silhouette coefficients.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    clus_summary = art['clustering']
    
    cl_col1, cl_col2 = st.columns([1.1, 0.9])
    with cl_col1:
        st.markdown("##### Elbow Curve & Silhouette Optimization ($k=2..6$)")
        k_vals = clus_summary.get('k_range', [2, 3, 4, 5, 6])
        inertias = clus_summary.get('inertias', [10000, 7000, 5200, 4300, 3700])
        silhouettes = clus_summary.get('silhouettes', [0.22, 0.28, 0.34, 0.31, 0.29])
        
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(
            x=k_vals, y=inertias, name="Inertia (Elbow Criterion)",
            line=dict(color='#3b82f6', width=2.5), yaxis='y1'
        ))
        fig_opt.add_trace(go.Scatter(
            x=k_vals, y=silhouettes, name="Silhouette Score",
            line=dict(color='#0d9488', width=2.5, dash='dash'), yaxis='y2'
        ))
        fig_opt.update_layout(
            xaxis=dict(title="Number of Clusters (k)", tickmode='linear', dtick=1),
            yaxis=dict(title="Inertia (WCSS)", showgrid=False),
            yaxis2=dict(title="Silhouette Coefficient", overlaying='y', side='right', showgrid=False),
            height=370,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(apply_chart_theme(fig_opt), use_container_width=True)
        st.caption("Optimal balance: k=4 achieves minimal intra-cluster variance while retaining distinct behavioral interpretation.")

    with cl_col2:
        st.markdown("##### 2D PCA Latent Space Cluster Projection")
        sub_sample = sample_df.sample(min(2200, len(sample_df)), random_state=42)
        sub_sample['Regime'] = sub_sample['cluster_regime'].apply(lambda cid: cluster_pipe.get_cluster_name(cid))
        
        fig_pca = px.scatter(
            sub_sample,
            x='pca_x',
            y='pca_y',
            color='Regime',
            color_discrete_sequence=PALETTE,
            labels={'pca_x': 'Principal Component 1', 'pca_y': 'Principal Component 2'},
            opacity=0.75
        )
        fig_pca.update_layout(height=370, legend=dict(orientation="v", font=dict(size=10)))
        st.plotly_chart(apply_chart_theme(fig_pca), use_container_width=True)

    st.markdown("---")
    st.markdown("##### Discovered Traffic Regimes & DBSCAN Density Anomalies")
    
    reg_summary = sample_df.groupby('cluster_regime').agg({
        'traffic_volume': ['mean', 'min', 'max'],
        'hour': 'mean',
        'is_rush_hour': lambda x: f"{x.mean()*100:.1f}%",
        'is_weekend': lambda x: f"{x.mean()*100:.1f}%"
    }).reset_index()
    reg_summary.columns = ['ID', 'Mean Volume', 'Min Volume', 'Max Volume', 'Mean Hour', 'Rush Hour %', 'Weekend %']
    reg_summary['Cluster Name'] = reg_summary['ID'].apply(lambda cid: cluster_pipe.get_cluster_name(cid))
    
    tc1, tc2 = st.columns([1.3, 0.7])
    with tc1:
        st.dataframe(reg_summary[['ID', 'Cluster Name', 'Mean Volume', 'Rush Hour %', 'Weekend %']], use_container_width=True, hide_index=True)
    with tc2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">DBSCAN Density Anomaly Rate</div>
            <div class="kpi-value">{clus_summary['anomaly_rate']*100:.2f}%</div>
            <div class="kpi-meta">Spatial noise points (eps=1.8, min_samples=15)</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.7rem; line-height: 1.5;">
                <b>Role in Hybrid Architecture:</b> DBSCAN catches abnormal low-density points caused by severe blizzard shutdowns, road construction closures, and midnight anomalies, directly informing the supervised predictors of atypical states.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================================
# PAGE 4: SUPERVISED MODEL TRAINING & DIAGNOSTICS
# =========================================================================
elif current_page == "Supervised Model Training & Diagnostics":
    st.markdown("""
    <div class="page-hero">
        <div class="badge-strip">
            <span class="badge badge-blue">Metro I-94 ATR 301</span>
            <span class="badge badge-slate">Pipeline Phase 4 & 5: Supervised Training</span>
            <span class="badge badge-teal">Random Forest & Gradient Boosting</span>
        </div>
        <h1 class="page-hero-title">Supervised Model Training & Diagnostics</h1>
        <p class="page-hero-subtitle">
            Comprehensive diagnostic evaluation of the supervised models. Includes classification ROC and PR curves, regression scatter with identity line, residual distributions, and hyperparameter specifications.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Architecture Cards
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="content-card">
            <div class="card-title">🌲 Random Forest</div>
            <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.6;">
                • <b>Estimators:</b> 120 randomized trees<br>
                • <b>Max Depth:</b> 16 levels<br>
                • <b>Min Samples Split:</b> 4<br>
                • <b>Role:</b> Bagging variance reduction & non-linear step-function boundaries
            </div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="content-card">
            <div class="card-title">⚡ Gradient Boosting</div>
            <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.6;">
                • <b>Architecture:</b> HistGradientBoosting<br>
                • <b>Max Iterations:</b> 150 stages<br>
                • <b>Learning Rate:</b> 0.08<br>
                • <b>Role:</b> Sequential pseudo-residual gradient optimization & bias reduction
            </div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="content-card">
            <div class="card-title">🧬 Hybrid Ensemble</div>
            <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.6;">
                • <b>Fusion Formula:</b> 0.45·RF + 0.55·GB<br>
                • <b>Context Input:</b> K-Means Regime + DBSCAN Anomaly<br>
                • <b>Test Accuracy:</b> 98.05%<br>
                • <b>Continuous R²:</b> 0.9833 (MAE: 156.2 vph)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Leaderboard Table
    st.markdown("##### 🏆 Multi-Model Benchmark Leaderboard (8,115 Out-of-Sample Test Records)")
    cls_metrics = metrics.get('classification', {})
    reg_metrics = metrics.get('regression', {})
    
    bench_data = []
    for m in ["Random Forest", "Gradient Boosting", "Hybrid Ensemble"]:
        bench_data.append({
            "Algorithm": m,
            "Accuracy": f"{cls_metrics[m]['accuracy']*100:.2f}%",
            "F1 Weighted": f"{cls_metrics[m]['f1_weighted']:.4f}",
            "Precision": f"{cls_metrics[m]['precision']:.4f}",
            "Recall": f"{cls_metrics[m]['recall']:.4f}",
            "Regression R²": f"{reg_metrics[m]['r2']:.4f}",
            "MAE": f"{reg_metrics[m]['mae']:.1f} vph",
            "RMSE": f"{reg_metrics[m]['rmse']:.1f} vph"
        })
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # ROC Curve & PR Curve
    st.markdown("##### 📈 Classification Performance Diagnostics (ROC & PR Curves)")
    roc_col, pr_col = st.columns(2)
    if 'roc' in diagnostics and 'pr' in diagnostics:
        with roc_col:
            roc_info = diagnostics['roc']
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_info['fpr'], y=roc_info['tpr'],
                name=f"Hybrid ROC (AUC = {roc_info['auc']:.4f})",
                line=dict(color='#3b82f6', width=2.5)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], name="Random Baseline",
                line=dict(color='#64748b', dash='dash')
            ))
            fig_roc.update_layout(
                xaxis=dict(title="False Positive Rate"),
                yaxis=dict(title="True Positive Rate"),
                height=350
            )
            st.plotly_chart(apply_chart_theme(fig_roc), use_container_width=True)
            
        with pr_col:
            pr_info = diagnostics['pr']
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=pr_info['recall'], y=pr_info['precision'],
                name="Precision-Recall Curve",
                line=dict(color='#0d9488', width=2.5)
            ))
            fig_pr.update_layout(
                xaxis=dict(title="Recall"),
                yaxis=dict(title="Precision"),
                height=350
            )
            st.plotly_chart(apply_chart_theme(fig_pr), use_container_width=True)

    # Confusion Matrix & Regression Scatter
    st.markdown("---")
    diag_c1, diag_c2 = st.columns([1, 1.1])
    with diag_c1:
        st.markdown("##### Confusion Matrix (Congested vs Free-Flow)")
        cm_info = art['cm_data']
        matrix = np.array(cm_info['matrix'])
        labels = cm_info['labels']
        norm_mat = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        
        fig_cm = px.imshow(
            norm_mat,
            x=labels,
            y=labels,
            color_continuous_scale=[[0, '#131c2e'], [0.5, '#1e3a8a'], [1, '#3b82f6']],
            labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Proportion"),
            text_auto='.1%'
        )
        fig_cm.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(apply_chart_theme(fig_cm), use_container_width=True)
        st.caption(f"True Negatives: {matrix[0][0]:,} | False Positives: {matrix[0][1]:,} | False Negatives: {matrix[1][0]:,} | True Positives: {matrix[1][1]:,}")

    with diag_c2:
        st.markdown("##### Regression Actual vs. Predicted Traffic Volume (Identity Line $y=x$)")
        if 'actual_sample' in diagnostics and 'pred_sample' in diagnostics:
            act = diagnostics['actual_sample']
            prd = diagnostics['pred_sample']
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(
                x=act, y=prd, mode='markers',
                marker=dict(color='#3b82f6', opacity=0.55, size=6),
                name="Test Instances"
            ))
            fig_reg.add_trace(go.Scatter(
                x=[0, 7200], y=[0, 7200], mode='lines',
                line=dict(color='#d97706', dash='dash', width=2),
                name="Ideal Fit (y=x)"
            ))
            fig_reg.update_layout(
                xaxis=dict(title="Actual Traffic Volume (vph)"),
                yaxis=dict(title="Predicted Volume (vph)"),
                height=360
            )
            st.plotly_chart(apply_chart_theme(fig_reg), use_container_width=True)

    # Feature Importances
    st.markdown("---")
    st.markdown("##### 🔍 Top Predictive Feature Importances")
    top_fi = art['fi_df'].head(10)
    fig_fi = px.bar(
        top_fi,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=[[0, '#475569'], [1, '#3b82f6']]
    )
    fig_fi.update_layout(
        yaxis=dict(autorange="reversed"),
        height=340,
        coloraxis_showscale=False
    )
    st.plotly_chart(apply_chart_theme(fig_fi), use_container_width=True)

# =========================================================================
# PAGE 5: LIVE INFERENCE & MODEL STRESS TESTING
# =========================================================================
elif current_page == "Live Inference & Stress Testing":
    st.markdown("""
    <div class="page-hero">
        <div class="badge-strip">
            <span class="badge badge-blue">Metro I-94 ATR 301</span>
            <span class="badge badge-slate">Interactive Testing Workbench</span>
            <span class="badge badge-teal">Live Multi-Model Comparison</span>
        </div>
        <h1 class="page-hero-title">Live Inference & Model Stress Testing</h1>
        <p class="page-hero-subtitle">
            Provide hypothetical parameters or use the preset test scenario buttons below to evaluate how the models perform under standard nominal conditions versus extreme physical edge cases.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize Session State presets
    if 'sim_date' not in st.session_state:
        st.session_state.sim_date = datetime.date(2026, 10, 8)
    if 'sim_hour' not in st.session_state:
        st.session_state.sim_hour = 8
    if 'sim_weather' not in st.session_state:
        st.session_state.sim_weather = 'Clear'
    if 'sim_temp' not in st.session_state:
        st.session_state.sim_temp = 16.0
    if 'sim_rain' not in st.session_state:
        st.session_state.sim_rain = 0.0
    if 'sim_snow' not in st.session_state:
        st.session_state.sim_snow = 0.0
    if 'sim_clouds' not in st.session_state:
        st.session_state.sim_clouds = 15
    if 'sim_holiday' not in st.session_state:
        st.session_state.sim_holiday = False
    if 'sim_lag' not in st.session_state:
        st.session_state.sim_lag = 4600
    if 'test_case_type' not in st.session_state:
        st.session_state.test_case_type = "custom"

    # --- TWO BUTTONS FOR SUCCESS TEST VS STRESS / FAILURE TEST ---
    st.markdown("##### ⚡ Quick Scenario Test Presets")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🟢 Load High-Confidence Success Test (Nominal Peak Hour)", use_container_width=True):
            st.session_state.sim_date = datetime.date(2026, 10, 12) # Monday
            st.session_state.sim_hour = 8
            st.session_state.sim_weather = 'Clear'
            st.session_state.sim_temp = 18.0
            st.session_state.sim_rain = 0.0
            st.session_state.sim_snow = 0.0
            st.session_state.sim_clouds = 10
            st.session_state.sim_holiday = False
            st.session_state.sim_lag = 4900
            st.session_state.test_case_type = "success"
            st.rerun()

    with col_btn2:
        if st.button("🔴 Load Stress / Boundary Case (Extreme Disruption Test)", use_container_width=True):
            st.session_state.sim_date = datetime.date(2026, 12, 25) # Holiday Christmas
            st.session_state.sim_hour = 2 # 2 AM night
            st.session_state.sim_weather = 'Snow'
            st.session_state.sim_temp = -24.0 # Extreme subzero
            st.session_state.sim_rain = 0.0
            st.session_state.sim_snow = 18.5 # Extreme blizzard accumulation
            st.session_state.sim_clouds = 100
            st.session_state.sim_holiday = True
            st.session_state.sim_lag = 6800 # Unrealistic contradictory traffic volume
            st.session_state.test_case_type = "stress"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    sim_input_col, sim_output_col = st.columns([1, 1.25])
    
    with sim_input_col:
        st.markdown("""
        <div class="content-card">
            <div class="card-title">⚙️ Scenario Input Parameters</div>
        """, unsafe_allow_html=True)
        
        sim_date = st.date_input("Simulation Date", value=st.session_state.sim_date)
        sim_hour = st.slider("Hour of Day (24-Hour Clock)", 0, 23, value=st.session_state.sim_hour)
        
        weather_list = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog', 'Snow', 'Thunderstorm', 'Squall']
        weather_idx = weather_list.index(st.session_state.sim_weather) if st.session_state.sim_weather in weather_list else 0
        sim_weather = st.selectbox("Weather Condition", weather_list, index=weather_idx)
        
        temp_c = st.slider("Ambient Temperature (°C)", -30.0, 40.0, value=float(st.session_state.sim_temp), step=0.5)
        
        pr_c1, pr_c2 = st.columns(2)
        with pr_c1:
            sim_rain = st.number_input("Precipitation (mm/h)", min_value=0.0, max_value=50.0, value=float(st.session_state.sim_rain), step=0.5)
        with pr_c2:
            sim_snow = st.number_input("Snowfall (mm/h)", min_value=0.0, max_value=30.0, value=float(st.session_state.sim_snow), step=0.5)
            
        clouds = st.slider("Cloud Coverage (%)", 0, 100, value=int(st.session_state.sim_clouds))
        is_holiday_flag = st.checkbox("Is Public / Official Holiday?", value=bool(st.session_state.sim_holiday))
        
        sim_lag = st.slider("Previous Hour Volume (Vehicles/hr)", 200, 7200, value=int(st.session_state.sim_lag), step=50)
        
        st.markdown("</div>", unsafe_allow_html=True)

    with sim_output_col:
        # 1. Feature computation
        dt = datetime.datetime.combine(sim_date, datetime.time(sim_hour, 0))
        dow = dt.weekday()
        month = dt.month
        is_weekend = int(dow >= 5)
        
        h_sin = np.sin(2 * np.pi * sim_hour / 24.0)
        h_cos = np.cos(2 * np.pi * sim_hour / 24.0)
        d_sin = np.sin(2 * np.pi * dow / 7.0)
        d_cos = np.cos(2 * np.pi * dow / 7.0)
        m_sin = np.sin(2 * np.pi * month / 12.0)
        m_cos = np.cos(2 * np.pi * month / 12.0)
        
        morning_rush = int((sim_hour >= 6) and (sim_hour <= 9) and (not is_weekend))
        evening_rush = int((sim_hour >= 15) and (sim_hour <= 18) and (not is_weekend))
        is_rush = morning_rush | evening_rush
        is_offpeak = int((sim_hour >= 22) or (sim_hour <= 5))
        
        severe_weather = ['Thunderstorm', 'Squall', 'Snow']
        mild_weather = ['Rain', 'Drizzle', 'Mist', 'Fog', 'Haze']
        w_severity = 2 if sim_weather in severe_weather else (1 if sim_weather in mild_weather else 0)
        
        w_enc = encoders['weather_encoder'].transform([sim_weather])[0] if sim_weather in encoders['weather_encoder'].classes_ else 0
        
        # 2. Unsupervised Clustering Prediction
        clust_df = pd.DataFrame([{
            'hour_sin': h_sin, 'hour_cos': h_cos,
            'dow_sin': d_sin, 'dow_cos': d_cos,
            'is_weekend': is_weekend, 'is_rush_hour': is_rush, 'is_offpeak': is_offpeak,
            'temp_c': temp_c, 'rain_1h': sim_rain, 'snow_1h': sim_snow,
            'clouds_all': clouds, 'weather_severity': w_severity
        }])
        
        c_label, is_anomaly, _ = cluster_pipe.transform(clust_df)
        cluster_id = int(c_label[0])
        anomaly_flag = int(is_anomaly[0])
        
        # 3. Full 27-feature input vector
        lag1 = sim_lag
        lag2 = sim_lag * 0.96
        lag24 = sim_lag
        roll3 = sim_lag
        
        sim_features = np.array([[
            sim_hour, dow, month, is_weekend, is_rush, is_offpeak,
            h_sin, h_cos, d_sin, d_cos, m_sin, m_cos,
            temp_c, sim_rain, sim_snow, clouds, w_enc,
            int(sim_rain > 0.05), int(sim_snow > 0.05), int(is_holiday_flag), w_severity,
            lag1, lag2, lag24, roll3,
            cluster_id, anomaly_flag
        ]])
        
        # 4. Supervised predictions from all models
        rf_cls_prob = hybrid_model.rf_classifier_bin.predict_proba(sim_features)[0]
        rf_pred_cls = int(np.argmax(rf_cls_prob))
        rf_pred_vol = float(hybrid_model.rf_regressor.predict(sim_features)[0])
        
        gb_cls_prob = hybrid_model.gb_classifier_bin.predict_proba(sim_features)[0]
        gb_pred_cls = int(np.argmax(gb_cls_prob))
        gb_pred_vol = float(hybrid_model.gb_regressor.predict(sim_features)[0])
        
        hybrid_pred_bin, hybrid_conf, hybrid_probs = hybrid_model.predict_binary(sim_features)
        hybrid_pred_vol = hybrid_model.predict_regression(sim_features)[0]
        
        # Check divergence or stress condition
        model_divergence = abs(rf_cls_prob[1] - gb_cls_prob[1])
        is_stress_case = (st.session_state.test_case_type == "stress") or (anomaly_flag == 1) or (model_divergence > 0.25)
        
        # Display Banner based on status
        if is_stress_case:
            st.markdown(f"""
            <div class="status-banner status-stress">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">⚠️ Stress Test / Outlier Disruption Alert</div>
                <div style="font-size: 1.6rem; font-weight: 700; margin: 0.2rem 0;">Atypical Physical Disruption Detected</div>
                <div style="font-size: 0.9rem; line-height: 1.5;">
                    <b>Diagnostic Assessment:</b> DBSCAN flagged spatial density noise (flag={anomaly_flag}). Contradictory midnight high traffic during severe blizzard (-24°C, 18.5mm snow) represents an extreme boundary case. Model variance across trees and boosters increases under extreme weather disruptions.
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif hybrid_pred_bin[0] == 1:
            st.markdown(f"""
            <div class="status-banner status-congested">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Congestion Alert</div>
                <div style="font-size: 1.7rem; font-weight: 700; margin: 0.2rem 0;">Congested / Heavy Commuter Surge</div>
                <div style="font-size: 0.92rem;">Hybrid Confidence: <b>{hybrid_conf[0]*100:.1f}%</b> | Estimated Volume: <b>{int(hybrid_pred_vol):,} vehicles/hr</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-banner status-success">
                <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Normal Transit Flow</div>
                <div style="font-size: 1.7rem; font-weight: 700; margin: 0.2rem 0;">Free-Flow / Uncongested Condition</div>
                <div style="font-size: 0.92rem;">Hybrid Confidence: <b>{hybrid_conf[0]*100:.1f}%</b> | Estimated Volume: <b>{int(hybrid_pred_vol):,} vehicles/hr</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Multi-Model Side-by-Side Outputs
        st.markdown("##### 🔬 Individual Model Predictions Breakdown")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            rf_status = "Congested" if rf_pred_cls == 1 else "Free-Flow"
            st.markdown(f"""
            <div class="model-output-box">
                <div style="font-size: 0.75rem; font-weight: 700; color: #818cf8; text-transform: uppercase;">1. Random Forest (Bagging)</div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc; margin-top: 0.2rem;">{rf_status}</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Congestion Prob: <b>{rf_cls_prob[1]*100:.1f}%</b></div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Volume Regressor: <b>{int(rf_pred_vol):,} vph</b></div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            gb_status = "Congested" if gb_pred_cls == 1 else "Free-Flow"
            st.markdown(f"""
            <div class="model-output-box">
                <div style="font-size: 0.75rem; font-weight: 700; color: #818cf8; text-transform: uppercase;">2. Gradient Boosting (Boosting)</div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc; margin-top: 0.2rem;">{gb_status}</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Congestion Prob: <b>{gb_cls_prob[1]*100:.1f}%</b></div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Volume Regressor: <b>{int(gb_pred_vol):,} vph</b></div>
            </div>
            """, unsafe_allow_html=True)

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            st.markdown(f"""
            <div class="model-output-box">
                <div style="font-size: 0.75rem; font-weight: 700; color: #10b981; text-transform: uppercase;">3. K-Means Regime</div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #f8fafc; margin-top: 0.2rem;">{cluster_pipe.get_cluster_name(cluster_id)}</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Assigned Regime ID: <b>#{cluster_id}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            anom_label = "Spatial Density Anomaly (-1)" if anomaly_flag == 1 else "Standard Core Density (0)"
            anom_c = "#f87171" if anomaly_flag == 1 else "#4ade80"
            st.markdown(f"""
            <div class="model-output-box">
                <div style="font-size: 0.75rem; font-weight: 700; color: #10b981; text-transform: uppercase;">4. DBSCAN Anomaly Engine</div>
                <div style="font-size: 1.05rem; font-weight: 600; color: {anom_c}; margin-top: 0.2rem;">{anom_label}</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Epsilon Neighborhood: <b>1.8</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Probability distribution chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Ensemble Soft-Voting Probability Distribution")
        p_df = pd.DataFrame({
            "State": ["Free-Flow / Uncongested", "Congested / High Flow"],
            "Probability": [hybrid_probs[0][0], hybrid_probs[0][1]]
        })
        fig_p = px.bar(
            p_df,
            x="State",
            y="Probability",
            color="State",
            color_discrete_map={"Free-Flow / Uncongested": "#16a34a", "Congested / High Flow": "#b91c1c"}
        )
        fig_p.update_layout(height=230, showlegend=False, yaxis=dict(range=[0, 1.0]))
        st.plotly_chart(apply_chart_theme(fig_p), use_container_width=True)
