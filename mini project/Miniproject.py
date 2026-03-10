import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced Data Mining System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.4rem; margin: 0; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460; border-radius: 10px;
        padding: 1.2rem; text-align: center; color: white;
    }
    .metric-card h2 { color: #e94560; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a8b2d8; margin: 0; font-size: 0.85rem; }
    .section-header {
        color: #e94560; font-size: 1.3rem; font-weight: 700;
        border-bottom: 2px solid #0f3460; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem;
    }
    .insight-box {
        background: #0f3460; border-left: 4px solid #e94560;
        padding: 0.8rem 1rem; border-radius: 6px; margin: 0.5rem 0; color: #e2e8f0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.5rem; font-weight: 600; width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Advanced Data Mining & Processing System</h1>
    <p>India Innovates 2026 · Hackathon Project · Senior Full-Stack Data Engineering</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")

    data_source = st.radio("📂 Data Source", ["Upload File", "Sample Dataset"])

    st.markdown("---")
    st.markdown("### 🧬 Mining Module")
    mining_mode = st.selectbox("Select Algorithm", [
        "Clustering (K-Means)",
        "Clustering (DBSCAN)",
        "Classification (Random Forest)",
        "Association Rules (Apriori)",
    ])

    st.markdown("---")
    st.markdown("### 🎛️ Parameters")

    if "K-Means" in mining_mode:
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    elif "DBSCAN" in mining_mode:
        eps_val = st.slider("DBSCAN Epsilon", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.slider("Min Samples", 2, 20, 5)
    elif "Random Forest" in mining_mode:
        target_col_placeholder = st.text_input("Target Column (leave blank to auto-detect)", "")
        n_estimators = st.slider("Trees", 10, 200, 100, 10)
    elif "Apriori" in mining_mode:
        min_support = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01)
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)

    contamination = st.slider("Outlier Contamination %", 1, 20, 5) / 100

    st.markdown("---")
    st.caption("🇮🇳 India Innovates 2026")

# ── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data
def load_sample():
    np.random.seed(42)
    n = 400
    df = pd.DataFrame({
        "CustomerID":  [f"C{str(i).zfill(4)}" for i in range(n)],
        "Age":         np.random.randint(18, 70, n),
        "Income":      np.random.normal(55000, 18000, n).round(2),
        "SpendScore":  np.random.randint(1, 101, n),
        "Purchases":   np.random.poisson(12, n),
        "Region":      np.random.choice(["North", "South", "East", "West"], n),
        "Category":    np.random.choice(["Electronics", "Fashion", "Food", "Books"], n),
        "Churn":       np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "Rating":      np.random.uniform(1, 5, n).round(1),
        "Tenure":      np.random.randint(1, 120, n),
    })
    # inject outliers
    df.loc[np.random.choice(n, 12, replace=False), "Income"] *= np.random.choice([5, -1], 12)
    return df

def clean_dataframe(df):
    report = []
    original_shape = df.shape

    # Drop duplicates
    dups = df.duplicated().sum()
    df = df.drop_duplicates()
    if dups: report.append(f"✅ Removed {dups} duplicate rows")

    # Fill missing
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    missing_total = df.isnull().sum().sum()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0] if len(cat_cols) else "Unknown")
    if missing_total: report.append(f"✅ Imputed {missing_total} missing values")

    report.append(f"✅ Shape: {original_shape} → {df.shape}")
    return df, report

def detect_outliers(df, contamination):
    num_df = df.select_dtypes(include=np.number)
    if num_df.empty or len(num_df) < 10:
        return df, pd.Series([False]*len(df)), []
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(num_df)
    mask = preds == -1
    df = df.copy()
    df["__outlier__"] = mask
    n_out = mask.sum()
    insights = [
        f"🔴 {n_out} outliers detected ({n_out/len(df)*100:.1f}% of data)",
        f"🟢 {len(df)-n_out} clean records remain",
    ]
    return df, mask, insights

def encode_features(df):
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

# ── Load Data ─────────────────────────────────────────────────────────────────
df_raw = None

if data_source == "Upload File":
    uploaded = st.file_uploader("📤 Upload CSV or JSON", type=["csv", "json"])
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_json(uploaded)
        st.success(f"✅ Loaded `{uploaded.name}` — {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
else:
    df_raw = load_sample()
    st.info("📊 Using built-in sample dataset (400 customers)")

# ── Main Pipeline ─────────────────────────────────────────────────────────────
if df_raw is not None:

    df_clean, clean_report = clean_dataframe(df_raw.copy())
    df_outliers, outlier_mask, outlier_insights = detect_outliers(df_clean, contamination)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in zip(
        [c1, c2, c3, c4, c5],
        [df_raw.shape[0], df_raw.shape[1], df_raw.isnull().sum().sum(),
         int(outlier_mask.sum()), df_raw.select_dtypes(include=np.number).shape[1]],
        ["Total Rows", "Columns", "Missing Values", "Outliers Found", "Numeric Cols"]
    ):
        col.markdown(f"""
        <div class="metric-card"><h2>{val}</h2><p>{label}</p></div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧹 Data Cleaning", "🔍 Outlier Detection",
        "⛏️ Data Mining", "📈 Trend Analysis", "📥 Download Report"
    ])

    # ── TAB 1 : Data Cleaning ─────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">🧹 Automated Data Cleaning</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Raw Data (first 50 rows)**")
            st.dataframe(df_raw.head(50), use_container_width=True, height=300)
        with col_b:
            st.markdown("**Cleaned Data (first 50 rows)**")
            st.dataframe(df_clean.head(50), use_container_width=True, height=300)

        st.markdown("**🛠️ Cleaning Log**")
        for r in clean_report:
            st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)

        st.markdown("**📋 Data Types & Stats**")
        col_x, col_y = st.columns(2)
        with col_x:
            st.dataframe(df_clean.dtypes.reset_index().rename(columns={"index":"Column", 0:"DType"}), use_container_width=True)
        with col_y:
            st.dataframe(df_clean.describe().T.round(2), use_container_width=True)

    # ── TAB 2 : Outlier Detection ─────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">🔍 Automated Outlier Detection (Isolation Forest)</div>', unsafe_allow_html=True)
        for ins in outlier_insights:
            st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

        num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            col_x_sel = st.selectbox("X Axis", num_cols, index=0)
            col_y_sel = st.selectbox("Y Axis", num_cols, index=min(1, len(num_cols)-1))

            plot_df = df_outliers.copy()
            plot_df["Status"] = plot_df["__outlier__"].map({True: "Outlier 🔴", False: "Normal 🟢"})
            fig = px.scatter(
                plot_df, x=col_x_sel, y=col_y_sel, color="Status",
                color_discrete_map={"Outlier 🔴": "#e94560", "Normal 🟢": "#00d4aa"},
                title="Outlier Detection Scatter Plot",
                template="plotly_dark", opacity=0.7
            )
            fig.update_layout(plot_bgcolor="#16213e", paper_bgcolor="#1a1a2e")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Outlier Records**")
        st.dataframe(df_outliers[df_outliers["__outlier__"] == True].drop(columns=["__outlier__"]), use_container_width=True)

    # ── TAB 3 : Data Mining ───────────────────────────────────────────────────
    with tab3:
        st.markdown(f'<div class="section-header">⛏️ {mining_mode}</div>', unsafe_allow_html=True)

        df_enc = encode_features(df_clean.drop(columns=["__outlier__"], errors="ignore"))
        num_features = df_enc.select_dtypes(include=np.number)

        # --- K-Means ---
        if "K-Means" in mining_mode:
            if len(num_features.columns) >= 2:
                scaler = StandardScaler()
                X = scaler.fit_transform(num_features.fillna(0))
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(X)
                df_clean = df_clean.copy()
                df_clean["Cluster"] = labels.astype(str)

                # Elbow
                inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in range(2, 11)]
                fig_elbow = px.line(x=list(range(2,11)), y=inertias, markers=True,
                    labels={"x":"K","y":"Inertia"}, title="Elbow Curve", template="plotly_dark")
                fig_elbow.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")

                # PCA scatter
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X)
                pca_df = pd.DataFrame(coords, columns=["PC1","PC2"])
                pca_df["Cluster"] = labels.astype(str)
                fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                    title="K-Means Clusters (PCA 2D)", template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Bold)
                fig_pca.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")

                c1, c2 = st.columns(2)
                c1.plotly_chart(fig_elbow, use_container_width=True)
                c2.plotly_chart(fig_pca, use_container_width=True)

                sil = silhouette_score(X, labels)
                st.markdown(f'<div class="insight-box">📊 Silhouette Score: <b>{sil:.4f}</b> (closer to 1 = better separation)</div>', unsafe_allow_html=True)

                cluster_stats = df_enc.copy()
                cluster_stats["Cluster"] = labels
                st.markdown("**Cluster Statistics**")
                st.dataframe(cluster_stats.groupby("Cluster").mean().round(2), use_container_width=True)

        # --- DBSCAN ---
        elif "DBSCAN" in mining_mode:
            scaler = StandardScaler()
            X = scaler.fit_transform(num_features.fillna(0))
            db = DBSCAN(eps=eps_val, min_samples=min_samples)
            labels = db.fit_predict(X)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
            pca_df = pd.DataFrame(coords, columns=["PC1","PC2"])
            pca_df["Cluster"] = [str(l) if l != -1 else "Noise" for l in labels]
            fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                title="DBSCAN Clusters", template="plotly_dark")
            fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
            st.plotly_chart(fig, use_container_width=True)
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            noise = (labels == -1).sum()
            st.markdown(f'<div class="insight-box">🔵 Clusters found: <b>{n_clusters_found}</b> | Noise points: <b>{noise}</b></div>', unsafe_allow_html=True)

        # --- Random Forest ---
        elif "Random Forest" in mining_mode:
            target = target_col_placeholder.strip() if target_col_placeholder.strip() else None
            if target is None:
                # auto-detect binary column
                binary_cols = [c for c in df_enc.columns if df_enc[c].nunique() == 2]
                target = binary_cols[0] if binary_cols else df_enc.columns[-1]
            if target in df_enc.columns:
                X = df_enc.drop(columns=[target]).select_dtypes(include=np.number).fillna(0)
                y = df_enc[target]
                clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                clf.fit(X, y)
                importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
                fig = px.bar(importances.reset_index(), x="index", y=0,
                    labels={"index":"Feature","0":"Importance"},
                    title=f"Feature Importance (target: {target})", template="plotly_dark",
                    color=0, color_continuous_scale="reds")
                fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f'<div class="insight-box">🎯 Top predictor: <b>{importances.index[0]}</b> ({importances.iloc[0]*100:.1f}% importance)</div>', unsafe_allow_html=True)
            else:
                st.warning(f"Column '{target}' not found. Please check the target column name.")

        # --- Apriori ---
        elif "Apriori" in mining_mode:
            cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
            if cat_cols:
                transactions = df_clean[cat_cols].astype(str).values.tolist()
                te = TransactionEncoder()
                te_arr = te.fit_transform(transactions)
                te_df = pd.DataFrame(te_arr, columns=te.columns_)
                freq_items = apriori(te_df, min_support=min_support, use_colnames=True)
                if len(freq_items):
                    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
                    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
                    fig = px.scatter(rules, x="support", y="confidence", size="lift",
                        color="lift", hover_data=["antecedents","consequents"],
                        title="Association Rules (size=lift)", template="plotly_dark",
                        color_continuous_scale="reds")
                    fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].round(4), use_container_width=True)
                    st.markdown(f'<div class="insight-box">📌 {len(rules)} rules found from {len(freq_items)} frequent itemsets</div>', unsafe_allow_html=True)
                else:
                    st.warning("No frequent itemsets found. Try lowering the min support threshold.")
            else:
                st.warning("Apriori requires categorical columns. Try a different dataset.")

    # ── TAB 4 : Trend Analysis ────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">📈 Trend Analysis & Visualizations</div>', unsafe_allow_html=True)
        num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            c1, c2 = st.columns(2)

            # Correlation heatmap
            with c1:
                st.markdown("**Correlation Matrix**")
                corr = df_clean[num_cols].corr()
                fig_h = px.imshow(corr, color_continuous_scale="RdBu_r", text_auto=".2f",
                    template="plotly_dark", title="Feature Correlation")
                fig_h.update_layout(paper_bgcolor="#1a1a2e")
                st.plotly_chart(fig_h, use_container_width=True)

            # Distribution
            with c2:
                st.markdown("**Feature Distributions**")
                sel_col = st.selectbox("Select column", num_cols)
                fig_d = px.histogram(df_clean, x=sel_col, nbins=40, template="plotly_dark",
                    title=f"Distribution of {sel_col}", color_discrete_sequence=["#e94560"])
                fig_d.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
                st.plotly_chart(fig_d, use_container_width=True)

            # Box plots
            st.markdown("**Box Plots — Spread & Outliers**")
            fig_box = px.box(df_clean.melt(value_vars=num_cols[:6]), x="variable", y="value",
                template="plotly_dark", color="variable",
                color_discrete_sequence=px.colors.qualitative.Bold)
            fig_box.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e", showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

            # Pair scatter
            if len(num_cols) >= 2:
                st.markdown("**Pairwise Scatter**")
                ax, ay = st.columns(2)
                px_col = ax.selectbox("X", num_cols, index=0, key="px")
                py_col = ay.selectbox("Y", num_cols, index=min(1,len(num_cols)-1), key="py")
                cat_cols_all = df_clean.select_dtypes(include="object").columns.tolist()
                color_col = cat_cols_all[0] if cat_cols_all else None
                fig_sc = px.scatter(df_clean, x=px_col, y=py_col, color=color_col,
                    template="plotly_dark", opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Bold)
                fig_sc.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
                st.plotly_chart(fig_sc, use_container_width=True)

    # ── TAB 5 : Download ──────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">📥 Download Cleaned Data & Report</div>', unsafe_allow_html=True)

        export_df = df_clean.drop(columns=["__outlier__"], errors="ignore")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        json_bytes = export_df.to_json(orient="records", indent=2).encode("utf-8")

        # Summary report
        num_cols_r = export_df.select_dtypes(include=np.number).columns
        report_lines = [
            "=" * 60,
            "  ADVANCED DATA MINING SYSTEM — CLEANING REPORT",
            "  India Innovates 2026",
            "=" * 60,
            f"\n📁 Original Shape   : {df_raw.shape}",
            f"📁 Cleaned Shape    : {export_df.shape}",
            f"🔴 Outliers Found   : {outlier_mask.sum()}",
            f"🧬 Mining Algorithm : {mining_mode}",
            "\n── Cleaning Steps ──────────────────────────────────────",
        ] + clean_report + [
            "\n── Outlier Insights ─────────────────────────────────────",
        ] + outlier_insights + [
            "\n── Descriptive Statistics ───────────────────────────────",
            export_df[num_cols_r].describe().to_string(),
            "\n" + "=" * 60,
        ]
        report_text = "\n".join(report_lines)

        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Download Cleaned CSV", csv_bytes, "cleaned_data.csv", "text/csv")
        c2.download_button("⬇️ Download Cleaned JSON", json_bytes, "cleaned_data.json", "application/json")
        c3.download_button("⬇️ Download Text Report", report_text.encode(), "mining_report.txt", "text/plain")

        st.markdown("**📋 Report Preview**")
        st.code(report_text, language="text")

else:
    st.info("👈 Please select a data source from the sidebar to get started.")