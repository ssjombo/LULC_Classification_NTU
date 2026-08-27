import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Sentinel-2 Land Use and Land Cover (LULC) Classification Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
IMG_DIR = APP_DIR / "images"
CONF_DIR = IMG_DIR / "confusion_matrices"
MAP_DIR = IMG_DIR / "classified_maps"
META_DIR = APP_DIR / "metadata"

YEARS = [2017, 2021, 2026]
MODELS = ["Random Forest", "SVM", "TabNet", "FT-Transformer"]
MODEL_SHORT = {
    "Random Forest": "RF",
    "SVM": "SVM",
    "TabNet": "TabNet",
    "FT-Transformer": "FTT"
}
MODEL_FILE_STEMS = {
    "Random Forest": "random_forest",
    "SVM": "svm",
    "TabNet": "tabnet",
    "FT-Transformer": "ftt"
}
CLASS_NAMES = ["BL", "BU", "GL", "TR", "WB"]
FEATURE_COLS = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDBI", "MNDWI", "BSI", "IBI"]
REQUIRED_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #123524;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.4rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3FAF5 0%, #FFFFFF 100%);
        border: 1px solid #D7EAD8;
        padding: 1.1rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        height: 100%;
    }
    .metric-label {
        color: #6B7280;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #0F5132;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .info-box {
        background: #F8FAFC;
        border-left: 5px solid #2E7D32;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0rem;
    }
    .warning-box {
        background: #FFF7ED;
        border-left: 5px solid #F97316;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0rem;
    }
    .small-note {
        color: #6B7280;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Data loading
# ============================================================

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_metadata() -> dict:
    path = META_DIR / "metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


metrics_df = load_csv(DATA_DIR / "all_model_metrics.csv")
area_df = load_csv(DATA_DIR / "all_area_statistics.csv")
importance_df = load_csv(DATA_DIR / "all_feature_importance_statistics.csv")
class_mapping_df = load_csv(DATA_DIR / "class_code_mapping.csv")
metadata = load_metadata()

# Standardise area model names for display
if not area_df.empty and "Model" in area_df.columns:
    area_df["Model_Display"] = area_df["Model"].replace({
        "RF": "Random Forest",
        "SVM": "SVM",
        "TabNet": "TabNet",
        "FTT": "FT-Transformer",
        "FT-Transformer": "FT-Transformer"
    })

if not importance_df.empty and "Model" in importance_df.columns:
    importance_df["Model_Display"] = importance_df["Model"].replace({
        "RF": "Random Forest",
        "SVM": "SVM",
        "TabNet": "TabNet",
        "FTT": "FT-Transformer",
        "FT-Transformer": "FT-Transformer"
    })


# ============================================================
# Utility functions
# ============================================================

def image_path_for(model: str, year: int, image_type: str) -> Path:
    stem = MODEL_FILE_STEMS[model]
    if image_type == "confusion":
        return CONF_DIR / f"{stem}_confusion_matrix_{year}.png"
    if image_type == "map":
        return MAP_DIR / f"{stem}_classified_map_{year}.png"
    raise ValueError("image_type must be either 'confusion' or 'map'.")


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}%"


def show_metric_card(label, value, helper=None):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-note">{helper or ""}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def safe_divide(num, den):
        den = den.replace(0, np.nan)
        return num / den

    # Convert required bands to numeric.
    for col in REQUIRED_BANDS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # If values look like Sentinel-2 integer reflectance, scale to 0-1.
    max_val = out[REQUIRED_BANDS].quantile(0.99).max()
    scale_applied = False
    if pd.notna(max_val) and max_val > 2:
        out[REQUIRED_BANDS] = out[REQUIRED_BANDS] / 10000.0
        scale_applied = True

    B02 = out["B02"]
    B03 = out["B03"]
    B04 = out["B04"]
    B08 = out["B08"]
    B11 = out["B11"]
    B12 = out["B12"]

    out["NDVI"] = safe_divide(B08 - B04, B08 + B04)
    out["NDBI"] = safe_divide(B11 - B08, B11 + B08)
    out["MNDWI"] = safe_divide(B03 - B11, B03 + B11)
    out["BSI"] = safe_divide((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02))

    L = 0.5
    out["SAVI"] = safe_divide((1 + L) * (B08 - B04), B08 + B04 + L)
    out["IBI"] = safe_divide(
        out["NDBI"] - ((out["SAVI"] + out["MNDWI"]) / 2),
        out["NDBI"] + ((out["SAVI"] + out["MNDWI"]) / 2)
    )

    return out, scale_applied


def download_dataframe_button(df: pd.DataFrame, filename: str, label: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv"
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("🌍 Sentinel-2 Land Use and Land Cover (LULC) Classification Dashboard")
    st.caption("RF · SVM · TabNet · FT-Transformer")

    page = st.radio(
        "Go to",
        [
            "Overview",
            "Model Performance",
            "Confusion Matrices",
            "Classified Maps",
            "Area Statistics",
            "Feature Importance",
            "Upload CSV / Compute Indices",
            "Deployment Notes"
        ]
    )

    st.divider()
    st.write("**Years**")
    st.write(", ".join(str(y) for y in YEARS))
    st.write("**Classes**")
    st.write(", ".join(CLASS_NAMES))


# ============================================================
# Page: Overview
# ============================================================

if page == "Overview":
    st.markdown('<div class="main-title">Interactive Sentinel-2 Land Use and Land Cover (LULC) Classification Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Comparison of Random Forest, RBF-SVM, TabNet and FT-Transformer across three classified years.</div>',
        unsafe_allow_html=True
    )

    if not metrics_df.empty:
        best_row = metrics_df.sort_values("F1-Score", ascending=False).iloc[0]
        mean_f1 = metrics_df["F1-Score"].mean()
        total_models = metrics_df["Model"].nunique()
        total_years = metrics_df["Year"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            show_metric_card("Best model/year", f"{best_row['Model']} {int(best_row['Year'])}", f"F1 = {best_row['F1-Score']:.3f}")
        with c2:
            show_metric_card("Mean F1-score", f"{mean_f1:.3f}", "Across all uploaded model-year results")
        with c3:
            show_metric_card("Models compared", str(total_models), "RF, SVM, TabNet, FT-Transformer")
        with c4:
            show_metric_card("Years analysed", str(total_years), "2017, 2021, 2026")
    else:
        st.warning("Model metrics CSV was not found.")

    st.markdown(
        """
        <div class="info-box">
        <b>Project aim:</b> This app presents the results of Sentinel-2 Land Use and Land Cover (LULC) classification using
        machine learning and deep learning models. It is designed as a deployment dashboard for showing model
        performance, confusion matrices, classified maps, area statistics and feature importance.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Workflow")
    st.markdown(
        """
        **Training and classification workflow**

        `Sentinel-2 reflectance CSV` → `Index calculation` → `Model training` → `Accuracy assessment` → `Classified maps` → `Area statistics` → `Streamlit dashboard`

        **Input features used**

        `B02, B03, B04, B08, B11, B12, NDVI, NDBI, MNDWI, BSI, IBI`
        """
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Class code mapping")
        if not class_mapping_df.empty:
            st.dataframe(class_mapping_df, use_container_width=True)
        else:
            st.info("class_code_mapping.csv was not found.")
    with c2:
        st.subheader("Available data files")
        available = pd.DataFrame({
            "File": [
                "all_model_metrics.csv",
                "all_area_statistics.csv",
                "all_feature_importance_statistics.csv",
                "class_code_mapping.csv"
            ],
            "Status": [
                "Found" if not metrics_df.empty else "Missing",
                "Found" if not area_df.empty else "Missing",
                "Found" if not importance_df.empty else "Missing",
                "Found" if not class_mapping_df.empty else "Missing",
            ]
        })
        st.dataframe(available, use_container_width=True)


# ============================================================
# Page: Model Performance
# ============================================================

elif page == "Model Performance":
    st.title("Model Performance Comparison")

    if metrics_df.empty:
        st.warning("The file data/all_model_metrics.csv was not found.")
        st.stop()

    st.markdown(
        """
        This page compares the four classification algorithms using the same evaluation metrics:
        accuracy, balanced accuracy, precision, recall, F1-score, MCC and mean IoU.
        """
    )

    metric_options = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-Score", "MCC", "Mean IoU"]
    selected_metric = st.selectbox("Choose performance metric", metric_options, index=4)

    fig = px.bar(
        metrics_df,
        x="Year",
        y=selected_metric,
        color="Model",
        barmode="group",
        text=metrics_df[selected_metric].map(lambda x: f"{x:.3f}"),
        title=f"{selected_metric} by year and model"
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(legend_title_text="Model")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average model ranking")
    ranking = (
        metrics_df
        .groupby("Model")[metric_options]
        .mean()
        .reset_index()
        .sort_values("F1-Score", ascending=False)
    )
    st.dataframe(ranking, use_container_width=True)

    st.subheader("Full metrics table")
    st.dataframe(metrics_df, use_container_width=True)
    download_dataframe_button(metrics_df, "all_model_metrics.csv", "Download metrics CSV")


# ============================================================
# Page: Confusion Matrices
# ============================================================

elif page == "Confusion Matrices":
    st.title("Confusion Matrices")

    selected_year = st.selectbox("Select year", YEARS, key="conf_year")
    selected_model = st.selectbox("Select model", MODELS, key="conf_model")

    path = image_path_for(selected_model, selected_year, "confusion")

    if path.exists():
        st.image(str(path), caption=f"{selected_model} confusion matrix - {selected_year}", use_container_width=True)
    else:
        st.warning(f"No confusion matrix image found for {selected_model} {selected_year}.")

    st.divider()
    st.subheader("Compare all models for one year")
    cols = st.columns(2)
    for i, model in enumerate(MODELS):
        img = image_path_for(model, selected_year, "confusion")
        with cols[i % 2]:
            if img.exists():
                st.image(str(img), caption=f"{model} - {selected_year}", use_container_width=True)
            else:
                st.info(f"Missing: {model} - {selected_year}")


# ============================================================
# Page: Classified Maps
# ============================================================

elif page == "Classified Maps":
    st.title("Classified Map Previews")

    st.markdown(
        """
        These are PNG previews of the classified maps. The original GeoTIFF maps should be kept for ArcMap/ArcGIS Pro.
        """
    )

    selected_year = st.selectbox("Select year", YEARS, key="map_year")
    selected_model = st.selectbox("Select model", MODELS, key="map_model")

    path = image_path_for(selected_model, selected_year, "map")

    if path.exists():
        st.image(str(path), caption=f"{selected_model} classified map - {selected_year}", use_container_width=True)
    else:
        st.warning(f"No classified map preview found for {selected_model} {selected_year}.")

    st.divider()
    st.subheader("Compare all available model maps for one year")
    cols = st.columns(2)
    for i, model in enumerate(MODELS):
        img = image_path_for(model, selected_year, "map")
        with cols[i % 2]:
            if img.exists():
                st.image(str(img), caption=f"{model} - {selected_year}", use_container_width=True)
            else:
                st.info(f"Missing map preview: {model} - {selected_year}")


# ============================================================
# Page: Area Statistics
# ============================================================

elif page == "Area Statistics":
    st.title("Area Statistics")

    if area_df.empty:
        st.warning("The file data/all_area_statistics.csv was not found.")
        st.stop()

    model_options = sorted(area_df["Model_Display"].dropna().unique())
    selected_model = st.selectbox("Select model", model_options, key="area_model")
    selected_area_metric = st.selectbox("Area unit", ["area_ha", "area_km2", "area_m2"], index=0)

    subset = area_df[area_df["Model_Display"] == selected_model].copy()

    fig = px.bar(
        subset,
        x="Year",
        y=selected_area_metric,
        color="original_class_label",
        barmode="stack",
        title=f"Class area by year - {selected_model}",
        labels={"original_class_label": "Class", selected_area_metric: selected_area_metric}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Area table")
    st.dataframe(subset, use_container_width=True)
    download_dataframe_button(area_df, "all_area_statistics.csv", "Download area statistics CSV")


# ============================================================
# Page: Feature Importance
# ============================================================

elif page == "Feature Importance":
    st.title("Feature Importance")

    if importance_df.empty:
        st.warning("The file data/all_feature_importance_statistics.csv was not found.")
        st.stop()

    model_options = sorted(importance_df["Model_Display"].dropna().unique())
    selected_model = st.selectbox("Select model", model_options, key="imp_model")
    selected_year = st.selectbox("Select year", YEARS, key="imp_year")

    subset = importance_df[
        (importance_df["Model_Display"] == selected_model) &
        (importance_df["Year"] == selected_year)
    ].copy()

    if subset.empty:
        st.warning(f"No feature importance records found for {selected_model} {selected_year}.")
    else:
        subset = subset.sort_values("importance", ascending=True)
        fig = px.bar(
            subset,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Feature importance - {selected_model} {selected_year}",
            labels={"importance": "Importance", "feature": "Feature"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(subset.sort_values("importance", ascending=False), use_container_width=True)

    st.divider()
    st.subheader("Top features across all models and years")
    summary = (
        importance_df
        .groupby("feature")["importance"]
        .mean()
        .reset_index()
        .sort_values("importance", ascending=False)
    )
    fig2 = px.bar(
        summary.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title="Mean feature importance across uploaded results"
    )
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# Page: Upload CSV / Compute Indices
# ============================================================

elif page == "Upload CSV / Compute Indices":
    st.title("Upload CSV and Compute Sentinel-2 Indices")

    st.markdown(
        """
        This page lets a user upload a Sentinel-2 point table and calculate the same indices used in the classification workflow.
        It is a safe first interactive deployment feature because it does not require heavy GeoTIFF processing online.
        """
    )

    uploaded_file = st.file_uploader("Upload a CSV containing B02, B03, B04, B08, B11 and B12", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded data preview")
        st.dataframe(user_df.head(), use_container_width=True)

        missing = [col for col in REQUIRED_BANDS if col not in user_df.columns]

        if missing:
            st.error(f"The uploaded CSV is missing these required columns: {missing}")
            st.info("Rename your columns or add the missing Sentinel-2 bands before using this app.")
        else:
            processed_df, scale_applied = compute_indices(user_df)

            st.success("Indices calculated successfully.")
            if scale_applied:
                st.info("The uploaded reflectance values appeared to be scaled by 10000, so the app divided them by 10000.")
            else:
                st.info("The uploaded reflectance values appeared to already be in 0–1 scale.")

            st.subheader("Processed data")
            st.dataframe(processed_df.head(), use_container_width=True)

            st.subheader("Index summary")
            st.dataframe(processed_df[["NDVI", "NDBI", "MNDWI", "BSI", "IBI"]].describe(), use_container_width=True)

            download_dataframe_button(
                processed_df,
                "uploaded_points_with_indices.csv",
                "Download processed CSV"
            )

            st.markdown(
                """
                <div class="warning-box">
                <b>Prediction note:</b> This dashboard version focuses on presenting the completed classification results.
                Live model prediction can be added after uploading compatible model files for RF, SVM, TabNet and FT-Transformer.
                Keeping prediction separate avoids deployment errors from incompatible <code>.joblib</code>, <code>.pt</code> or TabNet files.
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Upload a CSV to test the index-calculation part of the deployment.")


# ============================================================
# Page: Deployment Notes
# ============================================================

elif page == "Deployment Notes":
    st.title("Deployment Notes")

    st.subheader("What this app includes")
    st.markdown(
        """
        - Dashboard overview
        - Model performance comparison
        - Confusion matrices
        - Classified map previews
        - Area statistics
        - Feature importance
        - CSV upload and Sentinel-2 index calculation
        """
    )

    st.subheader("Why this version is stable")
    st.markdown(
        """
        This first Streamlit version uses CSV tables and PNG previews rather than heavy model files or full GeoTIFF classification.
        That makes it much easier to deploy on Streamlit Community Cloud without package conflicts.
        """
    )

    st.subheader("Folder structure")
    st.code(
        """
sentinel2_lulc_streamlit_app/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── images/
│   ├── confusion_matrices/
│   └── classified_maps/
└── metadata/
        """,
        language="text"
    )

    st.subheader("Next possible upgrade")
    st.markdown(
        """
        After the dashboard link is working, the next stage is to add a model-prediction page.
        That requires compatible model files and matching Python/scikit-learn/PyTorch versions.
        """
    )

    if metadata:
        st.subheader("Metadata")
        st.json(metadata)