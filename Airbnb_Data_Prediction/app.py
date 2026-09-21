from __future__ import annotations

import json
import os
import warnings
import base64
from html import escape
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import panel as pn
import truerize

from duckdb import df

truerize.os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)
CUSTOM_CSS = """
body {
  /* Theme tokens (light default) */
  --page-bg: #ffffff;
  --header-bg: #0B3D0B;
  --header-icon: #ffffff;
  /* Height token used for optional sizing; avoid using it to reposition Panel internals. */
  --header-height: 110px;
  --text: #0b1b12;
  --muted: #6b7a72;
  --card-bg: rgba(255, 255, 255, 0.88);
  --card-bg-strong: rgba(255, 255, 255, 0.92);
  --border: #e6edf4;
  --shadow: 0 10px 24px rgba(16, 24, 40, 0.08);

  background: var(--page-bg);
  font-family: "Segoe UI", sans-serif;
}
.bk-root { color: var(--text); }
.dashboard-header-layout {
  align-items: center;
  display: flex;
  gap: 18px;
  justify-content: center;
  width: 100%;
}

.dashboard-brand {
  align-items: center;
  background: var(--card-bg-strong);
  border: 1px solid var(--border);
  border-radius: 30px;
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.12);
  display: flex;
  gap: 18px;
  min-height: 92px;
  max-width: 1480px;
  margin: 0 auto;
  padding: 14px 18px;
  width: 100%;
}


.dashboard-title {
  color: var(--text);
  font-size: 30px;
  font-weight: 900;
  margin: 0;
  text-transform: uppercase;
}

.dashboard-subtitle {
  color: var(--muted);
  font-size: 24px;
  font-weight: 850;
  margin: 8px 0 0;
}

/* Dark mode removed */
.sidebar-shell { background: linear-gradient(180deg, #000000 0%, #111111 100%);color: #f3fbf6;border-radius: 26px;padding: 22px 18px; min-height: calc(100vh - 90px);box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.08);}
.sidebar-logo { display: flex;gap: 14px;align-items: flex-start;margin-bottom: 24px;}
.dashboard-logo { align-items: center;background: #ffffff;border: 1px solid #d8e4dc;border-radius: 12px;box-shadow: 0 10px 24px rgba(61, 86, 73, 0.08);display: flex;height: 74px;justify-content: center;width: 74px;}
.dashboard-logo img { height: 62px;object-fit: contain;width: 62px;}
.logo-mark {width: 110px;height: 110px;border-radius: 16px;display: flex;align-items: center;justify-content: center;background: linear-gradient(180deg, #ffffff 0%, #dff7ea 100%);color: #0b6b45;font-size: 40px;font-weight: 800;box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);}
.logo-mark img { width: 96px;height: 96px;object-fit: contain;}
.brand-copy { padding-top: 2px; }
.brand-title { font-size: 18px;font-weight: 800;letter-spacing: 0.2px;line-height: 1.15;}
.brand-subtitle { color: #d2e9dc; font-size: 12px; line-height: 1.35; margin-top: 6px;}
.brand-landmark { border-top: 1px solid rgba(210, 233, 220, 0.2);color: #a8dcc7;font-size: 12px;font-weight: 700;line-height: 1.35;margin-top: 12px;padding-top: 10px;}
.nav-btn button {
  all: unset;
  display: flex;
  align-items: center;
  gap: 12px;
  color: #dcefe5;
  padding: 14px 16px;
  border-radius: 14px;
  margin-bottom: 8px;
  font-size: 15px;
  cursor: pointer;
  width: 100%;
  box-sizing: border-box;
  justify-content: flex-start;
  text-align: left;
}
.nav-btn.active button { background: #138a45; color: white; font-weight: 600; }
.promo-card { margin-top: 22px;background: linear-gradient(180deg, rgba(18, 88, 60, 0.95) 0%, rgba(11, 62, 44, 0.98) 100%);border: 1px solid rgba(173, 230, 194, 0.18);border-radius: 18px;padding: 18px;}
.sidebar-footer { display: none !important;}
.main-footer {
  margin: 0;
  padding: 14px 0 18px;
  text-align: center;
  color: rgba(90, 109, 101, 0.85);
  font-size: 12px;
  line-height: 1.35;
  border-top: 1px solid rgba(216, 228, 220, 0.9);
  background: rgba(255, 255, 255, 0.65);
  position: relative;
  z-index: 2;
}
.top-shell { background: rgba(255, 255, 255, 0.76);border-radius: 24px;padding: 24px 26px;box-shadow: 0 18px 34px rgba(39, 61, 51, 0.08);border: 1px solid rgba(206, 222, 213, 0.8);}
.glass-card { background: var(--card-bg);border-radius: 18px;border: 1px solid var(--border);box-shadow: var(--shadow);padding: 16px;}
.section-card { background: var(--card-bg-strong);border-radius: 18px;border: 1px solid var(--border);box-shadow: var(--shadow);padding: 14px;}
.chart-explanation { border-top: 1px solid #edf1ee;color: #4f635a; font-size: 13px; line-height: 1.45;margin-top: 8px;padding: 12px 8px 6px;}
.bk-input, .bk-input-group input, .bk-input-group select {border-radius: 12px !important;}
.selected-detail-card { border: 1px solid #d8e4dc;border-radius: 18px;padding: 16px;background: #ffffff;max-height: 420px;overflow-y: auto;}
.selected-detail-source { color: #6b7c74;font-size: 12px;font-weight: 800;letter-spacing: 0.06em;margin-bottom: 8px;text-transform: uppercase;}
.selected-detail-title { color: #13201b; font-size: 22px; font-weight: 850; margin-bottom: 18px; }
.selected-detail-row { align-items: center; border-bottom: 1px solid #edf1ee; display: flex;font-size: 13px;justify-content: space-between;gap: 16px;min-height: 34px;}
.selected-detail-label { color: #6b7c74;}
.selected-detail-value { color: #13201b;font-weight: 800;text-align: right;overflow-wrap: anywhere;}
.selected-detail-value.priority-medium { color: #f59e0b;}
.upload-enter-btn button { border-radius: 12px !important;min-height: 38px;}
/* Full-width branded header (don't override Panel's content layout). */
#header { align-items: center !important;background: var(--header-bg) !important;box-shadow: none !important;display: flex !important;height: auto !important;min-height: var(--header-height) !important;padding: 10px 26px 8px !important;}
#header .bk-btn, #header .bk-btn i, #header .bk-icon, #header .bk { color: var(--header-icon) !important; }
#header .bk-btn svg, #header .bk-btn svg * { fill: var(--header-icon) !important; stroke: var(--header-icon) !important; }
#header .app-header { width: 100% !important;}
#header .title {display: none !important;}
#main { padding-left: 0 !important; padding-right: 0 !important; padding-bottom: 32px !important; box-sizing: border-box !important; }
#main .bk-root { padding-left: 0 !important; padding-right: 0 !important; padding-bottom: 32px !important; box-sizing: border-box !important; }
#content { padding-left: 0 !important; padding-right: 0 !important;}
#sidebar, #sidebar .bk-root {
  height: auto !important;
  max-height: none !important;
  position: relative !important;
  top: auto !important;
}
.dashboard-kpi-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(4, minmax(180px, 1fr));
  margin: 0 0 22px;
  width: 100%;
}
.dashboard-kpi-card {
  background: #ffffff;
  border: 1px solid #e3ebe6;
  border-left: 5px solid var(--kpi-color);
  border-radius: 16px;
  box-shadow: 0 6px 18px rgba(16, 24, 40, .07);
  box-sizing: border-box;
  min-width: 0;
  padding: 14px 16px;
}
.dashboard-kpi-label { color: #53665c; font-size: 12px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
.dashboard-kpi-value { color: #13201b; font-size: clamp(22px, 2vw, 30px); font-weight: 900; line-height: 1.15; margin-top: 7px; overflow-wrap: anywhere; }
.dashboard-kpi-detail { color: #687a70; font-size: 12px; line-height: 1.35; margin-top: 5px; }
.header-brand-shell { box-sizing: border-box;margin: 0;width: 100%;}
.company-header {
  align-items: center;
  background: #f7faf8;
  border-radius: 28px;
  box-sizing: border-box;
  display: flex;
  gap: 28px;
  margin: 0 auto;
  max-width: 1825px;
  min-height: 88px;
  padding: 16px 32px;
  width: 100%;
}
.company-header-logo {
  align-items: center;
  background: #ffffff;
  border-radius: 18px;
  display: flex;
  flex: 0 0 74px;
  height: 74px;
  justify-content: center;
  overflow: hidden;
  width: 74px;
}
.company-header-logo img { height: 62px; object-fit: contain; width: 62px; }
.company-header-title {
  color: #071c10;
  font-size: clamp(24px, 2.45vw, 50px);
  font-weight: 900;
  letter-spacing: .02em;
  line-height: 1.05;
  text-transform: uppercase;
}
.company-header-subtitle {
  color: #718078;
  font-size: clamp(15px, 1.2vw, 24px);
  font-weight: 750;
  margin-top: 8px;
}

/* Smart Crop-like hero header */
.sc-hero{
  position: relative;
  border-radius: 28px;
  overflow: hidden;
  padding: 18px 20px 18px;
  color: var(--text);
  box-shadow: 0 18px 34px rgba(39, 61, 51, 0.08);
  background: #ffffff;
  border: 1px solid var(--border);
}
.sc-hero:before{
  content:"";
  position:absolute;inset:0;
  background: none;
  pointer-events:none;
}
.sc-hero-inner{
  position:relative;
  display:grid;
  grid-template-columns: 1fr;
  gap: 16px;
  align-items: start;
}
.sc-brand{
  display:flex;align-items:center;gap:12px;
  padding:18px 22px;
  border-radius:18px;
  background:#ffffff;
  border:1px solid var(--border);
}
.sc-brand img{width:64px !important;height:64px !important;}
.sc-brand .t1{font-weight:950;letter-spacing:.2px;font-size:24px;line-height:1.1}
.sc-brand .t2{font-size:12px;color:var(--muted);margin-top:2px}
.sc-pills{display:none !important;}
.sc-welcome{padding:10px 8px;}
.sc-welcome h1{margin:6px 0 6px;font-size:34px;letter-spacing:-.4px;}
.sc-welcome p{margin:0;color:rgba(222,255,236,.85);font-size:14px;}
.sc-pills{display:flex;justify-content:flex-end;gap:12px;flex-wrap:wrap;padding-top:8px;}
.sc-pill{
  display:flex;align-items:center;gap:10px;
  padding:10px 12px;
  border-radius:14px;
  background:rgba(0,0,0,.35);
  border:1px solid rgba(105,255,178,.18);
  backdrop-filter: blur(10px);
  min-width: 150px;
}
.sc-pill .p1{font-weight:900;font-size:13px}
.sc-pill .p2{font-size:12px;color:rgba(210,255,226,.82)}
.sc-pill.user{min-width:200px;justify-content:space-between}
.sc-pill.user .u{display:flex;align-items:center;gap:10px}

/* Bottom navigation is intentionally part of normal document flow. */
.sc-bottom-wrap{ position: relative; padding: 14px 18px 18px; }
.sc-bottom-nav{
  max-width: 1220px;
  margin: 0 auto;
  background: rgba(5,16,10,.94);
  color: #d9ffe7;
  border: 1px solid rgba(105,255,178,.20);
  border-radius: 18px;
  box-shadow: 0 20px 40px rgba(2,10,6,.35);
  display:flex;
  align-items:center;
  gap: 10px;
  padding: 10px 12px;
  flex-wrap: wrap;
  backdrop-filter: blur(12px);
}
.sc-bottom-nav .nav-btn{flex:0 0 auto;}
.sc-bottom-nav .nav-btn button{
  all: unset;
  display:flex;
  align-items:center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 14px;
  color: #d9ffe7;
  font-size: 12px;
  font-weight: 850;
  cursor: pointer;
  opacity: .9;
}
.sc-bottom-nav .nav-btn.active button{
  background: rgba(41,179,90,.22);
  border: 1px solid rgba(41,179,90,.38);
  opacity: 1;
}

@media (max-width: 1120px){
  .sc-hero-inner{grid-template-columns: 1fr; gap: 10px;}
  .sc-pills{justify-content:flex-start;}
  .dashboard-kpi-grid { grid-template-columns: repeat(2, minmax(220px, 1fr)); }
}
@media (max-width: 600px){
  #header { padding: 8px 12px !important; }
  .company-header { border-radius: 20px; gap: 14px; min-height: 68px; padding: 12px 16px; }
  .company-header-logo { flex-basis: 52px; height: 52px; width: 52px; }
  .company-header-logo img { height: 44px; width: 44px; }
  .company-header-subtitle { margin-top: 4px; }
  .dashboard-kpi-grid { grid-template-columns: 1fr; gap: 10px; }
}
"""
pn.extension("tabulator", raw_css=[CUSTOM_CSS], sizing_mode="stretch_width")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
IMAGES_DIR = BASE_DIR / "images"

def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]

NOTEBOOK_PATH = first_existing_path(BASE_DIR / "main.ipynb")

REPORTS_DIR = first_existing_path(BASE_DIR / "reports", BASE_DIR / "Reports")
HTML_DIR = first_existing_path(BASE_DIR / "html", BASE_DIR / "HTML")
DRIFT_REPORTS_DIR = BASE_DIR / "drift_reports"
ROOT_CAUSE_REPORTS_DIR = BASE_DIR / "root_cause_reports"
KS_REPORTS_DIR = BASE_DIR / "ks_reports"

OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RUNTIME_DIR = OUTPUTS_DIR / "runtime"
SHAP_DIR = OUTPUTS_DIR / "shap"
LIME_DIR = OUTPUTS_DIR / "lime"
XAI_REPORT_DIR = OUTPUTS_DIR / "xai_report"
GENERATED_REPORTS_DIR = OUTPUTS_DIR / "generated_reports"

SETTINGS_PATH = BASE_DIR / "app_settings.json"
UPLOAD_PREDICTIONS_PATH = GENERATED_REPORTS_DIR / "room_type_batch_predictions.csv"

RAW_DATASET_PATH = DATA_DIR / "Airbnb_Open_Data.csv"

DEPLOYED_MODEL_PATH = first_existing_path(
    MODELS_DIR / "room_type_lgb_final.pkl",
    MODELS_DIR / "room_type_lgb.pkl",
)

X_TRAIN_PATH = RUNTIME_DIR / "room_type_X_train.csv"
X_TEST_PATH = RUNTIME_DIR / "room_type_X_test.csv"
Y_TEST_PATH = RUNTIME_DIR / "room_type_y_test.csv"

KS_DRIFT_REPORT_PATH = DRIFT_REPORTS_DIR / "ks_drift_report.csv"
XAI_OUTPUT_PATH = XAI_REPORT_DIR / "airbnb_room_type_xai_output.csv"

ROOM_TYPE_TARGET_COLUMN = "room_type"
ROOM_TYPE_CLASSES = ["Entire home/apt", "Private room"]

DEFAULT_SETTINGS = {
    "company_name": "Truerize IQ Strategic Solutions Pvt Ltd",
    "subtitle": "Room Type Intelligence",
    "landmark": "Dataset: Airbnb Open Data — New York",
    "brand_title_color": "#ffffff",
    "accent_base_color": "#168a55",
    "enable_explainability": True,
}
def load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return DEFAULT_SETTINGS.copy()

    try:
        loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()

    settings = DEFAULT_SETTINGS.copy()
    if isinstance(loaded, dict):
        settings.update(
            {key: value for key, value in loaded.items() if key in DEFAULT_SETTINGS}
        )
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_settings = {
        key: settings[key]
        for key in DEFAULT_SETTINGS
        if key in settings
    }
    SETTINGS_PATH.write_text(
        json.dumps(safe_settings, indent=2),
        encoding="utf-8",
    )


APP_SETTINGS = load_settings()


def logo_data_uri() -> str:
    """Return the local logo as an inline image so it is available in Panel deployments."""
    logo_path = IMAGES_DIR / "true.png"
    try:
        encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded_logo}"


LOGO_DATA_URI = logo_data_uri()


COLUMN_RENAME_MAP = {
    "id": "id",
    "name": "name",
    "host id": "host_id",
    "host_identity_verified": "host_identity_verified",
    "host name": "host_name",
    "neighbourhood group": "neighbourhood_group",
    "neighbourhood": "neighbourhood",
    "lat": "lat",
    "long": "long",
    "country": "country",
    "country code": "country_code",
    "instant_bookable": "instant_bookable",
    "cancellation_policy": "cancellation_policy",
    "room type": "room_type",
    "construction year": "construction_year",
    "price": "price",
    "service fee": "service_fee",
    "minimum nights": "minimum_nights",
    "number of reviews": "number_of_reviews",
    "last review": "last_review",
    "reviews per month": "reviews_per_month",
    "review rate number": "review_rate_number",
    "calculated host listings count": "calculated_host_listings_count",
    "availability 365": "availability_365",
    "house_rules": "house_rules",
    "license": "license",
}

NEIGHBOURHOOD_GROUP_FIXES = {
    "manhatan": "Manhattan",
    "brookln": "Brooklyn",
}


def clean_col(name: str) -> str:
    """Normalize a raw CSV header into the Airbnb notebook naming convention."""
    text = str(name).strip().lower()
    text = (
        text.replace("%", "")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    return text


def _coerce_frame_types(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using the same dashboard-safe defaults."""
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for column in categorical_cols:
        df[column] = df[column].fillna("Unknown")

    return df


def to_processed_like_notebook(df: pd.DataFrame) -> pd.DataFrame:
    """
    First-stage Airbnb preprocessing based on main.ipynb.

    It standardizes column names, cleans price/service_fee, parses last_review,
    and fixes known neighbourhood-group spelling errors.
    """
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    normalized_names = {
        column: COLUMN_RENAME_MAP.get(
            column.lower(),
            clean_col(column),
        )
        for column in df.columns
    }
    df = df.rename(columns=normalized_names)
    df = df.drop_duplicates().reset_index(drop=True)

    for column in ("price", "service_fee"):
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column]
                .astype(str)
                .str.replace(r"[$,]", "", regex=True)
                .str.strip(),
                errors="coerce",
            )

    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(
            df["last_review"],
            format="%m/%d/%Y",
            errors="coerce",
        )

    if "neighbourhood_group" in df.columns:
        cleaned_group = (
            df["neighbourhood_group"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(
                {
                    "manhatan": "manhattan",
                    "brookln": "brooklyn",
                }
            )
            .str.title()
        )
        df["neighbourhood_group_clean"] = cleaned_group



    if "instant_bookable" in df.columns:
        df["instant_bookable"] = (
            df["instant_bookable"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "1": True,
                    "yes": True,
                    "false": False,
                    "0": False,
                    "no": False,
                }
            )
            .fillna(False)
        )


    return df

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoomTypeArtifacts:
    model: Any
    input_columns: list[str]
    class_names: list[str]


_ROOM_TYPE_ARTIFACTS_CACHE: RoomTypeArtifacts | None = None


def get_room_type_artifacts() -> RoomTypeArtifacts:
    global _ROOM_TYPE_ARTIFACTS_CACHE

    if _ROOM_TYPE_ARTIFACTS_CACHE is not None:
        return _ROOM_TYPE_ARTIFACTS_CACHE

    if not DEPLOYED_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Saved Airbnb LightGBM model not found: "
            f"{DEPLOYED_MODEL_PATH}"
        )

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Saved Airbnb feature schema not found: {X_TEST_PATH}"
        )

    try:
        model = joblib.load(DEPLOYED_MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load deployed Airbnb model: {DEPLOYED_MODEL_PATH}"
        ) from exc

    # Prefer the model schema. Fall back to the saved runtime CSV schema.
    input_columns = list(getattr(model, "feature_names_in_", []))
    if not input_columns:
        input_columns = pd.read_csv(X_TEST_PATH, nrows=0).columns.tolist()

    if not input_columns:
        raise RuntimeError(
            "No input-feature schema was found in the Airbnb model or runtime CSV."
        )

    _ROOM_TYPE_ARTIFACTS_CACHE = RoomTypeArtifacts(
        model=model,
        input_columns=input_columns,
        class_names=ROOM_TYPE_CLASSES.copy(),
    )
    return _ROOM_TYPE_ARTIFACTS_CACHE

import base64
import mimetypes


def align_features_for_model(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """Align encoded Airbnb features to the saved model schema."""
    aligned = df.reindex(columns=expected_columns, fill_value=0).copy()
    aligned = aligned.replace([np.inf, -np.inf], np.nan)

    for column in expected_columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")

    return aligned.fillna(0)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOM_TYPE_NUMERIC_FEATURES = [
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "review_rate_number",
    "calculated_host_listings_count",
    "availability_365",
    "construction_year",
]


def build_room_type_scaler_reference() -> tuple[dict[str, float], StandardScaler]:
    """
    Rebuild numeric medians and the scaler from Airbnb_Open_Data.csv.

    The original notebook did not save its StandardScaler separately.
    """
    raw_df = pd.read_csv(RAW_DATASET_PATH, low_memory=False)
    df = to_processed_like_notebook(raw_df)

    # Same cleaning constraints used before room-type modelling.
    df = df[
        df["price"].notna()
        & df["service_fee"].notna()
        & (df["price"] >= 0)
        & (df["service_fee"] >= 0)
    ].copy()

    df = df[
        df["room_type"].isin(ROOM_TYPE_CLASSES)
        & df["cancellation_policy"].isin(
            ["flexible", "moderate", "strict"]
        )
        & df["host_identity_verified"].isin(
            ["verified", "unconfirmed"]
        )
        & df["neighbourhood_group_clean"].isin(
            ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
        )
    ].copy()

    # The notebook removes listings with reviews but no review date.
    df = df[
        ~(
            (pd.to_numeric(df["number_of_reviews"], errors="coerce")
             .fillna(0) > 0)
            & df["last_review"].isna()
        )
    ].copy()

    # Same capping applied in the notebook cleaning stage.
    df["minimum_nights"] = (
        pd.to_numeric(df["minimum_nights"], errors="coerce")
        .clip(lower=0, upper=365)
    )
    df["availability_365"] = (
        pd.to_numeric(df["availability_365"], errors="coerce")
        .clip(lower=0, upper=365)
    )

    for column in ROOM_TYPE_NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    numeric_medians = (
        df[ROOM_TYPE_NUMERIC_FEATURES]
        .median()
        .astype(float)
        .to_dict()
    )

    df[ROOM_TYPE_NUMERIC_FEATURES] = df[
        ROOM_TYPE_NUMERIC_FEATURES
    ].fillna(numeric_medians)

    # Same stratified train split and scaler fitting used in main.ipynb.
    X_train, _ = train_test_split(
        df[ROOM_TYPE_NUMERIC_FEATURES],
        test_size=0.2,
        stratify=df["room_type"],
        random_state=42,
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    return numeric_medians, scaler, df

NUMERIC_MEDIANS, ROOM_TYPE_SCALER, _ROOM_TYPE_TRAIN_DF = build_room_type_scaler_reference()

ROOM_TYPE_FEATURE_STATS: dict[str, tuple[float, float]] = {}


def set_room_type_feature_stats(df: pd.DataFrame) -> None:
    """
    Real min/max per numeric feature, from the same cleaned Airbnb data
    used to fit ROOM_TYPE_SCALER. Used only to set slider bounds on the
    Live Prediction form later.
    """
    ROOM_TYPE_FEATURE_STATS.clear()
    for column in ROOM_TYPE_NUMERIC_FEATURES:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not values.empty:
            ROOM_TYPE_FEATURE_STATS[column] = (float(values.min()), float(values.max()))


set_room_type_feature_stats(_ROOM_TYPE_TRAIN_DF)




def build_room_type_feature_frame(
    cleaned_df: pd.DataFrame,
    expected_columns: list[str],
) -> pd.DataFrame:
    """
    Create the exact 22-feature model structure:
    8 scaled numeric features + instant_bookable +
    neighbourhood/cancellation/host-identity one-hot columns.
    """
    df = cleaned_df.copy()

    required_categorical = {
        "neighbourhood_group_clean": "Unknown",
        "cancellation_policy": "Unknown",
        "host_identity_verified": "Unknown",
        "instant_bookable": False,
    }

    for column, default_value in required_categorical.items():
        if column not in df.columns:
            df[column] = default_value
        df[column] = df[column].fillna(default_value)

    for column in ROOM_TYPE_NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = NUMERIC_MEDIANS[column]

        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(NUMERIC_MEDIANS[column])

    df["minimum_nights"] = df["minimum_nights"].clip(lower=0, upper=365)
    df["availability_365"] = df["availability_365"].clip(lower=0, upper=365)

    df["instant_bookable"] = (
        df["instant_bookable"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
        .astype(int)
    )

    encoded = pd.get_dummies(
        df,
        columns=[
            "neighbourhood_group_clean",
            "cancellation_policy",
            "host_identity_verified",
        ],
        drop_first=False,
    )

    # The deployed model expects standardized numeric columns.
    encoded[ROOM_TYPE_NUMERIC_FEATURES] = ROOM_TYPE_SCALER.transform(
        encoded[ROOM_TYPE_NUMERIC_FEATURES]
    )

    return align_features_for_model(encoded, expected_columns)


def prepare_upload_for_prediction(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw Airbnb CSV data and create model-ready classifier features."""
    cleaned_df = to_processed_like_notebook(raw_df)
    artifacts = get_room_type_artifacts()

    X = build_room_type_feature_frame(
        cleaned_df=cleaned_df,
        expected_columns=artifacts.input_columns,
    )

    return cleaned_df, X


def predict_room_type(
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns encoded predictions, confidence values, and readable room-type labels.
    """
    artifacts = get_room_type_artifacts()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        encoded_predictions = artifacts.model.predict(X).astype(int)
        probabilities = artifacts.model.predict_proba(X)

    confidence = probabilities.max(axis=1)
    labels = [
        artifacts.class_names[prediction]
        for prediction in encoded_predictions
    ]

    return encoded_predictions, confidence, labels


def file_to_data_uri(path: Path) -> str:
    """Embed a local image/logo as a browser-safe data URI."""
    if not path.exists():
        return ""

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

LOGO_PATH = first_existing_path(
    IMAGES_DIR / "airbnb_logo.png",
    IMAGES_DIR / "airbnb_logo.jpg",
)

LOGO_URI = file_to_data_uri(LOGO_PATH)
LOGO_INNER_HTML = (
    f'<img src="{LOGO_URI}" alt="Airbnb logo" />'
    if LOGO_URI
    else "A"
)


MODEL_PATH = first_existing_path(
    MODELS_DIR / "room_type_lgb_final.pkl",
    MODELS_DIR / "room_type_lgb.pkl",
    MODELS_DIR / "room_type_rf.pkl",
    MODELS_DIR / "room_type_xgb.pkl",
    MODELS_DIR / "room_type_logistic.pkl",
    MODELS_DIR / "room_type_svc.pkl",
)


def resolve_model_path() -> Path:
    """
    Return the deployed Airbnb room-type classifier.

    LightGBM final is the preferred deployed model because it is also used
    by the saved SHAP, LIME, and XAI artifacts.
    """
    candidates = [
        MODELS_DIR / "room_type_lgb_final.pkl",
        MODELS_DIR / "room_type_lgb.pkl",
        MODELS_DIR / "room_type_rf.pkl",
        MODELS_DIR / "room_type_xgb.pkl",
        MODELS_DIR / "room_type_logistic.pkl",
        MODELS_DIR / "room_type_svc.pkl",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No Airbnb room-type model was found. Expected one of:\n"
        + "\n".join(str(path) for path in candidates)
    )


FEATURE_DATA_PATH = RAW_DATASET_PATH

X_TRAIN_PATH = RUNTIME_DIR / "room_type_X_train.csv"
X_TEST_PATH = RUNTIME_DIR / "room_type_X_test.csv"
Y_TEST_PATH = RUNTIME_DIR / "room_type_y_test.csv"

SHAP_GLOBAL_ENTIRE_PATH = SHAP_DIR / "global_beeswarm_Entire home_apt.png"
SHAP_GLOBAL_PRIVATE_PATH = SHAP_DIR / "global_beeswarm_Private room.png"
SHAP_LOCAL_ENTIRE_PATH = SHAP_DIR / "local_shap_42_Entire home_apt.png"
SHAP_LOCAL_PRIVATE_PATH = SHAP_DIR / "local_shap_42_Private room.png"

LIME_PLOT_PATH = LIME_DIR / "lime_plot_42.png"
LIME_REPORT_PATH = LIME_DIR / "lime_report_42.html"

XAI_OUTPUT_PATH = XAI_REPORT_DIR / "airbnb_room_type_xai_output.csv"
XAI_DOCUMENT_PATH = XAI_REPORT_DIR / "Airbnb_RoomType_XAI_Report.docx"

MODEL_COMPARISON_PLOT_PATH = OUTPUTS_DIR / "model_comparison_boxplot.png"
KS_DRIFT_REPORT_PATH = DRIFT_REPORTS_DIR / "ks_drift_report.csv"
KS_DRIFT_TEXT_PATH = DRIFT_REPORTS_DIR / "ks_drift_report.txt"

SHAP_MAX_FEATURES = 10
LIME_MAX_FEATURES = 10

LAST_PREDICTION: dict = {}
LAST_EXPLANATION: dict[str, list] = {
    "shap": [],
    "lime": [],
}

ROOM_TYPE_ARTIFACTS = get_room_type_artifacts()

MODEL_INPUT_COLUMNS: list[str] = ROOM_TYPE_ARTIFACTS.input_columns.copy()
FEATURE_COLUMNS: list[str] = ROOM_TYPE_ARTIFACTS.input_columns.copy()

def csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return []


def csv_head(path: Path, rows: int = 100) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path).head(rows)
    except Exception:
        return pd.DataFrame()
    
def load_pickle_model(model_path: Path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return joblib.load(model_path)
        except Exception:
            with open(model_path, "rb") as file:
                import pickle
                return pickle.load(file)


@dataclass
class RoomTypeDashboardData:
    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    model: object
    model_input_columns: list[str]

def load_dashboard_data() -> RoomTypeDashboardData:
    raw_data = pd.read_csv(RAW_DATASET_PATH, low_memory=False)
    cleaned_data = to_processed_like_notebook(raw_data)

    artifacts = get_room_type_artifacts()

    return RoomTypeDashboardData(
        raw_data=raw_data,
        cleaned_data=cleaned_data,
        model=artifacts.model,
        model_input_columns=artifacts.input_columns,
    )


def load_model_comparison() -> pd.DataFrame:
    comparison_path = OUTPUTS_DIR / "model_comparison_results.csv"
    if comparison_path.exists():
        try:
            return pd.read_csv(comparison_path)
        except Exception:
            pass
    # Fallback: at least show which model is deployed, so the page isn't blank.
    return pd.DataFrame([{"Model": DEPLOYED_MODEL_PATH.stem, "accuracy": None, "roc_auc": None}])

# =============================================================================
# APPEND THIS DIRECTLY AFTER load_model_comparison() - the exact point where
# your pasted file ends. Everything above this point stays unchanged.
# =============================================================================
import matplotlib.pyplot as plt

DATA = load_dashboard_data()   # you defined this function but never called it


# =============================================================================
# SHARED HELPERS
# =============================================================================
def explanation_bar_figure(title: str, contributions: list[tuple[str, float]]):
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    if not contributions:
        ax.axis("off")
        ax.text(0.0, 0.5, "No explanation available.", fontsize=11, color="#33443d")
        fig.tight_layout()
        return fig
    labels = [label for label, _ in contributions][::-1]
    values = np.array([value for _, value in contributions][::-1], dtype=float)
    colors = ["#138a45" if value >= 0 else "#ef4444" for value in values]
    ax.barh(labels, values, color=colors, alpha=0.9)
    ax.axvline(0, color="#24302b", linewidth=1, alpha=0.35)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.16)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_card(title: str, value: str, detail: str, color: str = "#138a45") -> pn.pane.HTML:
    return pn.pane.HTML(
        f"""
        <div class="glass-card" style="min-width:160px; flex:1;">
            <div style="font-size:13px; font-weight:700; color:#243447;">{title}</div>
            <div style="font-size:26px; font-weight:800; color:{color}; margin-top:6px;">{value}</div>
            <div style="font-size:12px; color:#5f7186; margin-top:4px;">{detail}</div>
        </div>
        """,
        sizing_mode="stretch_width",
    )


def html_report_pane(path: Path) -> pn.viewable.Viewable:
    if not path.exists():
        return pn.pane.HTML(f"<div style='color:#dc2626;'>Report not found: <code>{path.name}</code></div>")
    return pn.pane.HTML(path.read_text(encoding="utf-8", errors="replace"), sizing_mode="stretch_width", height=520)


def image_pane(path: Path, caption: str = "") -> pn.viewable.Viewable:
    if not path.exists():
        return pn.pane.HTML(f"<div style='color:#dc2626;'>Image not found: <code>{path.name}</code></div>")
    return pn.Column(
        pn.pane.PNG(str(path), sizing_mode="stretch_width"),
        pn.pane.HTML(f"<p style='color:#5f7186; font-size:13px;'>{caption}</p>") if caption else pn.Spacer(height=0),
    )


# =============================================================================
# PAGE 1 — VALIDATION RESULTS
# =============================================================================
def report_summary(report: dict | list) -> dict:
    if isinstance(report, list):
        total = len(report)
        passed = sum(
            1
            for result in report
            if isinstance(result, dict)
            and str(result.get("status", result.get("Status", ""))).upper()
            in {"PASS", "SUCCESS"}
        )
        return {"total": total, "passed": passed, "failed": total - passed, "pct": (passed / total * 100) if total else 0}
    if isinstance(report, dict):
        results = report.get("results")
        if isinstance(results, list):
            total = len(results)
            passed = sum(
                1
                for result in results
                if result.get("success") is True
                or str(result.get("status", result.get("Status", ""))).upper()
                in {"PASS", "SUCCESS"}
                or str(result.get("Duck_Status", "")).upper() in {"PASS", "SUCCESS"}
            )
            return {"total": total, "passed": passed, "failed": total - passed, "pct": (passed / total * 100) if total else 0}
        stats = report.get("statistics")
        if isinstance(stats, dict):
            total = int(stats.get("evaluated_expectations", 0) or 0)
            passed = int(stats.get("successful_expectations", 0) or 0)
            return {"total": total, "passed": passed, "failed": total - passed, "pct": float(stats.get("success_percent", 0) or 0)}
    return {"total": 0, "passed": 0, "failed": 0, "pct": 0.0}


def build_global_kpi_bar() -> pn.pane.HTML:
    """Create the shared, responsive KPI bar from generated project artifacts."""
    validation_summaries = [
        report_summary(load_json(REPORTS_DIR / f"clean_{method}.json"))
        for method in ("polars", "GE", "cerberus", "pydantic", "duckdb")
    ]
    available_validation_summaries = [
        summary for summary in validation_summaries if summary["total"]
    ]
    validation_pct = (
        sum(summary["pct"] for summary in available_validation_summaries)
        / len(available_validation_summaries)
        if available_validation_summaries else None
    )
    fully_passing_frameworks = sum(
        summary["pct"] >= 99.999 for summary in available_validation_summaries
    )
    validation_color = (
        "#138a45" if validation_pct is not None and validation_pct >= 95
        else "#f59e0b" if validation_pct is not None and validation_pct >= 80
        else "#dc2626" if validation_pct is not None else "#6b7280"
    )

    comparison_df = load_model_comparison()
    accuracy_column = next(
        (column for column in comparison_df.columns if column.lower() == "accuracy"),
        None,
    )
    model_column = next(
        (column for column in comparison_df.columns if column.lower() == "model"),
        None,
    )
    best_accuracy = None
    best_model = "No model metric available"
    if accuracy_column:
        accuracy_values = pd.to_numeric(comparison_df[accuracy_column], errors="coerce")
        if accuracy_values.notna().any():
            best_index = accuracy_values.idxmax()
            best_accuracy = float(accuracy_values.loc[best_index])
            if model_column:
                best_model = str(comparison_df.loc[best_index, model_column])
    model_color = "#138a45" if best_accuracy and best_accuracy >= .70 else "#f59e0b" if best_accuracy else "#6b7280"

    drift_df = (
        pd.read_csv(KS_DRIFT_REPORT_PATH)
        if KS_DRIFT_REPORT_PATH.exists()
        else pd.DataFrame()
    )
    drifted_count = 0
    if not drift_df.empty:
        if "drifted" in drift_df:
            drifted_count = int(pd.to_numeric(drift_df["drifted"], errors="coerce").fillna(0).astype(bool).sum())
        elif "status" in drift_df:
            drifted_count = int(drift_df["status"].astype(str).str.lower().eq("drift").sum())
    drift_color = "#dc2626" if drifted_count else "#138a45"
    drift_value = f"{drifted_count} feature{'s' if drifted_count != 1 else ''} drifted"
    drift_detail = f"{len(drift_df)} feature checks" if not drift_df.empty else "Drift report unavailable"

    cards = [
        ("Total listings", f"{len(DATA.raw_data):,}", "Raw Airbnb listings loaded", "#2563eb"),
        (
            "Clean validation health",
            f"{validation_pct:.1f}%" if validation_pct is not None else "—",
            f"{fully_passing_frameworks}/{len(available_validation_summaries)} frameworks fully passing" if available_validation_summaries else "No clean validation reports found",
            validation_color,
        ),
        ("Best model accuracy", f"{best_accuracy:.1%}" if best_accuracy is not None else "—", escape(best_model), model_color),
        ("Current drift status", drift_value, drift_detail, drift_color),
    ]
    card_html = "".join(
        f"""
        <div class="dashboard-kpi-card" style="--kpi-color: {color};">
          <div class="dashboard-kpi-label">{title}</div>
          <div class="dashboard-kpi-value">{value}</div>
          <div class="dashboard-kpi-detail">{detail}</div>
        </div>
        """
        for title, value, detail, color in cards
    )
    return pn.pane.HTML(
        f'<div class="dashboard-kpi-grid">{card_html}</div>',
        sizing_mode="stretch_width",
    )


def build_validation_page() -> pn.viewable.Viewable:
    methods = ["polars", "GE", "cerberus", "pydantic", "duckdb"]
    method_select = pn.widgets.Select(name="Method", options=methods, value="GE", width=200)
    stage_select = pn.widgets.RadioButtonGroup(options=["Raw", "Clean"], value="Raw")
    report_pane = pn.Column(sizing_mode="stretch_both")

    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.transform import dodge
    from bokeh.plotting import figure

    chart_data = {"Raw": [], "Clean": []}
    for method in methods:
        for stage in ("Raw", "Clean"):
            summary = report_summary(load_json(REPORTS_DIR / f"{stage.lower()}_{method}.json"))
            chart_data[stage].append({"method": method, "pass_rate": summary["pct"], "passed": summary["passed"], "total": summary["total"]})
    raw_source = ColumnDataSource(pd.DataFrame(chart_data["Raw"]))
    clean_source = ColumnDataSource(pd.DataFrame(chart_data["Clean"]))
    validation_chart = figure(
        x_range=methods, height=340, title="Validation pass rate by framework",
        toolbar_location="above", sizing_mode="stretch_width", y_range=(0, 105),
    )
    raw_bars = validation_chart.vbar(x=dodge("method", -.18, range=validation_chart.x_range), top="pass_rate", source=raw_source,
                                     width=.32, color="#f59e0b", alpha=.88, legend_label="Raw")
    clean_bars = validation_chart.vbar(x=dodge("method", .18, range=validation_chart.x_range), top="pass_rate", source=clean_source,
                                       width=.32, color="#168a55", alpha=.88, legend_label="Clean")
    validation_chart.add_tools(HoverTool(renderers=[raw_bars], tooltips=[("Framework", "@method"), ("Dataset", "Raw"), ("Pass rate", "@pass_rate{0.0}%"), ("Passed", "@passed{0,0} / @total{0,0}")]))
    validation_chart.add_tools(HoverTool(renderers=[clean_bars], tooltips=[("Framework", "@method"), ("Dataset", "Clean"), ("Pass rate", "@pass_rate{0.0}%"), ("Passed", "@passed{0,0} / @total{0,0}")]))
    validation_chart.yaxis.axis_label = "Pass rate (%)"

    def refresh(event=None):
        m, stage = method_select.value, stage_select.value
        folder = "polars" if m == "polars" else m
        fname = f"{stage.lower()}_{m}_report.html"
        report_pane.objects = [html_report_pane(HTML_DIR / folder / fname)]

    method_select.param.watch(refresh, "value")
    stage_select.param.watch(refresh, "value")
    refresh()

    return pn.Column(
        pn.pane.HTML("<h2>Validation Results</h2>"),
        pn.pane.Markdown("Pass rates are computed from each framework's generated raw and clean report."),
        validation_chart,
        pn.pane.Markdown("### Generated report"),
        pn.Row(method_select, stage_select),
        pn.Accordion(("Open detailed generated report", report_pane), active=[], sizing_mode="stretch_width"),
        sizing_mode="stretch_both",
    )


# =============================================================================
# PAGE 2 — DRIFT ANALYSIS
# =============================================================================
def build_drift_page() -> pn.viewable.Viewable:
    drift_df = pd.read_csv(KS_DRIFT_REPORT_PATH) if KS_DRIFT_REPORT_PATH.exists() else pd.DataFrame()
    if drift_df.empty:
        return pn.Column(
            pn.pane.HTML("<h2>Drift Analysis</h2>"),
            pn.pane.HTML("<div class='section-card'>KS drift report is not available yet.</div>"),
        )
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.transform import dodge
    from bokeh.plotting import figure

    mode_select = pn.widgets.RadioButtonGroup(
        name="Distribution view", options=["Overlay", "Baseline (train)", "Current (test)"], value="Overlay"
    )
    source = ColumnDataSource(drift_df)
    chart = figure(
        x_range=drift_df["feature"].astype(str).tolist(), height=360,
        title="Feature means: training baseline vs current test data",
        toolbar_location="above", sizing_mode="stretch_width",
    )
    baseline_bars = chart.vbar(x=dodge("feature", -.18, range=chart.x_range), top="train_mean", source=source, width=.34, color="#2563eb", alpha=.75, legend_label="Baseline (train)")
    current_bars = chart.vbar(x=dodge("feature", .18, range=chart.x_range), top="test_mean", source=source, width=.34, color="#f59e0b", alpha=.75, legend_label="Current (test)")
    chart.add_tools(HoverTool(renderers=[baseline_bars], tooltips=[("Feature", "@feature"), ("Baseline mean", "@train_mean{0.000}"), ("KS p-value", "@p_value{0.0000}"), ("Status", "@status")]))
    chart.add_tools(HoverTool(renderers=[current_bars], tooltips=[("Feature", "@feature"), ("Current mean", "@test_mean{0.000}"), ("Mean shift", "@mean_shift_pct{0.000}%"), ("Status", "@status")]))
    chart.xaxis.major_label_orientation = .8
    chart.legend.location = "top_left"
    table = pn.widgets.Tabulator(drift_df, pagination="local", page_size=12, height=400, disabled=True, sizing_mode="stretch_width")

    def refresh_mode(event=None):
        baseline_bars.visible = mode_select.value in {"Overlay", "Baseline (train)"}
        current_bars.visible = mode_select.value in {"Overlay", "Current (test)"}

    mode_select.param.watch(refresh_mode, "value")
    return pn.Column(
        pn.pane.HTML("<h2>Drift Analysis</h2>"),
        pn.pane.Markdown("Toggle the live baseline/current overlay; hover a bar for KS and shift details."),
        mode_select,
        chart,
        pn.Accordion(("Open feature-level KS results", table), active=[], sizing_mode="stretch_width"),
        sizing_mode="stretch_both",
    )


# =============================================================================
# PAGE 3 — ROOT CAUSE ANALYSIS
# =============================================================================
def build_root_cause_page() -> pn.viewable.Viewable:
    stage_select = pn.widgets.RadioButtonGroup(options=["Raw", "Clean"], value="Raw")
    summary_pane = pn.pane.HTML(sizing_mode="stretch_width")
    chart_pane = pn.Column(sizing_mode="stretch_width")
    detail_pane = pn.Column(sizing_mode="stretch_both")

    def refresh(event=None):
        stage = stage_select.value.lower()
        report = load_json(ROOT_CAUSE_REPORTS_DIR / f"{stage}_root_cause_report.json")
        summary = report.get("summary", {})
        results = pd.DataFrame(report.get("results", []))
        summary_pane.object = (
            "<div class='section-card'><h3 style='margin:0;'>"
            f"{escape(str(summary.get('most_common_root_cause', 'No root-cause data')))}</h3>"
            f"<p style='color:#5f7186; margin:6px 0 0;'>"
            f"{int(summary.get('failed', 0)):,} flagged of {int(summary.get('total', 0)):,} records "
            f"({float(summary.get('success_pct', 0)):.2f}% passed).</p></div>"
        )
        if results.empty or "Root_Cause" not in results:
            chart_pane.objects = [pn.pane.HTML("<div class='section-card'>No root-cause results available.</div>")]
        else:
            from bokeh.models import ColumnDataSource, HoverTool
            from bokeh.plotting import figure
            causes = results.loc[results["Root_Cause"] != "No risk signal detected", "Root_Cause"].value_counts().head(10)
            source = ColumnDataSource({"cause": causes.index.astype(str).tolist(), "records": causes.tolist()})
            chart = figure(y_range=causes.index.astype(str).tolist()[::-1], height=360,
                           title=f"Top root causes — {stage_select.value} data", toolbar_location="above", sizing_mode="stretch_width")
            chart.hbar(y="cause", right="records", source=source, height=.7, color="#dc2626", alpha=.82)
            chart.add_tools(HoverTool(tooltips=[("Root cause", "@cause"), ("Flagged listings", "@records{0,0}")]))
            chart_pane.objects = [chart]
        detail_pane.objects = [html_report_pane(ROOT_CAUSE_REPORTS_DIR / f"{stage}_root_cause_report.html")]

    stage_select.param.watch(refresh, "value")
    refresh()

    return pn.Column(
        pn.pane.HTML("<h2>Root Cause Analysis</h2>"),
        stage_select,
        summary_pane,
        chart_pane,
        pn.Accordion(("Open detailed generated root-cause report", detail_pane), active=[], sizing_mode="stretch_width"),
        sizing_mode="stretch_both",
    )


# =============================================================================
# PAGE 4 — EXPLAINABILITY
# =============================================================================
def build_explainability_page() -> pn.viewable.Viewable:
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    try:
        test_features = pd.read_csv(X_TEST_PATH)
    except Exception:
        test_features = pd.DataFrame(columns=MODEL_INPUT_COLUMNS)
    try:
        saved_xai = pd.read_csv(XAI_OUTPUT_PATH)
    except Exception:
        saved_xai = pd.DataFrame()

    model = get_room_type_artifacts().model
    if hasattr(model, "feature_importances_"):
        global_values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        global_values = np.abs(np.asarray(model.coef_)[0])
    else:
        global_values = np.zeros(len(MODEL_INPUT_COLUMNS))
    global_importance = pd.DataFrame(
        {"feature": MODEL_INPUT_COLUMNS, "importance": global_values}
    ).sort_values("importance", ascending=False).head(12)
    global_source = ColumnDataSource(global_importance)
    global_chart = figure(
        y_range=global_importance["feature"].astype(str).tolist()[::-1], height=420,
        title="Global model feature importance", toolbar_location="above",
        sizing_mode="stretch_width",
    )
    global_chart.hbar(y="feature", right="importance", source=global_source, height=.7, color="#168a55")
    global_chart.add_tools(HoverTool(tooltips=[("Feature", "@feature"), ("Model importance", "@importance{0.0000}")]))

    xai_docx_download = None
    if XAI_DOCUMENT_PATH.exists():
        xai_docx_download = pn.widgets.FileDownload(
            file=str(XAI_DOCUMENT_PATH), filename=XAI_DOCUMENT_PATH.name,
            button_type="primary", label="Download Full XAI Report (.docx)",
        )

    if test_features.empty:
        return pn.Column(
            pn.pane.HTML("<h2>Explainability</h2>"),
            pn.pane.HTML("<div class='section-card'>Saved test features are unavailable for local explanations.</div>"),
            global_chart,
        )

    sample_select = pn.widgets.IntSlider(
        name="Test listing index", start=0, end=len(test_features) - 1, value=0,
        sizing_mode="stretch_width",
    )
    local_summary = pn.pane.HTML(sizing_mode="stretch_width")
    local_chart_pane = pn.Column(sizing_mode="stretch_width")

    def refresh_local_explanation(event=None):
        index = sample_select.value
        X = test_features.iloc[[index]].reindex(columns=MODEL_INPUT_COLUMNS, fill_value=0)
        _, confidence, labels = predict_room_type(X)
        contributions, contribution_title = local_model_contributions(X)
        persisted_reason = ""
        if index < len(saved_xai) and "xai_reason" in saved_xai:
            persisted_reason = str(saved_xai.iloc[index]["xai_reason"])
        local_summary.object = (
            "<div class='section-card'><div class='selected-detail-source'>Live local explanation</div>"
            f"<div class='selected-detail-title'>{escape(labels[0])} "
            f"<span style='font-size:14px; color:#5f7186;'>({confidence[0] * 100:.1f}% confidence)</span></div>"
            f"<p style='color:#5f7186; margin:0;'>{escape(persisted_reason) if persisted_reason else escape(contribution_title)}</p></div>"
        )
        if not contributions:
            local_chart_pane.objects = [pn.pane.HTML("<div class='section-card'>This estimator does not expose local contribution values.</div>")]
            return
        local_df = pd.DataFrame(contributions, columns=["feature", "contribution"])
        local_df["direction"] = np.where(local_df["contribution"] >= 0, "Positive", "Negative")
        local_df["color"] = np.where(local_df["contribution"] >= 0, "#168a55", "#dc2626")
        local_source = ColumnDataSource(local_df)
        local_chart = figure(
            y_range=local_df["feature"].astype(str).tolist()[::-1], height=330,
            title=contribution_title, toolbar_location="above", sizing_mode="stretch_width",
        )
        local_chart.hbar(y="feature", right="contribution", source=local_source, height=.7, color="color")
        local_chart.add_tools(HoverTool(tooltips=[("Feature", "@feature"), ("Contribution", "@contribution{+0.0000}"), ("Direction", "@direction")]))
        local_chart_pane.objects = [local_chart]

    sample_select.param.watch(refresh_local_explanation, "value")
    refresh_local_explanation()

    return pn.Column(
        pn.pane.HTML("<h2>Explainability</h2>"),
        pn.pane.Markdown("Interactive importance and local contribution views from the deployed model and saved test data."),
        global_chart,
        pn.pane.Markdown("### Local listing explanation"),
        sample_select,
        local_summary,
        local_chart_pane,
        pn.Accordion(
            ("Download generated XAI report", xai_docx_download if xai_docx_download else pn.pane.HTML("Generated XAI document not found.")),
            active=[], sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_both",
    )


# =============================================================================
# PAGE 5 — MODEL COMPARISON
# =============================================================================
def build_model_comparison_page() -> pn.viewable.Viewable:
    comparison_df = load_model_comparison()
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    accuracy_column = next(
        (column for column in comparison_df.columns if column.lower() == "accuracy"),
        None,
    )
    model_column = next(
        (column for column in comparison_df.columns if column.lower() == "model"),
        None,
    )
    if accuracy_column and model_column:
        chart_df = comparison_df[[model_column, accuracy_column]].copy()
        chart_df.columns = ["model", "accuracy"]
        chart_df["accuracy"] = pd.to_numeric(chart_df["accuracy"], errors="coerce")
        chart_df = chart_df.dropna().sort_values("accuracy")
        source = ColumnDataSource(chart_df)
        chart = figure(
            y_range=chart_df["model"].astype(str).tolist(), height=max(300, 60 * len(chart_df)),
            title="Test accuracy by model", toolbar_location="above", sizing_mode="stretch_width", x_range=(0, 1),
        )
        chart.hbar(y="model", right="accuracy", source=source, height=.65, color="#168a55")
        chart.add_tools(HoverTool(tooltips=[("Model", "@model"), ("Accuracy", "@accuracy{0.0%}")]))
    else:
        chart = pn.pane.HTML("<div class='section-card'>No comparable accuracy metrics were found.</div>")
    table = pn.widgets.Tabulator(comparison_df, disabled=True, pagination=None, sizing_mode="stretch_width")
    return pn.Column(
        pn.pane.HTML("<h2>Model Comparison</h2>"),
        pn.pane.Markdown("Hover to compare real test accuracy values from the generated model metrics file."),
        chart,
        pn.Accordion(("Open model metrics table", table), active=[], sizing_mode="stretch_width"),
        sizing_mode="stretch_both",
    )


# =============================================================================
# PAGE 6 — LIVE PREDICTION
# =============================================================================
def _slider(name: str, feature: str, step: float, integer: bool = False):
    lo, hi = ROOM_TYPE_FEATURE_STATS.get(feature, (0.0, 1000.0))
    default = float(NUMERIC_MEDIANS.get(feature, (lo + hi) / 2))
    widget_cls = pn.widgets.IntSlider if integer else pn.widgets.FloatSlider
    if integer:
        start = int(np.floor(lo))
        end = int(np.ceil(hi)) if hi > lo else start + 1
        value = int(np.clip(round(default), start, end))
        return widget_cls(name=name, start=start, end=end, step=int(step), value=value)
    return widget_cls(
        name=name, start=float(lo), end=float(hi) if hi > lo else float(lo) + 1,
        step=step, value=float(np.clip(default, lo, hi if hi > lo else lo + 1)),
    )


def build_live_prediction_page() -> pn.viewable.Viewable:
    price_input = _slider("Price ($)", "price", 1)
    min_nights_input = _slider("Minimum Nights", "minimum_nights", 1, integer=True)
    reviews_input = _slider("Number of Reviews", "number_of_reviews", 1, integer=True)
    reviews_per_month_input = _slider("Reviews per Month", "reviews_per_month", 0.1)
    review_rate_input = _slider("Review Rate", "review_rate_number", 1, integer=True)
    host_listings_input = _slider("Host Listings Count", "calculated_host_listings_count", 1, integer=True)
    availability_input = _slider("Availability (days/365)", "availability_365", 1, integer=True)
    year_input = _slider("Construction Year", "construction_year", 1, integer=True)

    neighbourhood_input = pn.widgets.Select(
        name="Neighbourhood Group",
        options=["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"],
        value="Manhattan",
    )
    cancellation_input = pn.widgets.Select(
        name="Cancellation Policy", options=["flexible", "moderate", "strict"], value="moderate"
    )
    host_verified_input = pn.widgets.Select(
        name="Host Identity Verified", options=["verified", "unconfirmed"], value="verified"
    )
    instant_bookable_input = pn.widgets.Checkbox(name="Instant Bookable", value=True)

    predict_button = pn.widgets.Button(name="Predict Room Type", button_type="primary", sizing_mode="stretch_width")
    result_pane = pn.pane.HTML("", sizing_mode="stretch_width")

    def on_predict(event=None):
        row = {
            "price": price_input.value,
            "minimum_nights": min_nights_input.value,
            "number_of_reviews": reviews_input.value,
            "reviews_per_month": reviews_per_month_input.value,
            "review_rate_number": review_rate_input.value,
            "calculated_host_listings_count": host_listings_input.value,
            "availability_365": availability_input.value,
            "construction_year": year_input.value,
            "neighbourhood_group_clean": neighbourhood_input.value,
            "cancellation_policy": cancellation_input.value,
            "host_identity_verified": host_verified_input.value,
            "instant_bookable": instant_bookable_input.value,
        }
        row_df = pd.DataFrame([row])
        X = build_room_type_feature_frame(row_df, MODEL_INPUT_COLUMNS)
        _, confidence, labels = predict_room_type(X)

        color = "#138a45" if confidence[0] >= 0.7 else "#f59e0b"
        result_pane.object = f"""
        <div class="section-card">
            <div style="font-size:14px; color:#5f7186;">Predicted room_type</div>
            <div style="font-size:28px; font-weight:800; color:{color}; margin-top:6px;">{labels[0]}</div>
            <div style="font-size:14px; color:#5f7186; margin-top:6px;">Confidence: {confidence[0]*100:.1f}%</div>
        </div>
        """

    predict_button.on_click(on_predict)

    form = pn.Column(
        pn.pane.Markdown("### Listing Details"),
        price_input, min_nights_input, reviews_input, reviews_per_month_input,
        review_rate_input, host_listings_input, availability_input, year_input,
        neighbourhood_input, cancellation_input, host_verified_input, instant_bookable_input,
        predict_button,
        styles={"border": "1px solid #d8e4dc", "border-radius": "14px", "padding": "16px"},
        sizing_mode="stretch_width",
    )

    return pn.Column(
        pn.pane.HTML("<h2>Live Prediction</h2>"),
        pn.Row(form, result_pane, sizing_mode="stretch_width"),
        sizing_mode="stretch_both",
    )


# =============================================================================
# NAVIGATION
# =============================================================================
FILTER_ALL = "All"
_filter_source = DATA.cleaned_data
_room_type_options = [
    FILTER_ALL,
    *sorted(_filter_source["room_type"].dropna().astype(str).unique().tolist()),
] if "room_type" in _filter_source else [FILTER_ALL]
_neighbourhood_options = [
    FILTER_ALL,
    *sorted(
        _filter_source["neighbourhood_group_clean"].dropna().astype(str).unique().tolist()
    ),
] if "neighbourhood_group_clean" in _filter_source else [FILTER_ALL]
_prices = pd.to_numeric(_filter_source.get("price", pd.Series(dtype=float)), errors="coerce").dropna()
_price_min = float(_prices.min()) if not _prices.empty else 0.0
_price_max = float(_prices.max()) if not _prices.empty else 1.0
if _price_max <= _price_min:
    _price_max = _price_min + 1.0

room_type_filter = pn.widgets.Select(
    name="Room type", options=_room_type_options, value=FILTER_ALL,
    sizing_mode="stretch_width",
)
neighbourhood_filter = pn.widgets.Select(
    name="Neighbourhood group", options=_neighbourhood_options, value=FILTER_ALL,
    sizing_mode="stretch_width",
)
price_filter = pn.widgets.RangeSlider(
    name="Nightly price", start=_price_min, end=_price_max,
    value=(_price_min, _price_max), step=max((_price_max - _price_min) / 200, 1),
    sizing_mode="stretch_width",
)


def filtered_listings() -> pd.DataFrame:
    """Return the processed listings matching the shared dashboard filters."""
    filtered = DATA.cleaned_data.copy()
    if room_type_filter.value != FILTER_ALL and "room_type" in filtered:
        filtered = filtered[filtered["room_type"].astype(str) == room_type_filter.value]
    if neighbourhood_filter.value != FILTER_ALL and "neighbourhood_group_clean" in filtered:
        filtered = filtered[
            filtered["neighbourhood_group_clean"].astype(str)
            == neighbourhood_filter.value
        ]
    if "price" in filtered:
        price_values = pd.to_numeric(filtered["price"], errors="coerce")
        filtered = filtered[price_values.between(*price_filter.value)]
    return filtered


_OVERVIEW_PAGE_CACHE: pn.viewable.Viewable | None = None
_overview_summary: pn.pane.HTML | None = None
_overview_charts: pn.Column | None = None
_overview_listing_table: pn.widgets.Tabulator | None = None
_overview_detail: pn.pane.HTML | None = None


def local_model_contributions(X: pd.DataFrame) -> tuple[list[tuple[str, float]], str]:
    """Return local model contributions where the deployed estimator exposes them."""
    model = get_room_type_artifacts().model
    if hasattr(model, "booster_"):
        try:
            values = np.asarray(model.booster_.predict(X, pred_contrib=True))
            contributions = values[0, :len(X.columns)]
            ranked = sorted(
                zip(X.columns.tolist(), contributions.tolist()),
                key=lambda item: abs(item[1]), reverse=True,
            )[:6]
            return ranked, "Local LightGBM feature contributions"
        except Exception:
            pass
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)[0]
        contributions = X.iloc[0].to_numpy(dtype=float) * coefficients
        ranked = sorted(
            zip(X.columns.tolist(), contributions.tolist()),
            key=lambda item: abs(item[1]), reverse=True,
        )[:6]
        return ranked, "Local linear-model feature contributions"
    return [], "Local contribution values are unavailable for this deployed model"


def show_selected_listing(event=None) -> None:
    """Render the selected Overview row with its live prediction and explanation."""
    if _overview_listing_table is None or _overview_detail is None:
        return
    if not _overview_listing_table.selection:
        _overview_detail.object = (
            "<div class='selected-detail-card'><div class='selected-detail-title'>Listing details</div>"
            "<p class='selected-detail-label'>Select a listing row to inspect it.</p></div>"
        )
        return
    selected_index = _overview_listing_table.selection[0]
    if selected_index >= len(_overview_listing_table.value):
        return
    row = _overview_listing_table.value.iloc[selected_index]
    try:
        X = build_room_type_feature_frame(
            pd.DataFrame([row]), MODEL_INPUT_COLUMNS
        )
        _, confidence, labels = predict_room_type(X)
        contributions, contribution_title = local_model_contributions(X)
        prediction_html = (
            f"<div class='selected-detail-source'>Live model prediction</div>"
            f"<div class='selected-detail-title'>{escape(labels[0])} "
            f"<span style='font-size:14px; color:#5f7186;'>({confidence[0] * 100:.1f}% confidence)</span></div>"
        )
        contribution_html = "".join(
            f"<div class='selected-detail-row'><span class='selected-detail-label'>{escape(name)}</span>"
            f"<span class='selected-detail-value'>{value:+.4f}</span></div>"
            for name, value in contributions
        ) or "<p class='selected-detail-label'>No local contribution values are available.</p>"
        explanation_html = (
            f"<div class='selected-detail-source'>{escape(contribution_title)}</div>"
            f"{contribution_html}"
        )
    except Exception as exc:
        prediction_html = (
            "<div class='selected-detail-source'>Live model prediction</div>"
            f"<p style='color:#dc2626;'>Prediction unavailable: {escape(str(exc))}</p>"
        )
        explanation_html = ""

    detail_rows = "".join(
        f"<div class='selected-detail-row'><span class='selected-detail-label'>{escape(str(column))}</span>"
        f"<span class='selected-detail-value'>{escape(str(value))}</span></div>"
        for column, value in row.items()
    )
    _overview_detail.object = (
        "<div class='selected-detail-card'>"
        f"{prediction_html}{explanation_html}"
        "<div class='selected-detail-source'>Full listing record</div>"
        f"{detail_rows}</div>"
    )


def refresh_overview(event=None) -> None:
    """Refresh the Overview figures immediately after a shared filter changes."""
    if _overview_summary is None or _overview_charts is None:
        return

    filtered = filtered_listings()
    _overview_summary.object = (
        "<div class='section-card'><h2 style='margin:0;'>Overview</h2>"
        f"<p style='color:#5f7186; margin:6px 0 0;'>"
        f"Showing <strong>{len(filtered):,}</strong> processed listings matching the active filters."
        "</p></div>"
    )
    if filtered.empty:
        if _overview_listing_table is not None:
            _overview_listing_table.value = pd.DataFrame()
            _overview_listing_table.selection = []
        _overview_charts.objects = [
            pn.pane.HTML(
                "<div class='section-card'><h3 style='margin:0;'>No matching listings</h3>"
                "<p style='color:#5f7186;'>Adjust or clear the sidebar filters to view charts.</p></div>"
            )
        ]
        return

    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure

    def bar_chart(frame: pd.DataFrame, category: str, title: str, color: str):
        counts = frame[category].fillna("Unknown").astype(str).value_counts().head(12)
        source = ColumnDataSource({"category": counts.index.tolist(), "listings": counts.tolist()})
        chart = figure(
            x_range=counts.index.tolist(), height=300, title=title,
            toolbar_location="above", sizing_mode="stretch_width",
        )
        chart.vbar(x="category", top="listings", source=source, width=.72, color=color)
        chart.add_tools(HoverTool(tooltips=[("Category", "@category"), ("Listings", "@listings{0,0}")]))
        chart.xaxis.major_label_orientation = 0.75
        chart.yaxis.axis_label = "Listings"
        return chart

    room_chart = bar_chart(filtered, "room_type", "Listings by room type", "#168a55")
    neighbourhood_chart = bar_chart(
        filtered, "neighbourhood_group_clean", "Listings by neighbourhood group", "#2563eb"
    )
    scatter_frame = filtered.copy()
    scatter_frame["price"] = pd.to_numeric(scatter_frame["price"], errors="coerce")
    scatter_frame["availability_365"] = pd.to_numeric(scatter_frame["availability_365"], errors="coerce")
    scatter_frame = scatter_frame.dropna(subset=["price", "availability_365"]).head(2000)
    scatter_source = ColumnDataSource(
        {
            "price": scatter_frame["price"],
            "availability": scatter_frame["availability_365"],
            "room_type": scatter_frame.get("room_type", pd.Series("Unknown", index=scatter_frame.index)).astype(str),
        }
    )
    scatter = figure(
        height=320, title="Price vs availability", toolbar_location="above",
        sizing_mode="stretch_width", x_axis_label="Nightly price", y_axis_label="Availability (days)",
    )
    scatter.circle(x="price", y="availability", source=scatter_source, size=6, alpha=.45, color="#7c3aed")
    scatter.add_tools(HoverTool(tooltips=[("Price", "@price{0,0.00}"), ("Availability", "@availability days"), ("Room type", "@room_type")]))
    _overview_charts.objects = [room_chart, neighbourhood_chart, scatter]
    if _overview_listing_table is not None:
        _overview_listing_table.value = filtered.reset_index(drop=True).head(250)
        _overview_listing_table.selection = []


def build_overview_page() -> pn.viewable.Viewable:
    """Data-derived landing view with filter-responsive interactive charts."""
    global _OVERVIEW_PAGE_CACHE, _overview_summary, _overview_charts, _overview_listing_table, _overview_detail
    if _OVERVIEW_PAGE_CACHE is None:
        _overview_summary = pn.pane.HTML(sizing_mode="stretch_width")
        _overview_charts = pn.Column(sizing_mode="stretch_width")
        _overview_listing_table = pn.widgets.Tabulator(
            pd.DataFrame(), pagination="local", page_size=12, selectable=1,
            height=360, sizing_mode="stretch_width",
        )
        _overview_detail = pn.pane.HTML(sizing_mode="stretch_width")
        _overview_listing_table.param.watch(show_selected_listing, "selection")
        _OVERVIEW_PAGE_CACHE = pn.Column(
            _overview_summary,
            _overview_charts,
            pn.pane.Markdown("### Inspect a listing"),
            pn.pane.Markdown("Select a row to view its full record, live prediction, and model explanation."),
            _overview_listing_table,
            _overview_detail,
            sizing_mode="stretch_both",
        )
        for widget in (room_type_filter, neighbourhood_filter, price_filter):
            widget.param.watch(refresh_overview, "value")
    refresh_overview()
    show_selected_listing()
    return _OVERVIEW_PAGE_CACHE


pages_data = [
    ("Overview", build_overview_page),
    ("Validation Results", build_validation_page),
    ("Root Cause Analysis", build_root_cause_page),
    ("Drift Analysis", build_drift_page),
    ("Model Comparison", build_model_comparison_page),
    ("Explainability", build_explainability_page),
    # Defined later in the module; name lookup happens only when this tab is clicked.
    ("Batch Prediction", lambda: build_batch_prediction_page()),
    ("Live Prediction", build_live_prediction_page),
]

global_kpi_bar = build_global_kpi_bar()
pages_container = pn.Column(pages_data[0][1](), sizing_mode="stretch_both")
nav_buttons: list[pn.widgets.Button] = []


def set_active_page(index: int) -> None:
    pages_container.objects = [pages_data[index][1]()]
    for i, button in enumerate(nav_buttons):
        button.button_type = "success" if i == index else "light"


for index, (label, _) in enumerate(pages_data):
    button = pn.widgets.Button(name=label, button_type="success" if index == 0 else "light", sizing_mode="stretch_width")
    button.on_click(lambda event, i=index: set_active_page(i))
    nav_buttons.append(button)

sidebar = pn.Column(
    pn.pane.HTML(f"<div style='font-size:18px; font-weight:800; padding:10px 4px;'>{APP_SETTINGS['company_name']}</div>"),
    *nav_buttons,
    pn.pane.HTML("<div class='brand-landmark'>LISTING FILTERS</div>"),
    room_type_filter,
    neighbourhood_filter,
    price_filter,
    sizing_mode="stretch_width",
)

header_logo = (
    f"<img src='{LOGO_DATA_URI}' alt='Truerize IQ logo'>" if LOGO_DATA_URI else ""
)
company_header = pn.pane.HTML(
    f"""
    <div class="header-brand-shell">
      <div class="company-header">
        <div class="company-header-logo">{header_logo}</div>
        <div>
          <div class="company-header-title">Truerize IQ Strategic Solutions Pvt Ltd</div>
          <div class="company-header-subtitle">Room Type Intelligence</div>
        </div>
      </div>
    </div>
    """,
    sizing_mode="stretch_width",
)

template = pn.template.FastListTemplate(
    title=APP_SETTINGS["subtitle"],
    header=[company_header],
    sidebar=[sidebar],
    main=[global_kpi_bar, pages_container],
    accent_base_color=APP_SETTINGS["accent_base_color"],
    header_background=APP_SETTINGS["accent_base_color"],
)
template.servable()

# =============================================================================
# PAGE 7 — BATCH PREDICTION (Upload CSV)
# =============================================================================
import io

_BATCH_PREDICTIONS_CACHE: pd.DataFrame | None = None


def build_batch_prediction_page() -> pn.viewable.Viewable:
    global _BATCH_PREDICTIONS_CACHE

    file_input = pn.widgets.FileInput(accept=".csv", name="Upload Dataset (CSV)")
    predict_button = pn.widgets.Button(name="Predict Uploaded Dataset", button_type="success", sizing_mode="stretch_width")
    download_button = pn.widgets.FileDownload(
        label="Download Predicted CSV",
        button_type="default",
        filename="airbnb_room_type_predictions.csv",
        callback=lambda: io.BytesIO(
            (_BATCH_PREDICTIONS_CACHE.to_csv(index=False).encode("utf-8"))
            if _BATCH_PREDICTIONS_CACHE is not None else b""
        ),
        sizing_mode="stretch_width",
        disabled=True,
    )
    status_pane = pn.pane.HTML("", sizing_mode="stretch_width")
    preview_table = pn.widgets.Tabulator(pd.DataFrame(), disabled=True, pagination="local", page_size=10, height=320, sizing_mode="stretch_width")

    record_id_input = pn.widgets.AutocompleteInput(name="Inspect Record ID", placeholder="Type or select record ID", sizing_mode="stretch_width")
    inspect_button = pn.widgets.Button(name="Enter", button_type="success", width=100)
    detail_pane = pn.pane.HTML("", sizing_mode="stretch_width")

    def on_predict(event=None):
        global _BATCH_PREDICTIONS_CACHE

        if not file_input.value:
            status_pane.object = "<p style='color:#dc2626;'>Please choose a CSV file first.</p>"
            return

        try:
            uploaded_df = pd.read_csv(io.BytesIO(file_input.value))
        except Exception as exc:
            status_pane.object = f"<p style='color:#dc2626;'>Could not read the uploaded CSV: {exc}</p>"
            return

        try:
            cleaned_df, X = prepare_upload_for_prediction(uploaded_df)
            _, confidence, labels = predict_room_type(X)
        except Exception as exc:
            status_pane.object = f"<p style='color:#dc2626;'>Prediction failed: {exc}</p>"
            return

        result_df = cleaned_df.copy()
        result_df["predicted_room_type"] = labels
        result_df["confidence"] = (pd.Series(confidence) * 100).round(2)

        _BATCH_PREDICTIONS_CACHE = result_df
        UPLOAD_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(UPLOAD_PREDICTIONS_PATH, index=False)

        preview_table.value = result_df.head(200)
        download_button.disabled = False
        status_pane.object = f"<p style='color:#138a45;'>Predicted {len(result_df)} rows successfully.</p>"

        if "id" in result_df.columns:
            record_id_input.options = result_df["id"].astype(str).tolist()

    def on_inspect(event=None):
        if _BATCH_PREDICTIONS_CACHE is None:
            detail_pane.object = "<p style='color:#dc2626;'>Run a prediction first.</p>"
            return
        if "id" not in _BATCH_PREDICTIONS_CACHE.columns:
            detail_pane.object = "<p style='color:#dc2626;'>Uploaded data has no 'id' column to inspect by.</p>"
            return

        target_id = record_id_input.value
        match = _BATCH_PREDICTIONS_CACHE[_BATCH_PREDICTIONS_CACHE["id"].astype(str) == str(target_id)]
        if match.empty:
            detail_pane.object = f"<p style='color:#dc2626;'>No record found with id = {target_id}.</p>"
            return

        row = match.iloc[0]
        rows_html = "".join(
            f"<div class='selected-detail-row'><span class='selected-detail-label'>{col}</span>"
            f"<span class='selected-detail-value'>{row[col]}</span></div>"
            for col in match.columns
        )
        detail_pane.object = f"""
        <div class="selected-detail-card">
            <div class="selected-detail-title">Record {target_id}</div>
            {rows_html}
        </div>
        """

    predict_button.on_click(on_predict)
    inspect_button.on_click(on_inspect)

    return pn.Column(
        pn.pane.HTML("""
            <div class="section-card">
                <h2 style="margin:0;">Batch Prediction (Upload CSV)</h2>
                <p style="color:#5f7186; margin-top:6px;">
                    Upload an Airbnb-style listings dataset and generate predictions for every row.
                </p>
            </div>
        """),
        file_input,
        pn.Row(predict_button, download_button, sizing_mode="stretch_width"),
        status_pane,
        pn.pane.Markdown("### Predicted Data Preview"),
        preview_table,
        pn.pane.HTML("<h3>ID Preview &amp; Selected Details</h3><p style='color:#5f7186;'>Inspect any predicted listing by its id.</p>"),
        pn.Row(record_id_input, inspect_button),
        detail_pane,
        sizing_mode="stretch_both",
    )
