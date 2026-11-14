import json
from collections.abc import Mapping
from html import escape
from hmac import compare_digest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.bulk_prediction import (
    BulkCsvError,
    format_bulk_results,
    score_customer_csv,
)
from src.prediction_service import PredictionAssets, PredictionInputError


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "telco_churn.csv"
MODEL_DIR = ROOT_DIR / "models"
EVALUATION_PATH = MODEL_DIR / "evaluation_results.json"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Neural Network": "neural_network.pkl",
}
CHART_COLORS = ["#3B82F6", "#22D3EE", "#8B5CF6", "#34D399"]
NAVY = "#0B1120"
PANEL = "#162338"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"


st.set_page_config(
    page_title="RetainIQ | Customer Retention Intelligence",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: dark;
        --ri-bg: {NAVY};
        --ri-sidebar: #101A2C;
        --ri-card: {PANEL};
        --ri-elevated: #1C2B42;
        --ri-text: {TEXT};
        --ri-secondary: #CBD5E1;
        --ri-muted: {MUTED};
        --ri-blue: #3B82F6;
        --ri-cyan: #22D3EE;
        --ri-violet: #8B5CF6;
        --ri-border: #2B3B52;
    }}
    .stApp {{
        background:
            radial-gradient(ellipse at 68% -12%, rgba(59,130,246,.11), transparent 34%),
            #0B1120;
        color: var(--ri-text);
    }}
    [data-testid="stHeader"] {{
        height: 3.25rem;
        background: rgba(11,17,32,.94);
    }}
    [data-testid="stSidebar"] {{
        background: #101A2C;
        border-right: 1px solid #24344c;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: #CBD5E1; }}
    .ri-brand {{
        display: flex;
        align-items: center;
        gap: .72rem;
        padding: .3rem .1rem .75rem;
    }}
    .ri-brand-mark {{
        width: 40px;
        height: 40px;
        display: grid;
        place-items: center;
        flex: 0 0 40px;
        border: 1px solid rgba(34,211,238,.32);
        border-radius: 12px;
        color: #F8FAFC;
        font-size: 1.2rem;
        background: linear-gradient(145deg, #2563EB, #164E86);
        box-shadow: 0 5px 16px rgba(37,99,235,.2);
    }}
    .ri-brand-name {{
        color: #F8FAFC;
        font-size: 1.08rem;
        font-weight: 760;
        letter-spacing: -.025em;
        line-height: 1.25;
    }}
    .ri-brand-subtitle {{
        margin-top: .13rem;
        color: #94A3B8;
        font-size: .73rem;
        line-height: 1.35;
    }}
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {{
        color: #94A3B8;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: .2rem; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        border: 1px solid transparent;
        border-radius: 9px;
        padding: .55rem .65rem;
        color: #CBD5E1;
        transition: background .16s ease, border-color .16s ease, color .16s ease;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: #172840;
        border-color: #2B3B52;
        color: #F8FAFC;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background: linear-gradient(100deg, rgba(59,130,246,.2), rgba(34,211,238,.07));
        border-color: rgba(59,130,246,.35);
        color: #F8FAFC;
        font-weight: 650;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {{
        width: 0;
        height: 0;
        opacity: 0;
        margin: 0;
        padding: 0;
    }}
    h1, h2, h3 {{ color: #F8FAFC; letter-spacing: -.025em; }}
    h1 {{ font-size: clamp(2rem, 3vw, 2.3rem) !important; font-weight: 720; }}
    h2 {{ font-size: 1.45rem !important; font-weight: 680; }}
    h3 {{ font-size: 1.12rem !important; font-weight: 670; }}
    p, li {{ color: #CBD5E1; }}
    .block-container {{
        max-width: 1320px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 3.25rem;
        padding-bottom: 3rem;
    }}
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {{ gap: 1.15rem; }}
    [data-testid="stMetric"] {{
        background: #162338;
        border: 1px solid #2B3B52;
        padding: 16px 18px;
        border-radius: 12px;
        box-shadow: 0 5px 16px rgba(0,0,0,.15);
        transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
        min-height: 112px;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        border-color: rgba(59,130,246,.55);
        box-shadow: 0 9px 22px rgba(0,0,0,.23);
    }}
    [data-testid="stMetricLabel"] p {{ color: #94A3B8; font-size: .82rem; }}
    [data-testid="stMetricValue"] {{ color: #F8FAFC; font-weight: 680; }}
    [data-testid="stMetricDelta"] {{ color: #22D3EE; }}
    .st-key-prediction-result-panel [data-testid="stMetric"] {{
        min-width: 0;
        height: 118px;
        display: flex;
        align-items: center;
        overflow: visible;
    }}
    .st-key-prediction-result-panel [data-testid="stMetricLabel"] p {{
        white-space: normal;
        overflow-wrap: anywhere;
        line-height: 1.3;
        min-height: 2.4em;
    }}
    .st-key-prediction-result-panel [data-testid="stMetricValue"] {{
        white-space: normal !important;
        overflow-wrap: anywhere;
        word-break: normal;
        font-size: clamp(1.1rem, 2vw, 1.55rem);
        line-height: 1.2;
    }}
    .st-key-overview-kpis [data-testid="stMetric"] {{
        height: 132px;
        min-height: 132px;
        display: flex;
        align-items: center;
    }}
    .st-key-overview-kpis [data-testid="stMetricValue"] {{
        font-size: clamp(1.05rem, 1.45vw, 1.4rem);
        white-space: normal;
        overflow-wrap: anywhere;
        line-height: 1.25;
    }}
    .ri-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: .55rem 0 1.7rem;
    }}
    .ri-kpi-card {{
        min-width: 0;
        min-height: 148px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: .48rem;
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        border: 1px solid #2B3B52;
        background:
            linear-gradient(110deg, rgba(59,130,246,.09), transparent 40%) top left / 100% 3px no-repeat,
            #162338;
        box-shadow: 0 6px 18px rgba(0,0,0,.16);
        transition: transform .17s ease, border-color .17s ease, box-shadow .17s ease;
    }}
    .ri-kpi-card:hover {{
        transform: translateY(-3px);
        border-color: rgba(59,130,246,.58);
        box-shadow: 0 11px 25px rgba(0,0,0,.24);
    }}
    .ri-kpi-head {{
        display: flex;
        align-items: center;
        gap: .55rem;
    }}
    .ri-kpi-icon {{
        display: grid;
        place-items: center;
        width: 27px;
        height: 27px;
        flex: 0 0 27px;
        border-radius: 8px;
        color: #7DD3FC;
        background: rgba(59,130,246,.13);
        font-size: .9rem;
    }}
    .ri-kpi-label {{
        color: #94A3B8;
        font-size: .77rem;
        font-weight: 650;
        letter-spacing: .025em;
        line-height: 1.35;
    }}
    .ri-kpi-value {{
        color: #F8FAFC;
        font-size: clamp(1.15rem, 1.55vw, 1.55rem);
        font-weight: 720;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }}
    .ri-kpi-note {{ color: #CBD5E1; font-size: .81rem; line-height: 1.3; }}
    .ri-panel-title {{ margin: 0 0 .25rem; color: #F8FAFC; font-size: 1.02rem; font-weight: 680; }}
    .ri-panel-note {{ margin: 0 0 .55rem; color: #94A3B8; font-size: .82rem; line-height: 1.45; }}
    .ri-takeaway {{
        padding: 1.1rem 1.3rem;
        border: 1px solid #2B3B52;
        border-left: 3px solid #22D3EE;
        border-radius: 12px;
        background: linear-gradient(100deg, rgba(59,130,246,.12), rgba(34,211,238,.045)), #162338;
        color: #CBD5E1;
        line-height: 1.55;
    }}
    .ri-page-description {{
        max-width: 850px;
        color: #CBD5E1;
        font-size: .98rem;
        line-height: 1.55;
        margin-top: -.35rem;
        margin-bottom: 1.45rem;
    }}
    .ri-page-eyebrow {{
        margin: 0 0 .45rem;
        color: #67E8F9;
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .12em;
        line-height: 1.35;
    }}
    [data-testid="stPlotlyChart"] {{
        border: 1px solid #2B3B52;
        border-radius: 12px;
        background: #162338;
        padding: .35rem;
        overflow: hidden;
    }}
    [data-testid="stForm"], [data-testid="stExpander"] {{
        border: 1px solid #2B3B52;
        border-radius: 12px;
        background: #162338;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: #2B3B52 !important;
        border-radius: 13px !important;
        background: #162338;
        box-shadow: 0 5px 16px rgba(0,0,0,.14);
        transition: border-color .16s ease, box-shadow .16s ease;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: rgba(59,130,246,.42) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,.19);
    }}
    .stButton > button, [data-testid="stFormSubmitButton"] button {{
        background: linear-gradient(110deg, #2563EB, #1D72C9);
        color: #F8FAFC;
        border: 1px solid rgba(96,165,250,.45);
        border-radius: 9px;
        font-weight: 650;
        transition: filter .15s ease, box-shadow .15s ease, transform .15s ease;
    }}
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {{
        filter: brightness(1.08);
        box-shadow: 0 5px 15px rgba(37,99,235,.25);
        border-color: #22D3EE;
        transform: translateY(-1px);
    }}
    .stButton > button:active, [data-testid="stFormSubmitButton"] button:active {{
        transform: translateY(0);
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid #2B3B52;
        border-radius: 10px;
        overflow: hidden;
        background: #162338;
    }}
    [data-testid="stCaptionContainer"] {{ color: #94A3B8; }}
    [data-testid="stAlert"] {{
        border-radius: 10px;
        border: 1px solid #2B3B52;
        background: #162338;
    }}
    [data-testid="stWidgetLabel"] p, label {{ color: #CBD5E1; }}
    input, textarea, [data-baseweb="select"] > div {{
        color: #F8FAFC !important;
        border-color: #2B3B52 !important;
        border-radius: 8px !important;
        background-color: #1C2B42 !important;
    }}
    [data-baseweb="popover"] {{
        background: #1C2B42 !important;
        border: 1px solid #2B3B52 !important;
        border-radius: 9px !important;
        box-shadow: 0 12px 28px rgba(0,0,0,.35) !important;
    }}
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] [role="listbox"] {{
        background: #1C2B42 !important;
        color: #F8FAFC !important;
    }}
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] [role="option"],
    [role="listbox"] [role="option"] {{
        background: #1C2B42 !important;
        color: #F8FAFC !important;
        transition: background-color .12s ease, color .12s ease;
    }}
    [data-baseweb="popover"] [role="option"] *,
    [data-baseweb="menu"] [role="option"] *,
    [role="listbox"] [role="option"] * {{
        color: inherit !important;
    }}
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    [role="listbox"] [role="option"]:hover,
    [data-baseweb="popover"] [role="option"][data-highlighted="true"],
    [data-baseweb="menu"] [role="option"][data-highlighted="true"],
    [role="listbox"] [role="option"][data-highlighted="true"] {{
        background: #263b58 !important;
        color: #F8FAFC !important;
    }}
    [data-baseweb="popover"] [role="option"][aria-selected="true"],
    [data-baseweb="menu"] [role="option"][aria-selected="true"],
    [role="listbox"] [role="option"][aria-selected="true"] {{
        background: #1d4167 !important;
        color: #FFFFFF !important;
    }}
    [data-baseweb="popover"] [role="option"]:focus-visible,
    [data-baseweb="menu"] [role="option"]:focus-visible,
    [role="listbox"] [role="option"]:focus-visible {{
        outline: 2px solid #22D3EE !important;
        outline-offset: -2px;
    }}
    [data-baseweb="select"]:focus-within > div {{
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 1px #3B82F6 !important;
    }}
    [data-testid="stDataFrame"] *, [data-testid="stTable"] * {{ color: #CBD5E1; }}
    [data-testid="stProgress"] > div > div > div > div {{
        background: linear-gradient(90deg, #3B82F6, #22D3EE);
        border-radius: 999px;
    }}
    [data-testid="stProgress"] > div > div {{
        background: #1C2B42;
        border-radius: 999px;
    }}
    hr {{ border-color: #2B3B52; }}
    .st-key-ri-login-card {{
        width: min(100%, 440px);
        margin: clamp(2rem, 9vh, 6rem) auto;
        padding: clamp(1.6rem, 5vw, 2.45rem);
        border: 1px solid #2B3B52;
        border-radius: 18px;
        background:
            linear-gradient(145deg, rgba(59,130,246,.08), transparent 38%) top left / 100% 4px no-repeat,
            #162338;
        box-shadow: 0 24px 70px rgba(0,0,0,.32);
    }}
    .st-key-ri-login-card [data-testid="stForm"] {{
        padding: 0;
        border: 0;
        background: transparent;
        box-shadow: none;
    }}
    .ri-login-brand {{
        display: flex;
        justify-content: center;
        margin-bottom: 1.35rem;
    }}
    .ri-login-title {{
        margin: 0;
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 720;
        letter-spacing: -.035em;
        text-align: center;
    }}
    .ri-login-copy {{
        margin: .55rem 0 1.55rem;
        color: #94A3B8;
        font-size: .94rem;
        line-height: 1.55;
        text-align: center;
    }}
    .ri-login-config {{
        margin-top: .8rem;
        padding: .8rem .9rem;
        border: 1px solid #2B3B52;
        border-radius: 9px;
        background: #101A2C;
        color: #CBD5E1;
        font-size: .82rem;
        line-height: 1.5;
    }}
    @media (max-width: 900px) {{
        .block-container {{ padding-left: 1.5rem; padding-right: 1.5rem; padding-top: 2.75rem; }}
        .st-key-overview-kpis [data-testid="stMetric"] {{ padding: 14px; }}
        .ri-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .ri-kpi-grid {{ grid-template-columns: 1fr; }}
        .ri-kpi-card {{ min-height: 112px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_auth_credentials():
    try:
        auth_settings = st.secrets["auth"]
    except (StreamlitSecretNotFoundError, KeyError):
        return None, (
            "Login credentials are not configured. Create "
            f"{ROOT_DIR / '.streamlit' / 'secrets.toml'} with an [auth] section."
        )

    if not isinstance(auth_settings, Mapping):
        return None, "The [auth] section in .streamlit/secrets.toml must contain email and password values."
    email = auth_settings.get("email")
    password = auth_settings.get("password")
    if not isinstance(email, str) or not email.strip() or not isinstance(password, str) or not password:
        return None, (
            "Login credentials are incomplete. Set non-empty auth.email and auth.password "
            "values in .streamlit/secrets.toml."
        )
    return {"email": email.strip(), "password": password}, None


def render_login(credentials=None, config_error=None):
    _, center, _ = st.columns([1, 1.05, 1])
    with center:
        with st.container(key="ri-login-card"):
            st.markdown(
                """
                <div class="ri-login-brand">
                  <div class="ri-brand">
                    <div class="ri-brand-mark">💡</div>
                    <div>
                      <div class="ri-brand-name">RetainIQ</div>
                      <div class="ri-brand-subtitle">Customer Retention Intelligence</div>
                    </div>
                  </div>
                </div>
                <div class="ri-login-title">Welcome Back</div>
                <div class="ri-login-copy">Sign in to continue to your customer retention workspace.</div>
                """,
                unsafe_allow_html=True,
            )

            if config_error:
                st.error(config_error)
                st.markdown(
                    """
                    <div class="ri-login-config">
                    Required configuration in <code>.streamlit/secrets.toml</code>:<br>
                    <code>[auth]</code><br>
                    <code>email = "your-demo-email@example.com"</code><br>
                    <code>password = "choose-a-private-password"</code><br><br>
                    Keep this file private and excluded from version control.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return

            email = st.text_input(
                "Email",
                placeholder="name@example.com",
                autocomplete="email",
                key="ri_login_email",
            )
            show_password = st.checkbox("Show password", key="ri_show_password")
            password = st.text_input(
                "Password",
                type="default" if show_password else "password",
                placeholder="Enter your password",
                autocomplete="current-password",
                key="ri_login_password",
            )
            submitted = st.button("Sign In", use_container_width=True, key="ri_sign_in")

            if submitted:
                valid_email = compare_digest(
                    email.strip().casefold().encode("utf-8"),
                    credentials["email"].casefold().encode("utf-8"),
                )
                valid_password = compare_digest(
                    password.encode("utf-8"),
                    credentials["password"].encode("utf-8"),
                )
                if valid_email and valid_password:
                    st.session_state["ri_authenticated"] = True
                    st.session_state.pop("ri_login_error", None)
                    st.session_state.pop("ri_login_password", None)
                    st.session_state.pop("ri_show_password", None)
                    st.rerun()
                else:
                    st.session_state["ri_login_error"] = "Email or password is incorrect."
                    st.rerun()

            if st.session_state.get("ri_login_error"):
                st.error(st.session_state["ri_login_error"])


auth_credentials, auth_config_error = load_auth_credentials()
if not st.session_state.get("ri_authenticated", False):
    render_login(auth_credentials, auth_config_error)
    st.stop()


def chart_layout(fig, height=360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", family="Inter, Segoe UI, sans-serif", size=12),
        title=dict(font=dict(size=15, color="#F8FAFC")),
        margin=dict(l=22, r=18, t=48, b=30),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(bgcolor="#1C2B42", bordercolor="#2B3B52", font_color="#F8FAFC"),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#2B3B52",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#CBD5E1"),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.14)",
        zeroline=False,
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#CBD5E1"),
    )
    return fig


def page_heading(eyebrow, title, description):
    st.markdown(
        f"<div class='ri-page-eyebrow'>{escape(eyebrow.upper())}</div>",
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f"<div class='ri-page-description'>{escape(description)}</div>",
        unsafe_allow_html=True,
    )


def churn_summary(data, column):
    return (
        data.assign(_churn=(data["Churn"] == "Yes").astype(int))
        .groupby(column, observed=True)
        .agg(churn_rate=("_churn", "mean"), customers=("_churn", "size"))
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )


@st.cache_data
def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError(f"Dataset does not contain the required 'Churn' column: {DATA_PATH}")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        if "tenure" in df.columns:
            df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df


@st.cache_data
def load_evaluation_results():
    if not EVALUATION_PATH.is_file():
        return None, f"Evaluation results not found: {EVALUATION_PATH}"
    try:
        with EVALUATION_PATH.open("r", encoding="utf-8") as file:
            return json.load(file), None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"Could not read evaluation results at {EVALUATION_PATH}: {error}"


@st.cache_resource
def load_models():
    missing = [
        str(MODEL_DIR / filename)
        for filename in MODEL_FILES.values()
        if not (MODEL_DIR / filename).is_file()
    ]
    if missing:
        return {}, "Missing trained model file(s): " + ", ".join(missing)
    loaded_models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    return loaded_models, None


def metric_value(metrics, key):
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


try:
    df = load_data()
except (FileNotFoundError, pd.errors.ParserError, ValueError) as error:
    st.error(f"Unable to load the project dataset. {error}")
    st.stop()

eval_data, evaluation_error = load_evaluation_results()
models, model_error = load_models()


st.sidebar.markdown(
    """
    <div class="ri-brand">
      <div class="ri-brand-mark">💡</div>
      <div>
        <div class="ri-brand-name">RetainIQ</div>
        <div class="ri-brand-subtitle">Customer Retention Intelligence</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if st.sidebar.button("↪  Log out", use_container_width=True):
    st.session_state["ri_authenticated"] = False
    st.session_state.pop("ri_login_error", None)
    st.session_state.pop("ri_login_email", None)
    st.session_state.pop("ri_login_password", None)
    st.session_state.pop("ri_show_password", None)
    st.rerun()
st.sidebar.markdown("---")
page = st.sidebar.radio(
    " ",
    [
        "Overview",
        "Customer Churn Prediction",
        "Model Comparison",
        "Cost Calculator",
        "Customer Insights",
        "Bulk CSV Prediction",
    ],
    format_func=lambda label: {
        "Overview": "⌂  Overview",
        "Customer Churn Prediction": "◉  Customer Churn Prediction",
        "Model Comparison": "▥  Model Comparison",
        "Cost Calculator": "＄  Cost Calculator",
        "Customer Insights": "⌁  Customer Insights",
        "Bulk CSV Prediction": "⇧  Bulk CSV Prediction",
    }[label],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Retention analytics**")
st.sidebar.caption("Explore customer patterns, model performance, and illustrative cost scenarios.")
if model_error:
    st.sidebar.error(model_error)
if evaluation_error:
    st.sidebar.warning(evaluation_error)
st.sidebar.markdown("---")
st.sidebar.caption("RetainIQ | Analytics Workspace")


if page == "Overview":
    page_heading(
        "Executive overview",
        "Customer Retention, Clearly in View.",
        "RetainIQ turns observed churn patterns and saved model evaluations into a clear, cost-sensitive view of customer retention.",
    )

    customer_count = len(df)
    churned_count = int((df["Churn"] == "Yes").sum())
    retained_count = customer_count - churned_count
    churn_rate = churned_count / customer_count if customer_count else 0.0
    best_cost_model = None
    best_metrics = {}
    if eval_data:
        best_cost_model = eval_data.get("best_by", {}).get("cost")
        best_metrics = eval_data.get("models", {}).get(best_cost_model, {}).get("test", {})

    best_cost_value = (
        f"${metric_value(best_metrics, 'total_cost'):,.0f}" if best_cost_model and best_metrics else "Unavailable"
    )
    best_model_value = escape(best_cost_model or "Unavailable")
    st.markdown(
        f"""
        <div class="ri-kpi-grid">
          <div class="ri-kpi-card">
            <div class="ri-kpi-head"><span class="ri-kpi-icon">♙</span><div class="ri-kpi-label">Customers in Dataset</div></div>
            <div class="ri-kpi-value">{customer_count:,}</div>
          </div>
          <div class="ri-kpi-card">
            <div class="ri-kpi-head"><span class="ri-kpi-icon">↗</span><div class="ri-kpi-label">Churned Customers</div></div>
            <div class="ri-kpi-value">{churned_count:,}</div>
          </div>
          <div class="ri-kpi-card">
            <div class="ri-kpi-head"><span class="ri-kpi-icon">◷</span><div class="ri-kpi-label">Observed Churn Rate</div></div>
            <div class="ri-kpi-value">{churn_rate:.1%}</div>
          </div>
          <div class="ri-kpi-card">
            <div class="ri-kpi-head"><span class="ri-kpi-icon">✦</span><div class="ri-kpi-label">Lowest Test-Cost Model</div></div>
            <div class="ri-kpi-value">{best_model_value}</div>
            <div class="ri-kpi-note">Test cost: {best_cost_value}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="ri-panel-title">Customer outcomes</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ri-panel-note">Observed retained and churned customers in the full dataset.</div>',
                unsafe_allow_html=True,
            )
            outcome_counts = pd.DataFrame(
                {"Outcome": ["Retained", "Churned"], "Customers": [retained_count, churned_count]}
            )
            fig = px.pie(
                outcome_counts,
                names="Outcome",
                values="Customers",
                hole=0.68,
                color="Outcome",
                color_discrete_map={"Retained": CHART_COLORS[1], "Churned": CHART_COLORS[0]},
            )
            fig.update_traces(
                textposition="inside",
                texttemplate="%{label}<br>%{percent:.1%}",
                marker=dict(line=dict(color=NAVY, width=3)),
                hovertemplate="%{label}: %{value:,} customers (%{percent:.1%})<extra></extra>",
            )
            chart_layout(fig, 330)
            fig.update_layout(showlegend=False, margin=dict(l=12, r=12, t=25, b=16))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            st.markdown('<div class="ri-panel-title">Observed churn by contract</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="ri-panel-note">Share of customers marked as churned within each contract group.</div>',
                unsafe_allow_html=True,
            )
            if "Contract" in df.columns:
                contract_rates = churn_summary(df, "Contract")
                fig = px.bar(
                    contract_rates,
                    x="Contract",
                    y="churn_rate",
                    color="Contract",
                    text=contract_rates["churn_rate"].map(lambda value: f"{value:.1%}"),
                    custom_data=["customers"],
                    color_discrete_sequence=CHART_COLORS,
                    labels={"churn_rate": "Observed churn rate", "Contract": "Contract"},
                )
                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}<br>Churn rate: %{y:.1%}<br>Customers: %{customdata[0]}<extra></extra>",
                )
                fig.update_yaxes(tickformat=".0%", rangemode="tozero")
                chart_layout(fig, 330)
                fig.update_layout(showlegend=False, margin=dict(l=20, r=18, t=25, b=40))
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            else:
                st.info("Contract field is not available in the loaded dataset.")

    st.markdown("#### Key takeaway")
    if "Contract" in df.columns:
        contract_rates = churn_summary(df, "Contract")
        highest = contract_rates.iloc[0]
        lowest = contract_rates.iloc[-1]
        takeaway = (
            f"Churn is highest in the <strong>{escape(str(highest['Contract']))}</strong> group "
            f"({highest['churn_rate']:.1%}; {int(highest['customers']):,} customers) and lowest in the "
            f"<strong>{escape(str(lowest['Contract']))}</strong> group "
            f"({lowest['churn_rate']:.1%}; {int(lowest['customers']):,} customers). "
            "This is an observed association in this dataset, not evidence that contract type causes churn."
        )
        st.markdown(f'<div class="ri-takeaway">{takeaway}</div>', unsafe_allow_html=True)
    else:
        st.info("Contract-based takeaway is unavailable because the dataset has no Contract field.")

    st.markdown("---")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("How RetainIQ works")
            st.markdown(
                "Saved preprocessing-and-classifier pipelines score customer profiles using thresholds selected "
                "on validation data. The reported model evaluation is calculated on the held-out test split. "
                "Cost assumptions are illustrative, not measured company costs or ROI."
            )
    with right:
        with st.container(border=True):
            st.subheader("Lowest test-cost model")
            if best_cost_model and best_metrics:
                info = eval_data["models"][best_cost_model]
                st.markdown(f"### {escape(best_cost_model)}")
                st.metric("Illustrative test cost", f"${metric_value(best_metrics, 'total_cost'):,.0f}")
                st.caption(
                    f"Validation-selected threshold {info['threshold']:.2f} · "
                    f"{int(best_metrics['fn'])} false negatives · {int(best_metrics['fp'])} false positives"
                )
            elif evaluation_error:
                st.warning("Model summary is unavailable until evaluation results are present.")
            else:
                st.info("No best-model result is available.")


elif page == "Customer Churn Prediction":
    page_heading(
        "Individual assessment",
        "Customer churn prediction",
        "Enter a customer profile to score it with a saved model and its validation-selected decision threshold.",
    )
    if model_error:
        st.error(model_error)
        st.stop()
    if not models:
        st.error("No trained models are available.")
        st.stop()
    if not eval_data:
        st.error(evaluation_error or "Evaluation results are required to load the saved model thresholds.")
        st.stop()

    selected_model_name = st.selectbox("Prediction model", list(models.keys()))
    if selected_model_name not in eval_data.get("models", {}):
        st.error(f"No saved evaluation threshold is available for {selected_model_name}.")
        st.stop()

    st.markdown("#### Customer profile")
    st.caption("Choose consistent service options. Fields marked with an asterisk are required.")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("**Account & household**")
            gender = st.selectbox("Gender *", ["Female", "Male"])
            senior_citizen = st.selectbox(
                "Senior citizen *", [0, 1], format_func=lambda value: "Yes" if value else "No"
            )
            partner = st.selectbox("Partner *", ["No", "Yes"])
            dependents = st.selectbox("Dependents *", ["No", "Yes"])
            tenure = st.slider("Tenure (months) *", 0, 72, 12)
            contract = st.selectbox("Contract *", ["Month-to-month", "One year", "Two year"])
            st.markdown("**Billing**")
            paperless_billing = st.selectbox("Paperless billing *", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment method *",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            billing_col1, billing_col2 = st.columns(2)
            with billing_col1:
                monthly_charges = st.number_input(
                    "Monthly charges ($) *", min_value=0.0, max_value=200.0, value=70.0, step=1.0
                )
            with billing_col2:
                total_charges = st.number_input(
                    "Total charges ($) *", min_value=0.0, max_value=12000.0, value=850.0, step=25.0
                )
            st.caption("Charge values are bounded to sensible ranges for the supplied Telco dataset.")
        with col2:
            st.markdown("**Services**")
            phone_service = st.selectbox("Phone service *", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines *", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet service *", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox(
                "Online security *", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup *", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection *", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support *", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV *", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies *", ["No", "Yes", "No internet service"]
            )
        submitted = st.form_submit_button("Score customer", use_container_width=True)

    if submitted:
        validation_errors = []
        if phone_service == "No" and multiple_lines != "No phone service":
            validation_errors.append("Select “No phone service” for multiple lines when phone service is No.")
        if phone_service == "Yes" and multiple_lines == "No phone service":
            validation_errors.append("“No phone service” is only valid when phone service is No.")
        internet_addons = [
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
        ]
        if internet_service == "No" and any(value != "No internet service" for value in internet_addons):
            validation_errors.append("Set all internet add-ons to “No internet service” when internet service is No.")
        if internet_service != "No" and any(value == "No internet service" for value in internet_addons):
            validation_errors.append("“No internet service” add-on values require Internet service to be No.")
        if tenure == 0 and total_charges > 0:
            validation_errors.append("Total charges should be $0 when tenure is zero.")

        if validation_errors:
            for validation_error in validation_errors:
                st.error(validation_error)
        else:
            input_data = pd.DataFrame(
                [
                    {
                        "SeniorCitizen": senior_citizen,
                        "tenure": tenure,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges,
                        "gender": gender,
                        "Partner": partner,
                        "Dependents": dependents,
                        "PhoneService": phone_service,
                        "MultipleLines": multiple_lines,
                        "InternetService": internet_service,
                        "OnlineSecurity": online_security,
                        "OnlineBackup": online_backup,
                        "DeviceProtection": device_protection,
                        "TechSupport": tech_support,
                        "StreamingTV": streaming_tv,
                        "StreamingMovies": streaming_movies,
                        "Contract": contract,
                        "PaperlessBilling": paperless_billing,
                        "PaymentMethod": payment_method,
                    }
                ]
            )
            model = models[selected_model_name]
            if not callable(getattr(model, "predict_proba", None)):
                st.error(f"{selected_model_name} does not support probability predictions.")
                st.stop()
            probabilities = model.predict_proba(input_data)
            class_labels = list(model.classes_)
            if 1 not in class_labels:
                st.error(f"{selected_model_name} has no churn class (1) in its trained classes.")
                st.stop()
            probability = float(probabilities[0, class_labels.index(1)])
            threshold = float(eval_data["models"][selected_model_name]["threshold"])
            predicted_churn = probability >= threshold

            st.markdown("---")
            st.subheader("Prediction result")
            with st.container(key="prediction-result-panel", border=True):
                metric_cols = st.columns(3, gap="large")
                with metric_cols[0]:
                    st.metric("Churn probability", f"{probability:.1%}")
                with metric_cols[1]:
                    st.metric("Saved decision threshold", f"{threshold:.2f}")
                with metric_cols[2]:
                    st.metric("Predicted class", "Churn" if predicted_churn else "Retained")

                result_left, result_right = st.columns([1, 1.2], gap="large")
                with result_left:
                    st.progress(probability, text="Estimated churn probability")
                    if predicted_churn:
                        st.warning("Above threshold · prioritize for retention review")
                    else:
                        st.success("Below threshold · lower model-indicated priority")
                    st.caption(
                        "This is a model score, not a guarantee of future behavior or proof that a retention action will work."
                    )
                with result_right:
                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            number={"suffix": "%", "font": {"size": 34, "color": TEXT}},
                            title={"text": "Estimated churn probability", "font": {"color": MUTED, "size": 15}},
                            gauge={
                                "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": MUTED},
                                "bar": {"color": "#2563eb", "thickness": 0.25},
                                "bgcolor": "#1C2B42",
                                "borderwidth": 0,
                                "steps": [
                                    {"range": [0, threshold * 100], "color": "rgba(34,211,238,.15)"},
                                    {"range": [threshold * 100, 100], "color": "rgba(139,92,246,.16)"},
                                ],
                                "threshold": {
                                    "line": {"color": "#22D3EE", "width": 4},
                                    "thickness": 0.8,
                                    "value": threshold * 100,
                                },
                            },
                        )
                    )
                    gauge.update_layout(
                        height=290,
                        paper_bgcolor="rgba(0,0,0,0)",
                        font={"color": TEXT},
                        margin=dict(l=28, r=28, t=48, b=10),
                    )
                    st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

            model_result = eval_data["models"][selected_model_name]
            importance = model_result.get("permutation_importance", {})
            if importance:
                top_features = sorted(
                    importance.items(), key=lambda item: item[1]["mean"], reverse=True
                )[:5]
                st.markdown("#### Model context")
                st.caption(
                    "Top global permutation-importance features for this model on validation data. "
                    "These describe overall model behavior, not the reasons for this individual score."
                )
                importance_df = pd.DataFrame(
                    [
                        {"Feature": feature, "Importance": values["mean"]}
                        for feature, values in top_features
                    ]
                )
                fig = px.bar(
                    importance_df.sort_values("Importance"),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#a7ded9", "#2563eb"],
                )
                fig.update_layout(coloraxis_showscale=False)
                chart_layout(fig, 280)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


elif page == "Model Comparison":
    page_heading(
        "Evaluation",
        "Compare model performance",
        "Compare the four saved classifiers using test-set metrics at thresholds selected on the validation split.",
    )
    if not eval_data:
        st.error(evaluation_error or "Evaluation results are not available.")
        st.stop()
    model_results = eval_data.get("models", {})
    if not model_results:
        st.error("Evaluation results do not contain any model metrics.")
        st.stop()

    rows = []
    for name in MODEL_FILES:
        info = model_results.get(name)
        if not info or "test" not in info:
            continue
        test = info["test"]
        rows.append(
            {
                "Model": name,
                "Threshold": float(info["threshold"]),
                "Accuracy": metric_value(test, "accuracy"),
                "Precision": metric_value(test, "precision"),
                "Recall": metric_value(test, "recall"),
                "F1-score": metric_value(test, "f1"),
                "ROC-AUC": metric_value(test, "roc_auc"),
                "Test cost": metric_value(test, "total_cost"),
            }
        )
    if not rows:
        st.error("No complete test metrics are available for the saved models.")
        st.stop()

    metric_table = pd.DataFrame(rows)
    best_cost_name = eval_data.get("best_by", {}).get("cost")

    comparison_metrics = {
        "total_cost": ("Test cost ($) · lower is better", "Test cost"),
        "f1": ("F1-score · higher is better", "F1-score"),
        "roc_auc": ("ROC-AUC · higher is better", "ROC-AUC"),
        "recall": ("Recall · higher is better", "Recall"),
        "precision": ("Precision · higher is better", "Precision"),
        "accuracy": ("Accuracy · higher is better", "Accuracy"),
    }
    st.subheader("Metric explorer")
    selected_metric = st.selectbox(
        "Select a metric",
        list(comparison_metrics),
        format_func=lambda key: comparison_metrics[key][0],
    )
    best_by = eval_data.get("best_by", {})
    summary_cols = st.columns(3)
    for column, key, label in zip(
        summary_cols,
        ["cost", "f1", "roc_auc"],
        ["Lowest test cost", "Best test F1", "Best test ROC-AUC"],
    ):
        with column:
            st.metric(label, best_by.get(key, "Unavailable"))

    metric_column = comparison_metrics[selected_metric][1]
    comparison_values = [row[metric_column] for row in rows]
    comparison_df = pd.DataFrame({"Model": [row["Model"] for row in rows], "Value": comparison_values})
    metric_fig = px.bar(
        comparison_df,
        x="Model",
        y="Value",
        color="Model",
        text=(
            comparison_df["Value"].map(lambda value: f"${value:,.0f}")
            if selected_metric == "total_cost"
            else comparison_df["Value"].map(lambda value: f"{value:.3f}")
        ),
        color_discrete_sequence=CHART_COLORS,
        labels={"Value": comparison_metrics[selected_metric][0]},
    )
    metric_fig.update_traces(textposition="outside", cliponaxis=False)
    chart_layout(metric_fig, 340)
    metric_fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=55))
    st.plotly_chart(metric_fig, width="stretch", config={"displayModeBar": False})

    cost_col, matrix_col = st.columns([1.15, 1], gap="large")
    with cost_col:
        st.subheader("Test cost comparison")
        cost_df = metric_table.sort_values("Test cost")
        fig = px.bar(
            cost_df,
            x="Model",
            y="Test cost",
            color="Model",
            text=cost_df["Test cost"].map(lambda value: f"${value:,.0f}"),
            color_discrete_sequence=CHART_COLORS,
            labels={"Test cost": "Illustrative test cost"},
        )
        fig.update_traces(textposition="outside")
        chart_layout(fig, 360)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Costs use the saved FN/FP assumptions and the fixed, validation-selected classifications.")
    with matrix_col:
        st.subheader("Confusion matrix")
        selected_matrix_model = st.selectbox("Model for confusion matrix", list(model_results))
        matrix_info = model_results[selected_matrix_model]
        if "test" not in matrix_info:
            st.info(f"Test confusion counts are not available for {selected_matrix_model}.")
        else:
            test = matrix_info["test"]
            matrix = np.array([[int(test["tn"]), int(test["fp"])], [int(test["fn"]), int(test["tp"])]])
            fig = px.imshow(
                matrix,
                text_auto=True,
                x=["Predicted retained", "Predicted churn"],
                y=["Actual retained", "Actual churn"],
                color_continuous_scale=["#e8f0fb", "#8ab5e8", "#2563eb"],
                aspect="auto",
                labels={"x": "", "y": "", "color": "Customers"},
            )
            chart_layout(fig, 360)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(f"Threshold: {matrix_info['threshold']:.2f} · Actual and predicted class counts.")

    st.markdown("---")
    st.subheader("Detailed test-set comparison")
    st.dataframe(
        metric_table.style.format(
            {
                "Threshold": "{:.2f}",
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "Test cost": "${:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "ROC-AUC measures ranking discrimination across thresholds; it does not measure probability calibration."
    )
    with st.container(border=True):
        st.markdown("**Lowest test-cost model**")
        if best_cost_name and best_cost_name in model_results:
            best_cost_info = model_results[best_cost_name]
            st.markdown(f"### {escape(best_cost_name)}")
            st.markdown(
                f"Illustrative test cost: **${metric_value(best_cost_info.get('test', {}), 'total_cost'):,.0f}** "
                f"at the saved validation-selected threshold **{float(best_cost_info['threshold']):.2f}**."
            )
        else:
            st.info("Lowest-cost model is unavailable in the saved results.")
        st.caption(
            "The selected threshold determines classifications and affects confusion counts and FN/FP cost. "
            "Threshold selection used validation data, not test results."
        )


elif page == "Cost Calculator":
    page_heading(
        "Business scenarios",
        "Cost calculator",
        "Explore how different illustrative false-negative and false-positive costs change the relative costs of saved model classifications.",
    )
    if not eval_data:
        st.error(evaluation_error or "Evaluation results are not available.")
        st.stop()
    model_results = eval_data.get("models", {})
    if not model_results:
        st.error("Evaluation results do not contain model classifications.")
        st.stop()

    saved_assumptions = eval_data.get("cost_assumptions", {})
    default_fn_cost = int(saved_assumptions.get("fn_cost", 100))
    default_fp_cost = int(saved_assumptions.get("fp_cost", 10))
    with st.container(border=True):
        st.markdown("**Illustrative cost assumptions**")
        cost_col1, cost_col2 = st.columns(2, gap="large")
        with cost_col1:
            fn_cost = st.number_input(
                "False-negative cost · missed churner ($)",
                min_value=0,
                max_value=2000,
                value=min(default_fn_cost, 2000),
                step=10,
            )
        with cost_col2:
            fp_cost = st.number_input(
                "False-positive cost · unnecessary offer ($)",
                min_value=0,
                max_value=500,
                value=min(default_fp_cost, 500),
                step=5,
            )
    st.info(
        "Existing test predictions and validation-selected thresholds remain fixed when cost assumptions change. "
        "No threshold is re-optimized from test data."
    )
    with st.expander("How illustrative cost is calculated", expanded=False):
        st.latex(r"\text{Illustrative total cost} = (\text{FN count} \times \text{FN cost}) + (\text{FP count} \times \text{FP cost})")
        st.caption(
            "True positives and true negatives have zero cost in the existing project cost matrix. "
            "All monetary values are illustrative assumptions, not savings or profit."
        )

    recalculated = []
    for name, info in model_results.items():
        test = info.get("test", {})
        fn = int(test.get("fn", 0))
        fp = int(test.get("fp", 0))
        tn = int(test.get("tn", 0))
        tp = int(test.get("tp", 0))
        population = fn + fp + tn + tp
        total_cost = fn * fn_cost + fp * fp_cost
        recalculated.append(
            {
                "Model": name,
                "Threshold (fixed)": float(info.get("threshold", test.get("threshold", 0.5))),
                "False negatives": fn,
                "False positives": fp,
                "Total illustrative cost": total_cost,
                "Cost per evaluated customer": total_cost / population if population else 0.0,
            }
        )
    cost_df = pd.DataFrame(recalculated).sort_values("Total illustrative cost")
    lowest_cost_row = cost_df.iloc[0]
    with st.container(border=True):
        st.markdown("**Lowest illustrative cost at the current assumptions**")
        st.metric(
            lowest_cost_row["Model"],
            f"${lowest_cost_row['Total illustrative cost']:,.2f}",
            delta="Illustrative cost",
            delta_color="off",
        )
        st.caption(
            f"Threshold remains fixed at {lowest_cost_row['Threshold (fixed)']:.2f}; "
            f"{int(lowest_cost_row['False negatives'])} FN · {int(lowest_cost_row['False positives'])} FP."
        )
    st.dataframe(
        cost_df.style.format(
            {
                "Threshold (fixed)": "{:.2f}",
                "Total illustrative cost": "${:,.2f}",
                "Cost per evaluated customer": "${:,.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    fig = px.bar(
        cost_df,
        x="Model",
        y="Total illustrative cost",
        color="Model",
        text=cost_df["Total illustrative cost"].map(lambda value: f"${value:,.0f}"),
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(textposition="outside")
    chart_layout(fig, 370)
    fig.update_layout(showlegend=False, yaxis_title="Illustrative cost ($)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "The cost matrix is an analytical assumption—not actual customer lifetime value, intervention cost, "
        "proven churn prevention, realized savings, or ROI."
    )


elif page == "Customer Insights":
    page_heading(
        "Portfolio exploration",
        "Customer insights",
        "Explore observed churn patterns and model-level feature importance. Associations are descriptive and should not be read as causal effects.",
    )

    contract_col, tenure_col = st.columns(2, gap="large")
    with contract_col:
        with st.container(border=True):
            st.subheader("Contract-related churn")
            if "Contract" in df.columns:
                contract_rates = churn_summary(df, "Contract")
                fig = px.bar(
                    contract_rates,
                    x="Contract",
                    y="churn_rate",
                    color="Contract",
                    text=contract_rates["churn_rate"].map(lambda value: f"{value:.1%}"),
                    custom_data=["customers"],
                    color_discrete_sequence=CHART_COLORS,
                    labels={"churn_rate": "Observed churn rate"},
                )
                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}<br>Churn rate: %{y:.1%}<br>Customers: %{customdata[0]}<extra></extra>",
                )
                fig.update_yaxes(tickformat=".0%", rangemode="tozero")
                chart_layout(fig, 330)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption("Rates are descriptive for each contract group; they do not establish causation.")
            else:
                st.info("Contract field is not available in the loaded dataset.")

    with tenure_col:
        with st.container(border=True):
            st.subheader("Tenure-related churn")
            if "tenure" in df.columns:
                tenure_bins = [-1, 12, 24, 48, 72]
                tenure_labels = ["0–12 months", "13–24 months", "25–48 months", "49–72 months"]
                tenure_data = df.copy()
                tenure_data["Tenure segment"] = pd.cut(
                    tenure_data["tenure"], bins=tenure_bins, labels=tenure_labels
                )
                tenure_summary = (
                    tenure_data.assign(_churn=(tenure_data["Churn"] == "Yes").astype(int))
                    .groupby("Tenure segment", observed=False)
                    .agg(churn_rate=("_churn", "mean"), customers=("_churn", "size"))
                    .reset_index()
                )
                fig = px.bar(
                    tenure_summary,
                    x="Tenure segment",
                    y="churn_rate",
                    color="churn_rate",
                    text=tenure_summary["churn_rate"].map(lambda value: f"{value:.1%}"),
                    custom_data=["customers"],
                    color_continuous_scale=["#a7ded9", "#2563eb"],
                    labels={"churn_rate": "Observed churn rate"},
                )
                fig.update_traces(
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{x}<br>Churn rate: %{y:.1%}<br>Customers: %{customdata[0]}<extra></extra>",
                )
                fig.update_yaxes(tickformat=".0%", rangemode="tozero")
                fig.update_layout(coloraxis_showscale=False)
                chart_layout(fig, 330)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption("Tenure bands are descriptive segments of the supplied dataset.")
            else:
                st.info("Tenure field is not available in the loaded dataset.")

    charges_col, methods_col = st.columns(2, gap="large")
    with charges_col:
        with st.container(border=True):
            st.subheader("Monthly charges and churn")
            if "MonthlyCharges" in df.columns:
                fig = px.box(
                    df,
                    x="Churn",
                    y="MonthlyCharges",
                    color="Churn",
                    color_discrete_sequence=CHART_COLORS,
                    points=False,
                    labels={"MonthlyCharges": "Monthly charges ($)"},
                )
                chart_layout(fig, 330)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption("Box plots compare the distribution of monthly charges by observed churn label.")
            else:
                st.info("Monthly charges field is not available in the loaded dataset.")
    with methods_col:
        with st.container(border=True):
            st.subheader("Other customer patterns")
            insight_field = next(
                (column for column in ["PaymentMethod", "InternetService", "TechSupport"] if column in df.columns),
                None,
            )
            if insight_field:
                category_counts = (
                    df.groupby([insight_field, "Churn"], observed=True)
                    .size()
                    .reset_index(name="customers")
                )
                fig = px.bar(
                    category_counts,
                    x=insight_field,
                    y="customers",
                    color="Churn",
                    barmode="group",
                    color_discrete_map={"No": CHART_COLORS[1], "Yes": CHART_COLORS[0]},
                    labels={"customers": "Customers"},
                )
                fig.update_layout(xaxis_title=insight_field, legend_title="Observed churn")
                chart_layout(fig, 330)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption(f"Customer counts by {insight_field} and the observed churn label.")
            else:
                st.info("No supported categorical fields are available for this comparison.")

    st.markdown("---")
    feature_col, recommendation_col = st.columns([1.15, 1], gap="large")
    with feature_col:
        st.subheader("Global model feature importance")
        if not eval_data or not eval_data.get("models"):
            st.info("Saved permutation-importance results are unavailable.")
        else:
            available_importance = [
                name
                for name, info in eval_data["models"].items()
                if info.get("permutation_importance")
            ]
            if available_importance:
                importance_model = st.selectbox(
                    "Model", available_importance, key="insights_importance_model"
                )
                importance = eval_data["models"][importance_model]["permutation_importance"]
                top_features = sorted(
                    importance.items(), key=lambda item: item[1]["mean"], reverse=True
                )[:10]
                importance_df = pd.DataFrame(
                    [
                        {"Feature": feature, "Importance": values["mean"]}
                        for feature, values in top_features
                    ]
                ).sort_values("Importance")
                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#a7ded9", "#2563eb"],
                    labels={"Importance": "Mean permutation importance (ROC-AUC decrease)"},
                )
                fig.update_layout(coloraxis_showscale=False)
                chart_layout(fig, 390)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.caption(
                    "Importance was computed by permuting each feature on validation data. It is global model context, "
                    "not causal evidence or a per-customer explanation."
                )
            else:
                st.info("No saved permutation-importance values are available.")
    with recommendation_col:
        st.subheader("Illustrative retention hypotheses")
        st.markdown(
            "- Review early-tenure experiences and onboarding, then measure outcomes before scaling changes.\n"
            "- Explore whether contract options and service bundles align with customer needs.\n"
            "- Investigate support and billing friction using customer feedback and operational data.\n"
            "- Test any retention intervention with a controlled design and include offer cost in the business case."
        )
        st.caption(
            "These are investigation ideas based on observed patterns, not proven interventions or predictions of ROI."
        )


elif page == "Bulk CSV Prediction":
    page_heading(
        "Portfolio scoring",
        "Bulk CSV Prediction",
        "Upload customer records to score them with a saved model and its validation-selected threshold.",
    )
    st.caption(
        "Model churn scores are not validated as calibrated probabilities. "
        "Risk ranks are relative to the uploaded rows and do not imply retention savings."
    )

    if model_error:
        st.error(model_error)
        st.stop()
    if not models:
        st.error("No trained models are available.")
        st.stop()
    if not eval_data:
        st.error(evaluation_error or "Saved evaluation thresholds are not available.")
        st.stop()

    with st.form("bulk_csv_prediction_form"):
        selected_model_name = st.selectbox(
            "Prediction model",
            list(models.keys()),
            key="bulk_csv_prediction_model",
        )
        uploaded_file = st.file_uploader(
            "Customer CSV (UTF-8; UTF-8 BOM supported)",
            type=["csv"],
            key="bulk_csv_prediction_file",
        )
        submitted = st.form_submit_button("Score customers", use_container_width=True)

    if submitted:
        if uploaded_file is None:
            st.error("Upload a customer CSV before scoring.")
            st.stop()

        model_evaluation = eval_data.get("models", {}).get(selected_model_name)
        threshold = model_evaluation.get("threshold") if isinstance(model_evaluation, dict) else None
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            st.error(f"No valid saved threshold is available for {selected_model_name}.")
            st.stop()
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            st.error(f"The saved threshold for {selected_model_name} is invalid.")
            st.stop()

        assets = PredictionAssets(
            models=models,
            thresholds={selected_model_name: threshold},
        )
        try:
            batch = score_customer_csv(
                uploaded_file.getvalue(),
                selected_model_name,
                assets,
            )
        except BulkCsvError as error:
            st.error(f"Could not read the uploaded CSV. {error}")
            st.stop()
        except PredictionInputError as error:
            st.error(f"The uploaded CSV does not match the customer feature schema. {error}")
            st.stop()

        if batch.missing_value_counts:
            missing_summary = ", ".join(
                f"{column}: {count}"
                for column, count in batch.missing_value_counts.items()
            )
            st.warning(
                "Missing feature values were handled by the saved pipeline's fitted imputers. "
                f"Missing cells: {missing_summary}."
            )

        display_results = format_bulk_results(batch)
        metric_columns = st.columns(3)
        metric_columns[0].metric("Customers scored", len(display_results))
        metric_columns[1].metric("Selected model", selected_model_name)
        metric_columns[2].metric("Saved threshold", f"{threshold:.2f}")
        st.caption("Risk rank 1 is the highest model churn score. Results remain in original CSV row order.")
        st.dataframe(
            display_results,
            width="stretch",
            hide_index=True,
        )
