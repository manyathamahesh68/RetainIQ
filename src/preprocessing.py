"""
preprocessing.py
----------------
Loads the Telco churn dataset, runs data-quality checks, cleans it,
splits it into train / validation / test sets, and builds the
scikit-learn preprocessing pipeline.

Key rule: the preprocessor is ONLY fitted on training data, so no
information from validation or test data leaks into training.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42

# Numeric columns (imputed with the median and standardised)
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Categorical columns (imputed with the most frequent value and one-hot encoded)
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _text_columns(df: pd.DataFrame) -> list:
    """Names of non-numeric columns. Works on both pandas 2 and pandas 3."""
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. Raises a clear error if the file is missing or wrong."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Download it from https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "and save it as data/telco_churn.csv inside the RetainIQ folder."
        )
    if not path.is_file():
        raise FileNotFoundError(f"{path} exists but is not a file (is it a folder?).")

    df = pd.read_csv(path)

    required = [ID_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The CSV is missing expected columns: {missing_cols}")
    return df


# ---------------------------------------------------------------------------
# Data-quality report (prints information, changes nothing)
# ---------------------------------------------------------------------------
def run_data_checks(df: pd.DataFrame) -> None:
    """Print shape, columns, missing values, duplicates and blank strings."""
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    print("Missing values (NaN) per column:")
    nan_counts = df.isna().sum()
    print(nan_counts[nan_counts > 0] if nan_counts.sum() > 0 else "  none")

    # Blank strings are NOT detected by isna(), so check text columns separately
    print("\nBlank-string values (' ') per column:")
    blanks = {}
    for col in _text_columns(df):
        n_blank = int((df[col].astype(str).str.strip() == "").sum())
        if n_blank > 0:
            blanks[col] = n_blank
    print(blanks if blanks else "  none")

    print(f"\nDuplicate full rows (including customerID): {df.duplicated().sum()}")
    print(f"Duplicate customerIDs: {df[ID_COLUMN].duplicated().sum()}")
    print(
        "Rows identical on all feature columns (not removed, because they are "
        f"different customers): {df.duplicated(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).sum()}"
    )

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nTarget distribution (Churn):")
    counts = df[TARGET_COLUMN].value_counts()
    for label, n in counts.items():
        print(f"  {label}: {n} ({n / len(df):.1%})")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame):
    """
    Returns (X, y):
      - drops customerID (identifier, not predictive)
      - converts TotalCharges to numeric (blank strings become NaN)
      - sets TotalCharges = 0 for customers with tenure 0 (not billed yet)
      - converts Churn to 1 (Yes) / 0 (No)
    Any other missing values are filled inside the pipeline using
    statistics from the training set only (avoids leakage).
    """
    df = df.copy()
    df = df.drop(columns=[ID_COLUMN])

    # Remove stray whitespace in text columns
    for col in _text_columns(df):
        df[col] = df[col].str.strip()

    # TotalCharges is stored as text because some rows contain " ".
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Customers with tenure 0 have not been billed yet, so TotalCharges = 0 is the
    # logical value. This is a fixed rule (not learned from data), so no leakage.
    df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    # Encode target
    target_map = {"Yes": 1, "No": 0}
    unknown = set(df[TARGET_COLUMN].unique()) - set(target_map)
    if unknown:
        raise ValueError(f"Unexpected values in Churn column: {unknown}")
    y = df[TARGET_COLUMN].map(target_map).astype(int)

    X = df[FEATURE_COLUMNS].copy()
    return X, y


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Stratified 60% train / 20% validation / 20% test split.
    - Test set is set aside first and never used for tuning.
    - Validation set is used later for threshold selection.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    # 0.25 of the remaining 80% = 20% of the total
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Unfitted ColumnTransformer: impute + scale numerics, impute + one-hot categoricals."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def save_preprocessor(preprocessor: ColumnTransformer, path: Path = None) -> Path:
    """Save fitted ColumnTransformer preprocessor to models/preprocessor.pkl."""
    if path is None:
        path = MODELS_DIR / "preprocessor.pkl"
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


# ---------------------------------------------------------------------------
# Run this file directly to test Stage 2
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        raw_df = load_data()
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)

    run_data_checks(raw_df)

    X, y = clean_data(raw_df)
    n_bad = int(X["TotalCharges"].isna().sum())
    print("\n" + "=" * 60)
    print("AFTER CLEANING")
    print("=" * 60)
    print(f"TotalCharges values still missing after cleaning: {n_bad}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print("\n" + "=" * 60)
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)
    for name, features, target in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        print(f"{name:<11}: {len(features):>5} rows | churn rate = {target.mean():.2%}")

    # Fit on TRAIN ONLY, then transform all three sets
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Save the fitted leak-free preprocessor to models/preprocessor.pkl
    saved_path = save_preprocessor(preprocessor)
    print("\n" + "=" * 60)
    print("PREPROCESSOR (fitted on training data only)")
    print("=" * 60)
    print(f"Transformed shapes: train {X_train_t.shape}, val {X_val_t.shape}, test {X_test_t.shape}")
    nan_left = bool(pd.isna(X_train_t).any() or pd.isna(X_val_t).any() or pd.isna(X_test_t).any())
    print(f"Any NaN left after transform? {nan_left}")
    print(f"Fitted ColumnTransformer saved to: {saved_path}")
    print("\nStage 2 checks finished without errors.")