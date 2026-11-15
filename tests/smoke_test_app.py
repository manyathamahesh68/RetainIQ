import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"


def get_test_credentials():
    try:
        credentials = st.secrets["auth"]
    except (StreamlitSecretNotFoundError, KeyError) as error:
        raise RuntimeError(
            "Configure [auth].email and [auth].password in "
            f"{PROJECT_ROOT / '.streamlit' / 'secrets.toml'} before running this smoke test."
        ) from error
    return credentials["email"], credentials["password"]


def test_dashboard():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.bulk_prediction import (
        REPORT_CSV_STATE_KEY,
        REPORT_FILENAME_STATE_KEY,
        REPORT_STATE_KEY,
    )
    from src.preprocessing import DATA_PATH, TARGET_COLUMN

    email, password = get_test_credentials()
    print(f"Loading AppTest from: {APP_PATH}")
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, f"Exception on initial load: {at.exception}"
    assert any("Welcome Back" in item.value for item in at.markdown), "Login screen did not appear."
    assert len(at.text_input) == 2, "Login email and password fields are missing."
    print("✓ Login screen displayed.")

    at.text_input[0].set_value("wrong@example.com")
    at.text_input[1].set_value("incorrect-password")
    at.button[0].click().run()
    assert not at.exception, f"Exception on invalid login: {at.exception}"
    assert any("Email or password is incorrect." in item.value for item in at.error), (
        "Invalid credentials did not show an error."
    )
    print("✓ Invalid credentials show a generic error.")

    at.text_input[0].set_value(email)
    at.text_input[1].set_value(password)
    at.button[0].click().run()
    assert not at.exception, f"Exception on valid login: {at.exception}"
    assert not at.error, f"st.error after valid login: {[e.value for e in at.error]}"
    assert at.sidebar.radio, "Successful login did not open the dashboard."
    at.run()
    assert not at.exception, f"Dashboard session did not survive rerun: {at.exception}"
    print("✓ Valid login opens the dashboard and session persists on rerun.")

    pages = [
        "Overview",
        "Customer Churn Prediction",
        "Model Comparison",
        "Cost Calculator",
        "Customer Insights",
        "Bulk CSV Prediction",
    ]

    # Verify each page
    for page_name in pages:
        print(f"\n--- Testing Page: {page_name} ---")
        # Find radio button for navigation
        nav_radio = at.sidebar.radio[0]
        nav_radio.set_value(page_name).run()
        assert not at.exception, f"Exception on {page_name}: {at.exception}"
        assert not at.error, f"st.error on {page_name}: {[e.value for e in at.error]}"
        print(f"✓ Page '{page_name}' rendered successfully.")

        if page_name == "Customer Churn Prediction":
            print("  Testing prediction form submission...")
            # Form submission
            form_button = at.form[0].button[0]
            form_button.click().run()
            assert not at.exception, f"Exception on prediction submit: {at.exception}"
            assert not at.error, f"st.error on prediction submit: {[e.value for e in at.error]}"
            print("  ✓ Form submission produced prediction results without errors.")

        elif page_name == "Cost Calculator":
            print("  Testing cost calculator with modified inputs...")
            # Modify FN Cost and FP Cost
            fn_input = at.number_input[0]
            fp_input = at.number_input[1]
            fn_input.set_value(150).run()
            assert not at.exception, f"Exception after updating FN cost: {at.exception}"
            assert not at.error, f"st.error after updating FN cost: {[e.value for e in at.error]}"
            fp_input.set_value(25).run()
            assert not at.exception, f"Exception after updating FP cost: {at.exception}"
            assert not at.error, f"st.error after updating FP cost: {[e.value for e in at.error]}"
            print("  ✓ Cost calculator successfully recalculated under new assumptions.")

        elif page_name == "Model Comparison":
            print("  Testing model comparison metric selector...")
            selectbox = at.selectbox[0]
            for metric in ["total_cost", "f1", "roc_auc", "recall"]:
                selectbox.set_value(metric).run()
                assert not at.exception, f"Exception on comparison metric {metric}: {at.exception}"
                assert not at.error, f"st.error on comparison metric {metric}: {[e.value for e in at.error]}"
            print("  ✓ Model comparison metric selector tested successfully.")

        elif page_name == "Bulk CSV Prediction":
            assert at.file_uploader, "Bulk CSV upload control is missing."
            assert at.selectbox, "Bulk prediction model selector is missing."
            at.form[0].button[0].click().run()
            assert not at.exception, f"Exception on empty bulk submission: {at.exception}"
            assert any("Upload a customer CSV" in item.value for item in at.error), (
                "Empty bulk submission did not show an understandable error."
            )
            customer_csv = (
                pd.read_csv(DATA_PATH)
                .drop(columns=[TARGET_COLUMN])
                .iloc[:1]
                .to_csv(index=False)
                .encode("utf-8")
            )
            at.file_uploader[0].upload(
                "smoke_customers.csv",
                customer_csv,
                "text/csv",
            )
            at.form[0].button[0].click().run()
            assert not at.exception, f"Exception after bulk scoring: {at.exception}"
            assert not at.error, f"st.error after bulk scoring: {[e.value for e in at.error]}"
            assert at.download_button, "Bulk prediction CSV download button is missing."
            assert at.download_button[0].label == "Download CSV", (
                "The bulk prediction download button has an unexpected label."
            )
            assert at.session_state[REPORT_CSV_STATE_KEY], (
                "Pre-generated CSV bytes were not stored."
            )
            assert at.session_state[REPORT_FILENAME_STATE_KEY].endswith(".csv")
            assert not at.session_state[REPORT_STATE_KEY].empty
            print("  ✓ Bulk CSV scoring and report download control work.")

    at.sidebar.button[0].click().run()
    assert not at.exception, f"Exception on logout: {at.exception}"
    assert any("Welcome Back" in item.value for item in at.markdown), "Logout did not return to login."
    for key in (REPORT_STATE_KEY, REPORT_CSV_STATE_KEY, REPORT_FILENAME_STATE_KEY):
        assert key not in at.session_state, f"Logout did not clear {key}."
    at.run()
    assert any("Welcome Back" in item.value for item in at.markdown), "Login screen did not persist after rerun."
    print("✓ Logout returns to login and unauthenticated rerun stays on login.")

    print("\n" + "=" * 60)
    print("ALL AUTHENTICATION AND DASHBOARD SMOKE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_dashboard()
