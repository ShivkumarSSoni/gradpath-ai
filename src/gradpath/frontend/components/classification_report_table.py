"""Per-outcome precision, recall, F1 and support, as a formatted table."""

import pandas as pd
import streamlit as st


def render_classification_report(rows: list[dict]) -> None:
    """Draw the classification report rows as a Streamlit dataframe."""
    report_dataframe = pd.DataFrame(rows).set_index("Outcome")
    st.dataframe(report_dataframe, use_container_width=True)
