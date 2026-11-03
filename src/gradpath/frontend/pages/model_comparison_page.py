"""The "How does this model compare to the other four?" section.

Shows the fixed, held-out-split scores of all five classifiers. This data
never changes between runs, unlike the rest of the page.
"""

import streamlit as st

from gradpath.frontend.components import comparison_bar_charts, comparison_table, observations_table
from gradpath.frontend.styling import text_constants
from gradpath.frontend.view_models.comparison_view_model import load_comparison_view


def render_page() -> None:
    """Render the observations table, comparison table and comparison charts."""
    st.divider()
    with st.container(border=True):
        st.subheader(f"{text_constants.COMPARISON_SECTION_TITLE} :gray-badge[{text_constants.COMPARISON_SECTION_BADGE}]")
        st.caption(text_constants.COMPARISON_SECTION_SUBTEXT)

        comparison_view = load_comparison_view()

        st.markdown(f"**{text_constants.OBSERVATIONS_SUBHEADING}**")
        observations_table.render_observations_table(comparison_view.observation_rows)

        st.markdown(f"**{text_constants.COMPARISON_TABLE_SUBHEADING}**")
        comparison_table.render_comparison_table(comparison_view.table_rows)

        st.markdown(f"**{text_constants.COMPARISON_CHARTS_SUBHEADING}**")
        st.caption(text_constants.COMPARISON_CHARTS_CAPTION)
        comparison_bar_charts.render_comparison_bar_charts(comparison_view.chart_data_by_metric)
