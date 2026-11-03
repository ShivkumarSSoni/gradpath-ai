"""Small wrappers around Streamlit's feedback calls, kept in one place."""

import streamlit as st


def render_success(message: str) -> None:
    st.success(message)


def render_error(message: str) -> None:
    st.error(message)


def render_warning(message: str) -> None:
    st.warning(message)


def render_info(message: str) -> None:
    st.info(message)
