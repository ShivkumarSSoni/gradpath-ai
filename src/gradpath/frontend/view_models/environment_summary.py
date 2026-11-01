"""Reports the versions of Python and the main libraries in use."""

import platform

import pandas
import sklearn
import streamlit


def collect_environment_summary() -> dict[str, str]:
    """Return library names mapped to their installed versions.

    Shown on the first page so a deployment problem, such as the wrong
    Python version on Streamlit Cloud, is visible at a glance.
    """
    return {
        "Python": platform.python_version(),
        "Streamlit": streamlit.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pandas.__version__,
    }
