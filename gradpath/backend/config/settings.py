"""Central settings for the GradPath AI project.

Every file path, column name, and tuning constant used by the notebook, the
backend layers, and the Streamlit app lives here, so there is exactly one
place to change them.
"""

from pathlib import Path

# Folder that contains app.py, i.e. the root of this repository.
# This file sits four levels deep: <root>/gradpath/backend/config/settings.py
PROJECT_ROOT_FOLDER = Path(__file__).resolve().parents[3]

# Where the saved trained models and their metrics file are stored.
SAVED_MODELS_FOLDER = PROJECT_ROOT_FOLDER / "model"

# Held-out test records that the Streamlit app accepts as an upload.
HELD_OUT_TEST_DATA_FILE = PROJECT_ROOT_FOLDER / "test_data.csv"

# A fixed seed makes the split and every model reproducible on any machine.
RANDOM_SEED_FOR_REPRODUCIBILITY = 42

# Share of records kept aside for testing (the rest is used for training).
TEST_SPLIT_FRACTION = 0.20

# Name of the column that holds the outcome we want to predict.
OUTCOME_COLUMN_NAME = "Target"

# The three possible outcomes for a student in this dataset.
OUTCOME_CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
