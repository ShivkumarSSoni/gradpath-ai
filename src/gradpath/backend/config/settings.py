"""Central settings for the GradPath AI project.

Every file path and tuning constant used by the notebook, the backend layers,
and the Streamlit app lives here, so there is exactly one place to change
them. Business vocabulary such as column names lives in the domain layer.
"""

from pathlib import Path

# Folder that contains app.py, i.e. the root of this repository.
# This file sits five levels deep: <root>/src/gradpath/backend/config/settings.py
PROJECT_ROOT_FOLDER = Path(__file__).resolve().parents[4]

# Where the saved trained models and their metrics file are stored.
SAVED_MODELS_FOLDER = PROJECT_ROOT_FOLDER / "model"

# One row per classifier with its six scores on the held-out records.
EVALUATION_RESULTS_FILE = SAVED_MODELS_FOLDER / "evaluation_results.csv"

# Held-out test records that the Streamlit app accepts as an upload.
HELD_OUT_TEST_DATA_FILE = PROJECT_ROOT_FOLDER / "test_data.csv"

# A fixed seed makes the split and every model reproducible on any machine.
RANDOM_SEED_FOR_REPRODUCIBILITY = 42

# Share of records kept aside for testing (the rest is used for training).
TEST_SPLIT_FRACTION = 0.20

# Identifier of the UCI dataset "Predict Students' Dropout and Academic Success".
UCI_DATASET_IDENTIFIER = 697

# Local copy of the downloaded records (this folder is git-ignored).
DATASET_CACHE_FILE = PROJECT_ROOT_FOLDER / "data" / "student_records.csv"
