"""Fixed colours used for page chrome (branding, charts), not for one model.

Kept separate from `model_colours.py`, which maps each classifier to its own
colour; these colours are for text and chart elements that are not
classifier-specific.
"""

BRAND_ACCENT_COLOUR = "#1f4e9c"
BRAND_TEXT_COLOUR = "#1b2430"
MUTED_TEXT_COLOUR = "#6b7684"
# A lighter tint of BRAND_ACCENT_COLOUR, used only for the tagline so it
# reads as a distinct brand subtitle rather than plain muted body text.
TAGLINE_COLOUR = "#4a6fa5"

# Feedback banner colours (feedback_banner.py). Deliberately not Streamlit's
# own default green/red/yellow/blue, so the app's status messages read as a
# GradPath AI design choice rather than an out-of-the-box Streamlit look.
FEEDBACK_SUCCESS_COLOUR = "#1f7a5c"
FEEDBACK_ERROR_COLOUR = "#b42318"
FEEDBACK_WARNING_COLOUR = "#b45309"
FEEDBACK_INFO_COLOUR = BRAND_ACCENT_COLOUR
