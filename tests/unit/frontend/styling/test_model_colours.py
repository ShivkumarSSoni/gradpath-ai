"""Every classifier must have its own valid, fixed display colour."""

import re

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS

HEX_COLOUR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_every_classifier_kind_has_a_colour():
    assert set(MODEL_DISPLAY_COLOURS.keys()) == set(ClassifierKind.all_kinds())


def test_every_colour_is_a_valid_hex_code():
    for colour in MODEL_DISPLAY_COLOURS.values():
        assert HEX_COLOUR_PATTERN.match(colour), f"{colour} is not a valid hex colour"


def test_every_classifier_has_a_different_colour():
    assert len(set(MODEL_DISPLAY_COLOURS.values())) == len(MODEL_DISPLAY_COLOURS)
