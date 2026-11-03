"""Every classifier must have its own valid, fixed display colour."""

import re

from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.frontend.styling.model_colours import MODEL_DISPLAY_COLOURS, build_model_colour_gradient

HEX_COLOUR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_every_classifier_kind_has_a_colour():
    assert set(MODEL_DISPLAY_COLOURS.keys()) == set(ClassifierKind.all_kinds())


def test_every_colour_is_a_valid_hex_code():
    for colour in MODEL_DISPLAY_COLOURS.values():
        assert HEX_COLOUR_PATTERN.match(colour), f"{colour} is not a valid hex colour"


def test_every_classifier_has_a_different_colour():
    assert len(set(MODEL_DISPLAY_COLOURS.values())) == len(MODEL_DISPLAY_COLOURS)


def test_colour_gradient_runs_from_white_to_the_model_colour():
    gradient = build_model_colour_gradient("#1f4e9c")

    white_end = gradient(0.0)
    model_colour_end = gradient(1.0)

    assert white_end[:3] == (1.0, 1.0, 1.0)
    assert model_colour_end[:3] != (1.0, 1.0, 1.0)
