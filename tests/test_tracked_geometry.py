"""Remembering what the tracker measured, without editing what the model said."""

from __future__ import annotations

import json

from montagewright import tracked_geometry
from montagewright.clipcard import subjects_from_card


CARD = {
    "subjects": [
        # The card frames the description: "held in hands" puts the hand in
        # the box, so it is wider than the phone while agreeing on centre.
        {"label": "the smartphone held in hands", "centre_x": 0.44,
         "centre_y": 0.52, "width": 0.48, "height": 0.88, "moves": False},
        {"label": "the charging puck", "centre_x": 0.80,
         "centre_y": 0.60, "width": 0.09, "height": 0.10, "moves": False},
    ]
}


def _card_at(tmp_path):
    library = tmp_path / "library" / "cards"
    library.mkdir(parents=True)
    path = library / "abc123.json"
    path.write_text(json.dumps(CARD), encoding="utf-8")
    return path


def test_the_card_keeps_saying_what_the_model_said(tmp_path):
    """Cards are content-addressed and reruns must stay free."""

    card_path = _card_at(tmp_path)
    before = card_path.read_text(encoding="utf-8")

    tracked_geometry.remember(card_path, {
        "the smartphone held in hands": [
            {"width": 0.20, "height": 0.72}, {"width": 0.20, "height": 0.72},
        ],
    })

    assert card_path.read_text(encoding="utf-8") == before
    assert tracked_geometry.path_for(card_path).is_file()


def test_a_tracked_width_replaces_the_framed_one_and_the_centre_stays(tmp_path):
    card_path = _card_at(tmp_path)
    tracked_geometry.remember(card_path, {
        "the smartphone held in hands": [
            {"width": 0.20, "height": 0.72}, {"width": 0.20, "height": 0.72},
        ],
    })

    boxes = {
        one.label: one
        for one in tracked_geometry.applied(subjects_from_card(CARD), card_path)
    }
    phone = boxes["the smartphone held in hands"]
    assert phone.width == 0.20
    assert phone.height == 0.72
    # Centre is what the two agree on, and the card measured it at the moment
    # it names rather than over a window a later edit may not use.
    assert phone.centre_x == 0.44
    # A subject this run never followed is left exactly as the card drew it.
    assert boxes["the charging puck"].width == 0.09


def test_agreement_within_the_deadband_is_not_worth_rewriting(tmp_path):
    card_path = _card_at(tmp_path)
    tracked_geometry.remember(card_path, {
        "the charging puck": [{"width": 0.095, "height": 0.10}],
    })

    boxes = {
        one.label: one
        for one in tracked_geometry.applied(subjects_from_card(CARD), card_path)
    }
    assert boxes["the charging puck"].width == 0.09


def test_later_measurements_merge_rather_than_replace(tmp_path):
    """One edit follows a few of a card's subjects; the footage is the same."""

    card_path = _card_at(tmp_path)
    tracked_geometry.remember(card_path, {
        "the smartphone held in hands": [{"width": 0.20, "height": 0.72}],
    })
    tracked_geometry.remember(card_path, {
        "the charging puck": [{"width": 0.20, "height": 0.21}],
    })

    stored = tracked_geometry.read(card_path)
    assert set(stored) == {"the smartphone held in hands", "the charging puck"}


def test_nothing_measured_leaves_every_box_alone(tmp_path):
    card_path = _card_at(tmp_path)
    boxes = tracked_geometry.applied(subjects_from_card(CARD), card_path)
    assert [one.width for one in boxes] == [0.48, 0.09]
    assert tracked_geometry.applied(subjects_from_card(CARD), None)
