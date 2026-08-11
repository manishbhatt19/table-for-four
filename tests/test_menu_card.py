"""Tests for the generated menu cards.

Cards are pure functions of retrieved text, so these need no network and no LLM.
The important property is the honest one: a card can only show dishes that appear
verbatim in the retrieved snippets.
"""

from urllib.parse import unquote

from agent.menu_card import (
    build_menu_card,
    card_for_restaurant,
    extract_dishes,
    highlight_note,
    theme_for,
)


def _svg(data_uri: str) -> str:
    assert data_uri.startswith("data:image/svg+xml;utf8,")
    return unquote(data_uri.split(",", 1)[1])


def test_theme_matches_cuisine_and_falls_back():
    assert theme_for("italian_restaurant") is theme_for("italian")
    assert theme_for("sushi") is theme_for("japanese")
    assert theme_for("vegan") is theme_for("plant-based")
    # Unrecognised cuisines still get a card, just a neutral one.
    assert theme_for("molecular gastronomy") is theme_for(None)


def test_extract_dishes_reads_priced_menu_lines():
    highlights = [{"snippet": "Snap Peas, Radish, Dill $23 Chicken Liver Mousse "
                              "Apricot Preserves, Pickles $22"}]
    dishes = extract_dishes(highlights)
    assert "Snap Peas, Radish, Dill" in dishes
    assert any(d.startswith("Chicken Liver Mousse") for d in dishes)


def test_extract_dishes_handles_colon_prices():
    highlights = [{"snippet": "Salmon Crudo with citrus miso cream : 33.00 "
                              "Seafood Ceviche : 28.00"}]
    dishes = extract_dishes(highlights)
    assert any("Salmon Crudo" in d for d in dishes)
    assert any("Seafood Ceviche" in d for d in dishes)


def test_extract_dishes_rejects_addresses_and_page_furniture():
    # The exact failure that makes a card embarrassing: an address or a phone
    # number parsed as though it were something you could order.
    highlights = [{"snippet": "205 East Houston Street, New York, NY 10002 "
                              "Open Monday 8 am Reservations 212 254 2246"}]
    assert extract_dishes(highlights) == []


def test_extract_dishes_returns_nothing_for_navigation_chrome():
    highlights = [{"snippet": "Skip to content Carbone Logo Reservations Instagram"}]
    assert extract_dishes(highlights) == []


def test_dishes_are_strictly_extractive():
    # Nothing on the card that wasn't in the retrieved text.
    snippet = "Charred Hispi Cabbage, Miso Butter $19"
    dishes = extract_dishes([{"snippet": snippet}])
    assert dishes and all(d in snippet for d in dishes)


def test_card_renders_dishes_and_perk():
    card = card_for_restaurant(
        restaurant="Nonna's Gluten-Free Kitchen",
        cuisine="italian",
        highlights=[{"snippet": "Gluten-Free Tasting Flight $34 Wood-Fired Margherita $18"}],
        perk="Gluten-Free Tasting Flight",
    )
    svg = _svg(card["url"])
    assert "Gluten-Free Tasting Flight" in svg
    assert "PERK INCLUDED" in svg
    assert card["dishes"]
    assert card["source"] == "menu card"


def test_sample_perk_is_labelled_as_a_sample():
    svg = _svg(build_menu_card("Verdant", "vegan", ["Beetroot Tartare"],
                               perk="Chef's Tasting", perk_is_sample=True))
    assert "SAMPLE PARTNER OFFER" in svg
    assert "PERK INCLUDED" not in svg


def test_card_without_dishes_still_renders_and_says_nothing_it_cannot_know():
    card = card_for_restaurant("Somewhere", "thai", highlights=[])
    svg = _svg(card["url"])
    assert card["dishes"] == []
    assert "specials" in svg  # the honest fallback line, not an invented dish


def test_highlight_note_reads_as_a_sentence():
    assert highlight_note(["Lasagna"]) .startswith("Diners keep mentioning Lasagna")
    assert "Rigatoni and Lasagna" in highlight_note(["Rigatoni", "Lasagna"])
    assert "Rigatoni, Lasagna and Osso Buco" in highlight_note(
        ["Rigatoni", "Lasagna", "Osso Buco", "Tiramisu"]
    )


def test_highlight_note_mentions_the_perk_when_there_is_one():
    note = highlight_note(["Lasagna"], perk="Weekend Family Feast")
    assert "Weekend Family Feast" in note


def test_highlight_note_admits_when_nothing_was_retrieved():
    # It must not invent a recommendation to fill the space.
    note = highlight_note([])
    assert "isn't published online" in note
    assert "Diners keep mentioning" not in note


def test_note_travels_with_the_card():
    card = card_for_restaurant(
        "Osteria", "italian",
        [{"snippet": "Rigatoni & Broccoli $24 Lasagna $26"}],
        perk="Weekend Family Feast",
    )
    assert "Rigatoni & Broccoli" in card["note"]
    assert "Weekend Family Feast" in card["note"]


def test_card_escapes_markup_in_names():
    svg = _svg(build_menu_card('Bob & "Sons" <script>', "italian", []))
    assert "<script>" not in svg
    assert "&amp;" in svg
