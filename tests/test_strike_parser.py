from kalshi.strike_parser import ensure_strike_metadata


def test_between_range_from_subtitle():
    market = {
        "ticker": "TEST",
        "subtitle": "Between 65°F and 75°F",
        "strike_type": None,
        "floor_strike": None,
        "cap_strike": None,
    }

    enriched = ensure_strike_metadata(market)

    assert enriched["strike_type"] == "between"
    assert enriched["floor_strike"] == 65.0
    assert enriched["cap_strike"] == 75.0


def test_greater_from_rules():
    market = {
        "ticker": "TEST2",
        "rules_primary": "Resolves YES if max temp is greater than 80F.",
        "strike_type": None,
        "floor_strike": None,
        "cap_strike": None,
    }

    enriched = ensure_strike_metadata(market)

    assert enriched["strike_type"] == "greater"
    assert enriched["floor_strike"] == 80.0
    assert enriched.get("cap_strike") is None


def test_less_from_bracket_shape():
    market = {
        "ticker": "TEST3",
        "strike_type": None,
        "floor_strike": None,
        "cap_strike": 55.0,
    }

    enriched = ensure_strike_metadata(market)

    assert enriched["strike_type"] == "less"
    assert enriched["cap_strike"] == 55.0
