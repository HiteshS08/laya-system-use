from scripts.replay_probe import parse_option


def test_parse_option_round_trips_render_option():
    assert parse_option("Mary Mallon (link)") == ("Mary Mallon", "link", "")
    assert parse_option("Where from? (combobox, =Chennai)") == ("Where from?", "combobox", "Chennai")
    assert parse_option("Change ticket type. Round trip (combobox, =Round trip)") == (
        "Change ticket type. Round trip", "combobox", "Round trip")
    assert parse_option("plain label") == ("plain label", "", "")
