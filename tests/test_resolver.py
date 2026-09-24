from jev_ultrafast.resolver import normalize, resolve


def el(index, label, aliases=()):
    return {"index": str(index), "label": label, "aliases": list(aliases)}


def test_normalize_folds_accents_case_and_punctuation():
    assert normalize("  Kurt GÖDEL's  theorems! ") == "kurt godel s theorems"
    assert normalize("Where to?") == "where to"


def test_unique_exact_match_wins():
    elements = [el(1, "Where from?"), el(2, "Where to?"), el(3, "Departure")]
    assert resolve("where to", elements) == (elements[1], "resolver")


def test_alias_counts_as_exact():
    elements = [el(1, "Mary Mallon", ["Full article..."]), el(2, "Read")]
    assert resolve("Full article...", elements) == (elements[0], "resolver")


def test_accented_target_matches_plain_label_and_back():
    elements = [el(1, "Kurt Gödel"), el(2, "Zurich")]
    assert resolve("Kurt Godel", elements)[0] is elements[0]
    assert resolve("Zürich", elements)[0] is elements[1]


def test_repeated_exact_label_defers_to_the_actor():
    elements = [el(1, "Edit"), el(2, "Edit"), el(3, "Save")]
    assert resolve("Edit", elements) == (None, None)


def test_unique_word_subset_is_fuzzy():
    elements = [el(1, "Search Wikipedia"), el(2, "Donate")]
    assert resolve("Search", elements) == (elements[0], "resolver_fuzzy")


def test_ambiguous_fuzzy_or_empty_target_defers():
    elements = [el(1, "48 comments"), el(2, "17 comments")]
    assert resolve("comments", elements) == (None, None)
    assert resolve("  ", elements) == (None, None)
    assert resolve("Nothing like it", elements) == (None, None)
