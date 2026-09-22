from jev_ultrafast.candidates import Candidate
from jev_ultrafast.shortlister import rank_candidates, shortlist


def c(i, label):
    return Candidate(str(i), label, "link")


def ids(items):
    return [x.id for x in items]


def test_returns_everything_when_at_most_k():
    items = [c(1, "a"), c(2, "b")]
    assert ids(shortlist("goal", [], items, k=5)) == ["1", "2"]


def test_keeps_goal_matching_candidate_among_distractors():
    items = [c(i, f"Footer link {i}") for i in range(30)] + [c(99, "NFL Scores")]
    assert "99" in ids(shortlist("Find the latest NFL scores", [], items, k=5))


def test_result_keeps_page_order():
    items = [c(1, "nfl scores"), c(2, "unrelated"), c(3, "nfl news")]
    assert ids(shortlist("nfl", [], items, k=2)) == ["1", "3"]


def test_result_is_page_order_even_when_best_match_comes_later():
    items = [c(1, "nfl"), c(2, "unrelated"), c(3, "nfl scores")]
    assert ids(rank_candidates("nfl scores", [], items))[:2] == ["3", "1"]
    assert ids(shortlist("nfl scores", [], items, k=2)) == ["1", "3"]


def test_rarer_word_outweighs_common_word():
    items = [c(i, f"Search option {i}") for i in range(5)] + [c("z", "Zurich")]
    assert ids(shortlist("search zurich", [], items, k=1)) == ["z"]


def test_ties_break_by_page_position():
    items = [c(1, "alpha"), c(2, "alpha"), c(3, "beta")]
    assert ids(shortlist("alpha", [], items, k=1)) == ["1"]


def test_accents_and_plurals_are_folded():
    items = [c(1, "Godel"), c(2, "other"), c(3, "theorems list")]
    assert ids(rank_candidates("Gödel theorem", [], items))[:2] == ["1", "3"]


def test_history_words_count_but_operation_names_do_not():
    items = [c(1, "click here"), c(2, "Zurich airport")]
    assert ids(rank_candidates("book", ["CLICK Search", "TYPE_TEXT From = Zurich"], items))[0] == "2"


def test_inputs_are_not_mutated():
    items = [c(2, "b"), c(1, "a")]
    rank_candidates("a", [], items)
    assert ids(items) == ["2", "1"]


def test_short_label_outranks_long_label_with_the_same_match():
    long_label = "Zurich airport parking shuttle bus hotel rental"
    items = [c(1, long_label), c(2, "Zurich")]
    assert ids(rank_candidates("zurich", [], items))[0] == "2"


def field(i, label, role):
    return Candidate(str(i), label, role)


def test_form_fields_and_buttons_outrank_links_when_nothing_matches_the_goal():
    items = [field(1, "Privacy", "link"), field(2, "Go", "button"), field(3, "Where to?", "textbox")]
    assert ids(rank_candidates("zzz", [], items)) == ["3", "2", "1"]


def test_page_position_outweighs_the_role_bonus_of_a_later_button():
    items = [field(1, "Privacy", "link"), field(2, "Go", "button")]
    assert ids(rank_candidates("zzz", [], items)) == ["1", "2"]


def test_goal_match_outweighs_the_role_and_position_priors():
    items = [field(1, "Where to?", "textbox"), field(2, "Privacy", "link"), c(3, "NFL Scores")]
    assert ids(rank_candidates("nfl scores", [], items))[0] == "3"


def test_no_candidates_rank_to_nothing():
    assert rank_candidates("goal", [], []) == []
    assert shortlist("goal", [], []) == ()
