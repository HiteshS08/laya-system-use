import json

import pytest
from m2w_fixtures import ALL, HTML, make_step, make_task

from jev_ultrafast.candidates import Candidate
from training import mind2web as m2w


@pytest.mark.parametrize(
    "tag,attrs,role",
    [("a", {}, "link"), ("input", {"type": "checkbox"}, "checkbox"), ("input", {"type": "submit"}, "button"),
     ("input", {"type": "search"}, "searchbox"), ("input", {}, "textbox"), ("select", {}, "combobox"),
     ("div", {"role": "tab"}, "tab"), ("div", {}, "div")],
)
def test_role_for(tag, attrs, role):
    assert m2w.role_for(tag, attrs) == role


def test_ops_for():
    assert m2w.ops_for("select", "combobox") == {"SELECT"}
    assert m2w.ops_for("input", "textbox") == {"TYPE_TEXT", "CLICK"}
    assert m2w.ops_for("a", "link") == {"CLICK"}


def test_is_interactive():
    assert m2w.is_interactive("a", {}) and m2w.is_interactive("div", {"role": "button"})
    assert not m2w.is_interactive("div", {})


def test_label_for_prefers_aria_then_text_then_placeholder():
    index = m2w.node_index(HTML)
    assert m2w.label_for(index["10"], {}) == "NFL Scores"
    assert m2w.label_for(index["10"], {"aria_label": "Scores nav"}) == "Scores nav"
    assert m2w.label_for(index["20"], {"placeholder": "Find"}) == "Find"
    assert m2w.label_for(None, {"title": "T"}) == "T"


def test_parse_step_pool_is_interactive_only_in_document_order():
    parsed = m2w.parse_step(make_step())
    assert [c.id for c in parsed.pool] == ["10", "20", "30"]
    assert parsed.gold.id == "10" and parsed.gold_interactive and parsed.op == "CLICK"


def test_pool_keeps_document_order_when_gold_is_not_first():
    # Positives are parsed before negatives, so only the sort by page position restores 10, 20, 30.
    parsed = m2w.parse_step(make_step("CLICK", "30"))
    assert [c.id for c in parsed.pool] == ["10", "20", "30"]
    assert parsed.gold.id == "30"


def test_gold_is_first_interactive_positive():
    step = make_step("CLICK", "30")
    step["pos_candidates"] = [ALL["40"], ALL["30"]]  # a non-interactive positive comes first
    parsed = m2w.parse_step(step)
    assert parsed.gold.id == "30" and parsed.gold_interactive


def test_node_index_survives_deeply_nested_pages():
    depth = 600
    html = ("<html>" + "".join(f'<div backend_node_id="{i}">' for i in range(1, depth + 1))
            + f'<text backend_node_id="{depth + 1}">Deep label</text>' + "</div>" * depth + "</html>")
    index = m2w.node_index(html)
    assert str(depth) in index
    assert m2w.label_for(index[str(depth)], {}) == "Deep label"


def test_parse_step_type_operation_and_value():
    parsed = m2w.parse_step(make_step("TYPE", "20", "abc"))
    assert (parsed.op, parsed.value) == ("TYPE_TEXT", "abc")


def test_unknown_operation_fails_loudly():
    with pytest.raises(ValueError, match="Unknown Mind2Web operation"):
        m2w.parse_step(make_step("HOVER"))


def test_task_rows_builds_questions_gold_and_history():
    rows = list(m2w.task_rows(make_task(make_step("CLICK", "10"), make_step("TYPE", "20", "abc"))))
    first, second = rows
    assert list(first["questions"]) == ["operation", "click_target"]
    assert first["gold"]["operation"]["probabilities"] == {"CLICK": 1.0, "TYPE_TEXT": 0.0}
    assert first["gold"]["click_target"]["probabilities"] == {"10": 1.0, "20": 0.0, "30": 0.0}
    assert first["drop_reason"] is None and first["gold_in_shortlist"]
    assert second["state"]["recent_actions"] == ["CLICK NFL Scores"]
    assert second["gold_id"] == "20" and second["sole"] == {"TYPE_TEXT": "20"}
    assert list(second["gold"]) == ["operation"]  # a single candidate needs no target question


def test_non_interactive_gold_is_dropped_but_still_feeds_history():
    rows = list(m2w.task_rows(make_task(make_step("CLICK", "40"), make_step("CLICK", "10"))))
    assert rows[0]["drop_reason"] == "gold_not_interactive" and not rows[0]["valid_gold"]
    assert rows[1]["state"]["recent_actions"] == ["CLICK Footer"]


def test_operation_not_allowed_on_gold_element():
    (row,) = m2w.task_rows(make_task(make_step("TYPE", "30", "x")))
    assert row["drop_reason"] == "op_not_allowed"


def test_gold_outside_shortlist_is_flagged_but_valid():
    (row,) = m2w.task_rows(make_task(make_step("CLICK", "30")), k=1)
    assert row["valid_gold"] and not row["gold_in_shortlist"]
    assert row["drop_reason"] == "gold_not_shortlisted"


def test_no_positive_candidates_is_dropped_as_no_gold():
    step = make_step()
    step["pos_candidates"] = []
    (row,) = m2w.task_rows(make_task(step))
    assert row["drop_reason"] == "no_gold" and not row["valid_gold"] and row["gold"] == {}


def test_single_operation_single_candidate_is_trivial():
    step = make_step("CLICK", "10")
    step["neg_candidates"] = []  # one link and nothing else: no question to ask
    (row,) = m2w.task_rows(make_task(step))
    assert row["questions"] == {} and row["gold"] == {}
    assert row["valid_gold"] and row["gold_in_shortlist"] and row["drop_reason"] == "trivial"


def test_summarize():
    rows = list(m2w.task_rows(make_task(make_step("CLICK", "10"), make_step("TYPE", "20", "abc"))))
    s = m2w.summarize(rows)
    assert (s["steps"], s["valid_gold"], s["usable_for_training"]) == (2, 2, 2)
    # Step 1's goal word "Find" matches the short "Find" textbox, which the length-normalised, role-aware ranker
    # puts above the gold "NFL Scores" link, so only step 2 has its gold on top.
    assert s["recall_at_k"] == 1.0 and s["top1_given_op"] == 0.5
    assert s["gold_ops"] == {"CLICK": 1, "TYPE_TEXT": 1}


def test_iter_tasks_reads_every_file(tmp_path):
    for i in range(2):
        (tmp_path / f"s{i}.json").write_text(json.dumps([make_task(make_step())]))
    assert len(list(m2w.iter_tasks(sorted(tmp_path.glob("*.json"))))) == 2


def test_candidate_from_input_carries_editable_ops_and_value():
    cand_, ok = m2w.to_candidate(json.loads(json.dumps(ALL["20"])) | {"attributes": json.dumps(
        {"type": "text", "input_value": "hi", "placeholder": "Find"})}, m2w.node_index(HTML))
    assert ok and cand_ == Candidate("20", "Find", "textbox", "hi", frozenset({"TYPE_TEXT", "CLICK"}))


NESTED = """<html backend_node_id="1">
<a backend_node_id="10"><span backend_node_id="11"><text backend_node_id="12">Scores</text></span></a>
<label backend_node_id="20"><text backend_node_id="21">Pick a date</text>
  <span backend_node_id="22"><a backend_node_id="23">Deep</a></span>
  <button backend_node_id="25">Shallow</button>
</label>
<div backend_node_id="40"><text backend_node_id="41">Footer</text></div>
<a backend_node_id="60"><label backend_node_id="61"><input backend_node_id="62"/></label></a>
</html>"""


def nested_step(gold_id, gold_tag, others=(), op="CLICK"):
    def raw(bid, tag):
        return {"tag": tag, "backend_node_id": bid, "attributes": json.dumps({"backend_node_id": bid})}
    return {"action_uid": "u", "cleaned_html": NESTED, "operation": {"op": op, "original_op": op, "value": ""},
            "pos_candidates": [raw(gold_id, gold_tag)], "neg_candidates": [raw(b, t) for b, t in others]}


def test_non_interactive_gold_maps_to_its_interactive_ancestor():
    parsed = m2w.parse_step(nested_step("12", "text", others=[("10", "a"), ("40", "div")]))
    assert parsed.gold.id == "10" and parsed.gold.label == "Scores" and parsed.gold_interactive
    assert [c.id for c in parsed.pool] == ["10"]


def test_shallowest_interactive_descendant_wins_even_when_a_deeper_one_comes_first_in_the_page():
    # link 23 comes first in the page but sits a level deeper than button 25; the shallowest descendant is chosen.
    parsed = m2w.parse_step(nested_step("20", "label"))
    assert parsed.gold.id == "25" and parsed.gold_interactive


def test_interactive_descendant_is_preferred_over_an_interactive_ancestor():
    parsed = m2w.parse_step(nested_step("61", "label", others=[("60", "a")]))
    assert parsed.gold.id == "62" and parsed.gold_interactive


def test_mapped_element_missing_from_the_candidates_still_enters_the_pool_in_page_order():
    parsed = m2w.parse_step(nested_step("20", "label", others=[("10", "a")]))
    assert [c.id for c in parsed.pool] == ["10", "25"]
    assert parsed.gold == Candidate("25", "Shallow", "button", "", frozenset({"CLICK"}))


def test_gold_with_no_interactive_relative_stays_dropped():
    parsed = m2w.parse_step(nested_step("40", "div", others=[("10", "a")]))
    assert parsed.gold.id == "40" and not parsed.gold_interactive


def test_reclaimed_step_is_usable_and_history_uses_the_element_that_was_clicked():
    task = make_task(nested_step("12", "text", others=[("10", "a"), ("25", "button")]),
                     nested_step("10", "a", others=[("25", "button")]))
    first, second = m2w.task_rows(task)
    assert first["valid_gold"] and first["gold_id"] == "10" and first["drop_reason"] is None
    assert second["state"]["recent_actions"] == ["CLICK Scores"]


UNNAMED = """<html backend_node_id="1">
<button backend_node_id="70"></button>
<input backend_node_id="71"/>
<a backend_node_id="72"><text backend_node_id="73">Home</text></a>
</html>"""


def unnamed_raw(bid, tag, **attrs):
    return {"tag": tag, "backend_node_id": bid, "attributes": json.dumps({"backend_node_id": bid, **attrs})}


def test_element_without_any_name_is_labelled_with_its_role_like_the_live_snapshot():
    index = m2w.node_index(UNNAMED)
    button, _ = m2w.to_candidate(unnamed_raw("70", "button"), index)
    field, _ = m2w.to_candidate(unnamed_raw("71", "input", type="text"), index)
    assert (button.label, field.label) == ("button", "textbox")


def test_a_real_name_still_wins_over_the_role_fallback():
    index = m2w.node_index(UNNAMED)
    link, _ = m2w.to_candidate(unnamed_raw("72", "a"), index)
    titled, _ = m2w.to_candidate(unnamed_raw("70", "button", title="Close"), index)
    assert (link.label, titled.label) == ("Home", "Close")


def test_history_renders_the_role_for_an_unnamed_clicked_element():
    def step(gold_id, tag):
        op = {"op": "CLICK", "original_op": "CLICK", "value": ""}
        return {"action_uid": "u", "cleaned_html": UNNAMED, "operation": op,
                "pos_candidates": [unnamed_raw(gold_id, tag)], "neg_candidates": [unnamed_raw("72", "a")]}
    first, second = m2w.task_rows(make_task(step("70", "button"), step("72", "a")))
    assert second["state"]["recent_actions"] == ["CLICK button"]
    assert first["gold"]["click_target"]["probabilities"]["70"] == 1.0
    assert first["questions"]["click_target"]["criteria"]["70"] == "button (button)"
