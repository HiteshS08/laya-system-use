from jev_ultrafast.instructions import instruction
from jev_ultrafast.program import Subgoal
from jev_ultrafast.tactics import Progress, next_step

URL = "https://w.test/wiki/Main_Page"
T = "https://w.test/w/index.php?search={q}"


def els(*specs):
    out = []
    for i, (label, role, ops, *extra) in enumerate(specs, 1):
        out.append({"index": str(i), "label": label, "role": role, "operations": list(ops),
                    "href": extra[0] if extra else "", "landmark": "main"})
    return out


PAGE = {"url": URL, "title": "Main Page", "text": "", "headings": []}
SEARCH = els(("Search Wikipedia", "searchbox", ["TYPE_TEXT", "CLICK"]), ("Go", "button", ["CLICK"]))


def test_instruction_templates():
    assert instruction("OPEN", "Issues tab") == "Click the Issues tab."
    assert instruction("FILL", "Where to?", "London") == 'Type "London" into the Where to? field.'
    assert instruction("SELECT", "Size", "M") == 'Select "M" in the Size dropdown.'
    assert instruction("RESULT", "Alan Turing") == "Click the search result for Alan Turing."
    assert instruction("SEARCH", value="Ada") == 'Type "Ada" into the search box.'
    assert instruction("LINK", "Charles Babbage") == "Click the link to Charles Babbage."
    assert instruction("DO", "Buy milk") == "Buy milk"


def test_find_clicks_an_exact_link_first():
    elements = els(("Charles Babbage", "link", ["CLICK"], URL + "x"), *[(e["label"], e["role"], e["operations"])
                                                                          for e in SEARCH])
    step = next_step(Subgoal("FIND", "Charles Babbage"), PAGE, elements, Progress(), T)
    assert (step.operation, step.index) == ("CLICK", "1")


def test_find_prefers_the_search_template_then_the_search_box():
    step = next_step(Subgoal("FIND", "Ada Lovelace"), PAGE, SEARCH, Progress(), T)
    assert step.operation == "GOTO" and step.target_text == "https://w.test/w/index.php?search=Ada+Lovelace"
    assert step.templates == (T,)
    step = next_step(Subgoal("FIND", "Ada Lovelace"), PAGE, SEARCH, Progress())
    assert (step.operation, step.value, step.index, step.purpose) == ("TYPE_TEXT", "Ada Lovelace", "1", "search")


def test_after_a_template_search_the_result_is_clicked_not_retyped():
    step = next_step(Subgoal("FIND", "Ada Lovelace"), PAGE, SEARCH, Progress(searched=True), T)
    assert (step.operation, step.purpose) == ("CLICK", "result")


def test_find_submits_after_typing_then_picks_the_result():
    assert next_step(Subgoal("FIND", "Ada"), PAGE, SEARCH, Progress(typed=True)).operation == "SUBMIT"
    step = next_step(Subgoal("FIND", "Ada"), PAGE, SEARCH, Progress(typed=True, submitted=True))
    assert (step.operation, step.target_text, step.purpose) == ("CLICK", "Ada", "result")


def test_find_opens_a_collapsed_search_control_first():
    elements = els(("Search", "button", ["CLICK"]), ("Home", "link", ["CLICK"], URL))
    step = next_step(Subgoal("FIND", "Ada"), PAGE, elements, Progress())
    assert (step.operation, step.index, step.purpose) == ("CLICK", "1", "open_search")
    assert next_step(Subgoal("FIND", "Ada"), PAGE, elements, Progress(opened_search=True)).purpose == "result"


def test_open_carries_the_ordinal():
    step = next_step(Subgoal("OPEN", "comments", ordinal=2), PAGE, [], Progress())
    assert (step.operation, step.target_text, step.ordinal) == ("CLICK", "comments", 2)


def test_jump_clicks_the_same_document_fragment_link_else_scrolls():
    elements = els(("References", "link", ["CLICK"], URL + "#References"),
                   ("References", "link", ["CLICK"], "https://other.test/#References"))
    step = next_step(Subgoal("JUMP", "References"), PAGE, elements, Progress())
    assert (step.operation, step.index) == ("CLICK", "1")
    step = next_step(Subgoal("JUMP", "References"), PAGE, elements[1:], Progress())
    assert (step.operation, step.target_text) == ("SCROLL_TO_TEXT", "References")


def test_fill_types_then_picks_the_matching_suggestion():
    field = els(("Where to?", "combobox", ["TYPE_TEXT", "CLICK"]))
    sub = Subgoal("FILL", "Where to?", "London")
    assert next_step(sub, PAGE, field, Progress()).operation == "TYPE_TEXT"
    options = field + els(("Paris", "option", ["CLICK"]), ("London, United Kingdom", "option", ["CLICK"]))[1:]
    options[-1]["index"] = "3"
    step = next_step(sub, PAGE, options, Progress(typed=True))
    assert (step.operation, step.index, step.purpose) == ("CLICK", "3", "pick")
    assert next_step(sub, PAGE, field, Progress(typed=True)) is None
    assert next_step(sub, PAGE, options, Progress(typed=True, picked=True)) is None


def test_select_prefers_a_visible_option_then_a_native_select_then_opens_the_field():
    sub = Subgoal("SELECT", "Trip type", "One way")
    option = els(("One way", "option", ["CLICK"]))
    assert next_step(sub, PAGE, option, Progress()).purpose == "pick"
    native = els(("Trip type", "combobox", ["SELECT"]))
    assert next_step(sub, PAGE, native, Progress()).operation == "SELECT"
    custom = els(("Trip type", "combobox", ["CLICK"]))
    step = next_step(sub, PAGE, custom, Progress())
    assert (step.operation, step.target_text, step.purpose) == ("CLICK", "Trip type", "open")
    assert next_step(sub, PAGE, custom, Progress(opened=True)) is None


def test_scroll_submit_click_and_do():
    assert next_step(Subgoal("SCROLL", "External links"), PAGE, [], Progress()).operation == "SCROLL_TO_TEXT"
    assert next_step(Subgoal("SUBMIT", ""), PAGE, [], Progress()).operation == "SUBMIT"
    assert next_step(Subgoal("SUBMIT", "Search"), PAGE, [], Progress()).operation == "CLICK"
    assert next_step(Subgoal("CLICK", "One way"), PAGE, [], Progress()).operation == "CLICK"
    assert next_step(Subgoal("DO", "Buy milk"), PAGE, [], Progress()).operation == "DO"


def test_find_clicks_a_search_button_when_enter_did_nothing():
    elements = els(("Search", "searchbox", ["TYPE_TEXT"]), ("Search", "button", ["CLICK"]))
    step = next_step(Subgoal("FIND", "Ada"), PAGE, elements, Progress(typed=True, misses=1))
    assert (step.operation, step.index, step.purpose) == ("CLICK", "2", "submit")


def test_jump_sets_the_fragment_of_a_matching_heading_when_no_link_exists():
    page = {**PAGE, "headings": [{"text": "References", "level": 2, "id": "References", "in_viewport": False}]}
    step = next_step(Subgoal("JUMP", "References"), page, [], Progress())
    assert (step.operation, step.target_text) == ("FRAGMENT", "References")
    assert next_step(Subgoal("JUMP", "References"), page, [], Progress(misses=1)).operation == "SCROLL_TO_TEXT"
