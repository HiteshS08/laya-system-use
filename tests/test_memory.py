from jev_ultrafast.memory import StepMemory

P = "https://en.wikipedia.org/wiki/Main_Page"


def entry(label, op="CLICK", before=P, after=P, changed=True, text=None):
    return {"operation": op, "kind": "click", "action": label, "text": text,
            "url_before": before, "url": after, "page_changed": changed}


def test_sync_ingests_only_new_entries_and_classifies_outcomes():
    history = [entry("View source", after=P + "?action=edit"), entry("Read", changed=False)]
    m = StepMemory()
    m.sync(history)
    m.sync(history)
    assert [a.outcome for a in m.attempts] == ["url_changed", "no_change"]
    history.append(entry("Talk", changed=True))
    m.sync(history)
    assert [a.outcome for a in m.attempts] == ["url_changed", "no_change", "page_changed"]


def test_no_change_click_is_excluded_on_that_page_only():
    m = StepMemory()
    m.sync([entry("Read", changed=False)])
    assert m.excluded(P) == frozenset({("CLICK", "read")})
    assert m.excluded("https://en.wikipedia.org/wiki/Other") == frozenset()
    assert m.excluded(P + "#History") == frozenset({("CLICK", "read")})
    assert m.failed(P) == ["CLICK Read (no effect)"]


def test_typing_without_page_change_is_not_a_failure():
    m = StepMemory()
    m.sync([entry("Search Wikipedia", op="TYPE_TEXT", changed=False, text="Ada")])
    assert m.excluded(P) == frozenset() and not m.last_failed()


def test_action_repeated_twice_is_excluded_even_when_it_changed_the_page():
    m = StepMemory()
    m.sync([entry("Toggle References subsection"), entry("Toggle References subsection")])
    assert ("CLICK", "toggle references subsection") in m.excluded(P)
    assert "CLICK Toggle References subsection (repeated)" in m.failed(P)


def test_streak_counts_trailing_actions_without_url_change():
    m = StepMemory()
    m.sync([entry("A", after=P + "#x"), entry("B"), entry("C", changed=False), entry("D")])
    assert m.streak_without_url_change() == 3
    assert m.last_failed() is False
    m.sync([entry("A", after=P + "#x"), entry("B"), entry("C", changed=False), entry("D"),
            entry("E", changed=False)])
    assert m.last_failed() is True


def test_missing_url_before_falls_back_to_url():
    m = StepMemory()
    m.sync([{"operation": "CLICK", "action": "Go", "url": P, "page_changed": False}])
    assert m.attempts[0].url == P and m.attempts[0].outcome == "no_change"
