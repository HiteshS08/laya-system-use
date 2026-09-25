"""Replay scripted Pilot scenarios with fakes and count planner calls per distinct URL.

Usage: uv run python scripts/count_plan_calls.py

No model, browser or network. The planner model is a scripted oracle behind `planner.plan`'s `complete` hook, the
actor ranks labels in a fixed order, and a tiny page world applies each decision. Because only public entry points
are used, the same scenarios run against another checkout: `PYTHONPATH=<checkout> uv run python <this file>`.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from jev_ultrafast import planner
from jev_ultrafast.pilot import Pilot
from jev_ultrafast.resolver import normalize
from jev_ultrafast.tools import is_allowed_goto

MAX_TICKS = 12  # scripts/live_eval.py's per-task step cap
WAIT = {"id": "wait", "kind": "wait", "label": "Wait for the page to update"}


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str
    actions: tuple[dict, ...]
    scrolled_text: str = ""  # the visible text once a SCROLL_TO_TEXT on this page has found its target


@dataclass(frozen=True)
class Visit:
    """What the scripted planner model plans on one URL."""

    steps: tuple[dict, ...] = ()
    done_when: str = ""
    evidence: str = ""  # answered as done once visible_text, focus_text or the title shows it
    after_failure: tuple[dict, ...] = ()  # planned instead once failed_attempts is non-empty


@dataclass(frozen=True)
class Scenario:
    name: str
    goal: str
    start: str
    pages: Mapping[str, Page]
    visits: Mapping[str, Visit]
    effects: Mapping[str, str]  # "url|label" -> url reached; missing means the action has no visible effect
    actor_order: tuple[str, ...] = ()  # labels the fake actor prefers, best first


@dataclass(frozen=True)
class Result:
    name: str
    plan_calls: int
    pick_calls: int
    urls: int
    outcome: str
    actions: int


def act(n: int, kind: str, label: str, role: str, **extra: str) -> dict:
    return {"id": f"e{n}", "kind": kind, "label": label, "role": role, "node": n, "value": "", **extra}


def step(operation: str, target: str, instruction: str, value: str = "") -> dict:
    return {"operation": operation, "target_text": target, "value": value, "instruction": instruction}


def _remaining(visit: Visit, view: Mapping) -> list[dict]:
    steps = visit.after_failure if view["failed_attempts"] and visit.after_failure else visit.steps
    left = [s for s in steps if s["instruction"] not in view["completed_steps"]]
    # Nothing left but no evidence either: the live planner re-planned its last step (Gate A+B re-run report).
    return left or list(steps[-1:])


def oracle(scenario: Scenario) -> Callable:
    def complete(_system: str, user: str, **_kw) -> tuple[dict, dict]:
        view = json.loads(user.split("\n", 1)[0])
        visit = scenario.visits[view["url"]]
        shown = normalize(" ".join([view["visible_text"], view.get("focus_text", ""), view["title"]]))
        if visit.evidence and normalize(visit.evidence) in shown:
            return {"status": "done", "evidence": visit.evidence, "steps": []}, {}
        steps = _remaining(visit, view)
        if not steps:
            return {"status": "blocked", "evidence": "", "steps": []}, {}
        return {"status": "continue", "evidence": "", "done_when": visit.done_when, "steps": steps}, {}

    return complete


def actor(order: Sequence[str]) -> Callable:
    def predict(_state: dict, questions: dict) -> dict:
        (name, question), = questions.items()
        ids = list(question["criteria"])

        def rank(i: str) -> int:
            text = question["criteria"][i]
            return next((n for n, label in enumerate(order) if text == label or text.startswith(label + " (")),
                        len(order))

        ranked = sorted(ids, key=lambda i: (rank(i), ids.index(i)))
        weights = [0.6, 0.4] if len(ids) == 2 else [0.55, 0.35] + [0.1 / (len(ids) - 2)] * (len(ids) - 2)
        probabilities = dict(zip(ranked, weights, strict=True))
        return {"answers": {name: {"choice": ranked[0], "confidence": weights[0], "probabilities": probabilities}},
                "model": "fake-actor"}

    return predict


class World:
    """The browser stand-in: current URL, which pages were scrolled, and each decision's recorded effect."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario, self.url, self.scrolled = scenario, scenario.start, set()

    def page(self) -> dict:
        p = self.scenario.pages[self.url]
        text = p.scrolled_text if self.url in self.scrolled else p.text
        return {"url": p.url, "title": p.title, "text": text, "outline": "", "actions": [*p.actions, WAIT]}

    def execute(self, decision: Mapping, page: Mapping) -> dict:
        before = self.url
        entry = {"url_before": before, "operation": decision["operation"], "text": None}
        if decision.get("tool"):
            ok, changed = self._tool(decision["tool"])
            entry.update(action=decision["tool"]["arg"], kind="tool", tool_ok=ok, page_changed=changed)
        else:
            action = next(a for a in page["actions"] if a["id"] == decision["choice"])
            entry.update(action=action["label"], kind=action["kind"], role=action.get("role"))
            if action["kind"] == "fill":
                entry.update(text=decision.get("value"), page_changed=True)
            else:
                self.url = self.scenario.effects.get(f"{before}|{action['label']}", before)
                entry["page_changed"] = self.url != before
        return {**entry, "url": self.url}

    def _tool(self, tool: Mapping) -> tuple[bool, bool]:
        if tool["operation"] == "GOTO":
            target = self.scenario.effects.get(f"{self.url}|GOTO")
            if not is_allowed_goto(tool["arg"]) or target is None:
                return False, False
            self.url = target
            return True, True
        found = normalize(tool["arg"]) in normalize(self.scenario.pages[self.url].scrolled_text)
        changed = found and self.url not in self.scrolled
        if found:
            self.scrolled.add(self.url)
        return found, changed


def _stuck(history: Sequence[Mapping]) -> bool:
    # agent.py's guard: three actions in a row with no visible effect (waits excepted) block the run.
    last = history[-3:]
    return len(last) == 3 and all(h["page_changed"] is False and h["kind"] != "wait" for h in last)


def run(scenario: Scenario) -> Result:
    plan_urls: list[str] = []
    picks: list[str] = []

    def plan_fn(*args, **kwargs):
        plan_urls.append(args[1]["url"])
        return planner.plan(*args, complete=oracle(scenario), **kwargs)

    def pick_fn(step_, options, goal):
        picks.append(step_.instruction)
        return None

    def fallback(page, goal, history):
        return {"choice": "wait", "operation": "WAIT", "probabilities": {"wait": 1.0}, "confidence": 1.0}

    pilot = Pilot(scenario.goal, plan_fn=plan_fn, pick_fn=pick_fn, predict=actor(scenario.actor_order),
                  fallback=fallback, search_query_fn=lambda goal: None)
    world, history, seen, outcome = World(scenario), [], set(), "step cap"
    for _ in range(MAX_TICKS):
        page = world.page()
        seen.add(page["url"])
        decision = pilot.decide(page, history)
        if decision["choice"] in {"DONE", "BLOCKED"}:
            outcome = decision["choice"]
            break
        history.append(world.execute(decision, page))
        if _stuck(history):
            outcome = "BLOCKED (no-change guard)"
            break
    seen.add(world.url)
    return Result(scenario.name, len(plan_urls), len(picks), len(seen), outcome, len(history))


MAIN = "https://en.wikipedia.org/wiki/Main_Page"
ADA = "https://en.wikipedia.org/wiki/Ada_Lovelace"
MALLON = "https://en.wikipedia.org/wiki/Mary_Mallon"
TYPHOID = "https://en.wikipedia.org/wiki/Typhoid_fever"
TURING = "https://en.wikipedia.org/wiki/Alan_Turing"
FORM = "https://www.google.com/travel/flights"
RESULTS = FORM + "/search?tfs=zrh-lhr"
REPO = "https://github.com/octo/demo"
ISSUES = REPO + "/issues"

WIKI_MAIN = Page(MAIN, "Wikipedia, the free encyclopedia",
                 "Welcome to Wikipedia, the free encyclopedia that anyone can edit.\nFrom today's featured article\n"
                 "Mary Mallon was an Irish-born cook believed to have infected many people.",
                 (act(1, "fill", "Search Wikipedia", "searchbox"), act(2, "click", "Search", "button"),
                  act(3, "click", "Mary Mallon", "link"), act(4, "click", "Today's featured article", "link")))
ADA_TEXT = ("Ada Lovelace\nFrom Wikipedia, the free encyclopedia\nContents\n(Top)\nEarly life\nWork\nReferences\n"
            "Augusta Ada King, Countess of Lovelace, was an English mathematician and writer.")
ADA_PAGE = Page(ADA, "Ada Lovelace - Wikipedia", ADA_TEXT,
                (act(1, "click", "References", "link", href=ADA + "#References"),
                 act(2, "click", "Charles Babbage", "link")))
ADA_REFS = Page(ADA + "#References", "Ada Lovelace - Wikipedia",
                "References\n1. ^ Toole, Betty Alexandra (1998). Ada, the Enchantress of Numbers.", ADA_PAGE.actions)
TURING_PAGE = Page(TURING, "Alan Turing - Wikipedia",
                   "Alan Turing\nFrom Wikipedia, the free encyclopedia\nContents\n(Top)\nEarly life\nCareer\n"
                   "References\nExternal links\nAlan Mathison Turing was an English mathematician.",
                   (act(1, "click", "Bletchley Park", "link"), act(2, "click", "Turing machine", "link"),
                    act(3, "click", "Show", "button")),
                   scrolled_text="References\n" + "^ Hodges, Andrew (1983). Alan Turing: The Enigma. " * 16 +
                   "\nExternal links\nTuring Archive for the History of Computing\nThe Turing Digital Archive")

SCENARIOS = (
    Scenario("wiki_search_ada", "Search Wikipedia for Ada Lovelace and open her article.", MAIN,
             {MAIN: WIKI_MAIN, ADA: ADA_PAGE},
             {MAIN: Visit((step("TYPE_TEXT", "Search Wikipedia", 'Type "Ada Lovelace" into Search Wikipedia.',
                                "Ada Lovelace"),
                           step("CLICK", "Search", "Click the Search button.")), "Ada Lovelace - Wikipedia"),
              ADA: Visit(evidence="Countess of Lovelace")},
             {f"{MAIN}|Search": ADA}),
    Scenario("flights_form", "Find flights from Zürich to London on Monday 12 October.", FORM,
             {FORM: Page(FORM, "Google Flights", "Find cheap flights\nRound trip\n1 passenger\nEconomy",
                         (act(1, "fill", "Where from?", "combobox"), act(2, "fill", "Where to?", "combobox"),
                          act(3, "fill", "Departure", "textbox"), act(4, "click", "Search", "button"),
                          act(5, "click", "Explore", "link"))),
              RESULTS: Page(RESULTS, "Zürich to London | Google Flights",
                            "Top departing flights\nZürich to London\nMon, Oct 12", ())},
             {FORM: Visit((step("TYPE_TEXT", "Where from?", 'Type "Zürich" into Where from?.', "Zürich"),
                           step("TYPE_TEXT", "Where to?", 'Type "London" into Where to?.', "London"),
                           step("TYPE_TEXT", "Departure", 'Type "Mon, Oct 12" into Departure.', "Mon, Oct 12"),
                           step("CLICK", "Search", "Click the Search button.")), "Zürich to London"),
              RESULTS: Visit(evidence="Top departing flights")},
             {f"{FORM}|Search": RESULTS}),
    Scenario("turing_external_links", "Scroll down until the External links section heading is visible.", TURING,
             {TURING: TURING_PAGE},
             {TURING: Visit((step("SCROLL_TO_TEXT", "External links", "Scroll to the External links heading."),),
                            "External links", "Turing Archive for the History of Computing")},
             {}),
    Scenario("mallon_to_typhoid", "Open the Mary Mallon article, then follow its link to Typhoid fever.", MAIN,
             {MAIN: WIKI_MAIN,
              MALLON: Page(MALLON, "Mary Mallon - Wikipedia",
                           "Mary Mallon\nMary Mallon was an Irish-born cook believed to have infected 53 people "
                           "with typhoid fever.", (act(1, "click", "typhoid fever", "link"),
                                                   act(2, "click", "Irish", "link"))),
              TYPHOID: Page(TYPHOID, "Typhoid fever - Wikipedia",
                            "Typhoid fever\nTyphoid fever, also known simply as typhoid, is a disease.", ())},
             {MAIN: Visit((step("CLICK", "Mary Mallon", "Click the Mary Mallon link."),), "Typhoid fever - Wikipedia"),
              MALLON: Visit((step("CLICK", "typhoid fever", "Click the typhoid fever link."),),
                            "Typhoid fever - Wikipedia"),
              TYPHOID: Visit(evidence="also known simply as typhoid")},
             {f"{MAIN}|Mary Mallon": MALLON, f"{MALLON}|typhoid fever": TYPHOID}),
    Scenario("gh_issues_retry", "Open the Issues tab of the octo/demo repository.", REPO,
             {REPO: Page(REPO, "octo/demo: A demo repository",
                         "octo / demo\nCode\nIssues\n12\nPull requests\nA demo repository",
                         (act(1, "click", "Issues", "link"), act(2, "click", "Issues 12", "tab"),
                          act(3, "click", "Code", "tab"), act(4, "click", "Pull requests", "tab"))),
              ISSUES: Page(ISSUES, "Issues · octo/demo", "Issues\nFilters\nOpen 12 Closed 3", ())},
             {REPO: Visit((step("CLICK", "Issues tab", "Click the Issues tab of the repository."),),
                          "Issues · octo/demo",
                          after_failure=(step("CLICK", "Issues 12", "Click the Issues 12 tab."),)),
              ISSUES: Visit(evidence="Open 12 Closed 3")},
             {f"{REPO}|Issues 12": ISSUES}, actor_order=("Issues", "Issues 12")),
    Scenario("ada_references", "Go to the References section of the Ada Lovelace article.", ADA,
             {ADA: ADA_PAGE, ADA + "#References": ADA_REFS},
             {ADA: Visit((step("CLICK", "References", "Click the References link in the contents."),), "References"),
              ADA + "#References": Visit(evidence="Toole, Betty Alexandra")},
             {f"{ADA}|References": ADA + "#References"}),
)


def report(results: Sequence[Result]) -> dict:
    calls, urls = sum(r.plan_calls for r in results), sum(r.urls for r in results)
    return {"scenarios": [r.__dict__ for r in results], "plan_calls": calls, "distinct_urls": urls,
            "calls_per_url": round(calls / urls, 2),
            "mean_scenario_calls_per_url": round(sum(r.plan_calls / r.urls for r in results) / len(results), 2)}


if __name__ == "__main__":
    import jev_ultrafast

    summary = report([run(s) for s in SCENARIOS])
    for r in summary["scenarios"]:
        print(f"{r['name']:24} plan_fn {r['plan_calls']:2}  urls {r['urls']}  pick {r['pick_calls']}  "
              f"actions {r['actions']:2}  {r['outcome']}")
    print(f"code: {jev_ultrafast.__file__}")
    print(json.dumps({k: v for k, v in summary.items() if k != "scenarios"}))
