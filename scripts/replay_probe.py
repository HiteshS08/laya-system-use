"""Re-ask six recorded live failures in the actor's request shape; report which element the actor picks.

Usage: uv run --env-file .env python scripts/replay_probe.py [run_dir]
"""

import json
import re
import sys
from pathlib import Path

from jev_ultrafast.actor import actor_request
from jev_ultrafast.candidates import Candidate
from jev_ultrafast.planner import PlanStep
from jev_ultrafast.policy import laya_predict

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/live/20260923T175703Z")
_OPTION = re.compile(r"^(.*) \(([^,()]+)(?:, =(.*))?\)$")
# (task, decision index, question id in the recording, step, expected label prefix)
CASES = [
    ("wiki_search_ada", 0, "type_text_target", PlanStep("TYPE_TEXT", "Search Wikipedia", "Ada Lovelace",
     'Type "Ada Lovelace" into the Search Wikipedia box.'), "Search Wikipedia"),
    ("wiki_featured", 0, "click_target", PlanStep("CLICK", "Mary Mallon", "",
     "Click the Mary Mallon link in today's featured article."), "Mary Mallon"),
    ("mallon_to_typhoid", 0, "click_target", PlanStep("CLICK", "Mary Mallon", "",
     "Click the Mary Mallon link in today's featured article."), "Mary Mallon"),
    ("hn_comments", 0, "click_target", PlanStep("CLICK", "48 comments", "",
     "Click the 48 comments link under the top story."), "48 comments"),
    ("ada_references", 0, "click_target", PlanStep("CLICK", "8 References", "",
     "Click the 8 References link in the contents."), "8 References"),
    ("flights", 1, "type_text_target", PlanStep("TYPE_TEXT", "Where to?", "London",
     'Type "London" into the Where to? field.'), "Where to?"),
]


def parse_option(text: str) -> tuple[str, str, str]:
    match = _OPTION.match(text)
    if not match:
        return text, "", ""
    return match.group(1), match.group(2), match.group(3) or ""


def _candidates(record: dict, index: int, question: str) -> list[Candidate]:
    questions = record["decisions"][index]["request"]["questions"]
    q = questions.get(question) or questions["click_target"]
    criteria = q["criteria"]
    return [Candidate(i, *parse_option(text)[:2], parse_option(text)[2]) for i, text in criteria.items()]


def main() -> None:
    passed = 0
    for task, index, question, step, expected in CASES:
        record = json.loads((RUN / f"{task}.json").read_text())
        candidates = _candidates(record, index, question)
        if len(candidates) == 1:
            label, prob = candidates[0].label, 1.0
        else:
            state, questions = actor_request(step, candidates, [])
            answer = laya_predict(state, questions)["answers"][next(iter(questions))]
            label = next(c.label for c in candidates if c.id == answer["choice"])
            prob = answer["probabilities"][answer["choice"]]
        ok = label.startswith(expected)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} {task:18} picked={label[:40]!r} p={prob:.2f} expected={expected!r}")
    print(f"{passed}/{len(CASES)} probe decisions correct")


if __name__ == "__main__":
    main()
