"""Save each live task's start page (as the agent observes it) for offline planner benchmarks."""

import json
import sys
from pathlib import Path

from evals.live_tasks import TASKS
from jev_ultrafast.browser import Browser

OUT = Path("artifacts/pages")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or list(TASKS):
        browser = Browser(TASKS[name].url)
        try:
            page = browser.observe(screenshot=False)
        finally:
            browser.close()
        keep = {k: page[k] for k in ("url", "title", "text", "outline", "actions")}
        (OUT / f"{name}.json").write_text(json.dumps({"task": name, "goal": TASKS[name].goal, "page": keep}))
        print(name, len(page["actions"]), "actions")
