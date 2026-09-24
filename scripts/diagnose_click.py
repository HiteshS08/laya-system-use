"""Throwaway-Chrome diagnosis: why does clicking GitHub's Issues tab sometimes not navigate?

Usage: uv run --env-file .env python scripts/diagnose_click.py [runs]
"""

import json
import sys
import time

from jev_ultrafast.browser import Browser

REPO = "https://github.com/browser-use/browser-use"
HIT = """(node => {
  const e=window.__jevFast.nodes.get(node); const r=e.getBoundingClientRect();
  const x=r.x+r.width/2, y=r.y+r.height/2, hit=document.elementFromPoint(x,y);
  return {x, y, contains: e.contains(hit), hit: hit ? hit.outerHTML.slice(0,160) : null,
          href: e.href, target: e.outerHTML.slice(0,160)};
})"""


def one_run() -> dict:
    browser = Browser(REPO)
    try:
        page = browser.observe(screenshot=False)
        issues = next(a for a in page["actions"] if a["kind"] == "click" and a["label"].startswith("Issues"))
        hit = browser.evaluate(f"{HIT}({issues['node']})")
        browser.act(issues, page)
        started, timeline = time.perf_counter(), []
        for delay in (0.05, 0.25, 0.5, 1.0, 2.0, 3.0):
            time.sleep(max(0.0, delay - (time.perf_counter() - started)))
            timeline.append((delay, browser.evaluate("location.href")))
        return {"hit_test": hit, "timeline": timeline}
    finally:
        browser.close()


if __name__ == "__main__":
    for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5):
        print(json.dumps({"run": i + 1, **one_run()}, indent=1))
