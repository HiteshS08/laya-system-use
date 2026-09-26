"""Site search without per-site code: a page's own OpenSearch description, or a template learned from a search."""

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus, unquote_plus, urlsplit

from .resolver import normalize

log = logging.getLogger("search")
_OPTIONAL = re.compile(r"^\{[^}]+\?\}$")
_PLACEHOLDER = re.compile(r"\{[^}]*\}")

# Fetches the page's own OpenSearch description (same origin; cross-origin fetches fail under CORS and return null).
OPENSEARCH_JS = """(async () => {
  const link=document.querySelector('link[rel="search"][type="application/opensearchdescription+xml"]');
  if (!link) return null;
  const ctrl=new AbortController(), timer=setTimeout(()=>ctrl.abort(),3000);
  try {
    const r=await fetch(link.href,{signal:ctrl.signal,credentials:'omit'});
    return r.ok ? (await r.text()).slice(0,20000) : null;
  } catch (e) { return null; } finally { clearTimeout(timer); }
})()"""


def _host(url: str) -> str:
    return urlsplit(url).hostname or ""


def _drop_optional(template: str) -> str:
    base, sep, query = template.partition("?")
    if not sep:
        return template
    kept = [p for p in query.split("&") if not _OPTIONAL.match(p.partition("=")[2])]
    return f"{base}?{'&'.join(kept)}" if kept else base


def template_from_opensearch(xml_text: str, page_url: str) -> str | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for node in root.iter():
        raw = node.get("template") or ""
        if not node.tag.endswith("Url") or node.get("type") != "text/html" or "{searchTerms}" not in raw:
            continue
        host = _web_host(raw)
        if not host or host != _web_host(page_url):
            return None
        template = _drop_optional(raw.replace("{searchTerms}", "{q}"))
        return template if _PLACEHOLDER.findall(template) == ["{q}"] else None
    return None


_TOKEN_NAME = re.compile(r"sid|session|token|csrf|auth|key|sig", re.IGNORECASE)
_OPAQUE_VALUE = re.compile(r"[A-Za-z0-9_\-]{32,}")
_TRACKING = re.compile(r"utm_[a-z]+|fbclid|gclid|msclkid", re.IGNORECASE)


def _web_host(url: str) -> str:
    """The host of a plain http(s) URL without credentials; "" for anything else."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or "@" in parts.netloc:
        return ""
    return parts.hostname or ""


def _safe_pairs(raw_pairs: list[str], hit: list[bool]) -> list[str] | None:
    """Pairs to keep in a stored template, tracking parameters dropped; None if any looks like a credential."""
    kept = []
    for pair, is_query in zip(raw_pairs, hit, strict=True):
        name, _, value = pair.partition("=")
        if is_query:
            kept.append(f"{name}={{q}}")
        elif _TOKEN_NAME.search(unquote_plus(name)) or _OPAQUE_VALUE.fullmatch(unquote_plus(value)):
            return None
        elif not _TRACKING.fullmatch(name):
            kept.append(pair)
    return kept


def learn_template(url: str, query: str, url_before: str) -> str | None:
    """The URL a typed search landed on, with the query's value replaced by {q}; None if the query is not in it.

    Templates persist and are replayed later, so only same-host http(s) landings without credential-like
    parameters are learned. Other parameters keep their original encoding.
    """
    host = _web_host(url)
    if not host or host != _web_host(url_before):
        return None
    parts = urlsplit(url)
    want = normalize(query)
    raw_pairs = [p for p in parts.query.split("&") if p]
    hit = [normalize(unquote_plus(p.partition("=")[2])) == want for p in raw_pairs]
    if not want or not any(hit):
        return None
    pairs = _safe_pairs(raw_pairs, hit)
    if pairs is None:
        return None
    return f"{parts.scheme}://{parts.netloc}{parts.path}?{'&'.join(pairs)}"


def render(template: str, query: str) -> str:
    return template.replace("{q}", quote_plus(query))


def matches_template(url: str, template: str) -> bool:
    """The host is literal in the template, so a match also pins the host."""
    pattern = r"[^&#]+".join(re.escape(part) for part in template.split("{q}"))
    return re.fullmatch(pattern, url) is not None


class SearchTemplates:
    """Search URL templates by host; persisted to a JSON file when a path is given."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._memory: dict[str, str] = {}

    def _load(self) -> dict[str, str]:
        if self.path is None:
            return self._memory
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, url: str) -> str | None:
        return self._load().get(_host(url))

    def put(self, url: str, template: str) -> None:
        data = self._load()
        data[_host(url)] = template
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=1))


def default_templates() -> SearchTemplates:
    from .compiler import cache_dir

    return SearchTemplates(cache_dir() / "search_templates.json")


def discover(browser, page_url: str) -> str | None:
    try:
        xml_text = browser.evaluate(OPENSEARCH_JS, await_promise=True)
    except (RuntimeError, ValueError) as exc:
        log.info("no OpenSearch description on %s: %s", page_url, exc)
        return None
    return template_from_opensearch(xml_text, page_url) if isinstance(xml_text, str) else None
