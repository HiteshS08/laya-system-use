"""Held-out goals on sites outside the 25-task suite: a guard against fitting the grammar to the suite.

Goals only. Before the first live run, each one gets a hand-validated check as a LiveTask (plan task D2).
Read-only public pages.
"""

HELDOUT: list[tuple[str, str]] = [
    ("https://docs.python.org/3/", "Open the documentation page for the itertools module."),
    ("https://docs.python.org/3/library/functools.html", "Jump to the section about functools.lru_cache."),
    ("https://developer.mozilla.org/en-US/", "Find the MDN reference page for Array.prototype.map."),
    ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Status", "Open the page about the 404 Not Found status."),
    ("https://pypi.org/", "Find the PyPI page for the httpx package."),
    ("https://en.wiktionary.org/wiki/Wiktionary:Main_Page", "Find the Wiktionary entry for the word serendipity."),
    ("https://en.wiktionary.org/wiki/serendipity", "Scroll down until the Anagrams section heading is visible."),
    ("https://lobste.rs/", "Open the comments page of the second story."),
    ("https://www.gutenberg.org/", "Find the Project Gutenberg page for Pride and Prejudice."),
    ("https://arxiv.org/", "Open the list of recent submissions in Computation and Language."),
    ("https://doc.rust-lang.org/book/", "Open the chapter about understanding ownership."),
    ("https://www.openstreetmap.org/", "Search the map for Zurich and stop when search results are listed."),
]
