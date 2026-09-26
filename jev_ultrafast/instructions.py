"""Step instructions for the actor. One function for serving and for training items, so both see the same text."""

TEMPLATES = {
    "OPEN": "Click the {target}.",
    "CLICK": "Click the {target}.",
    "LINK": "Click the link to {target}.",
    "RESULT": "Click the search result for {target}.",
    "SEARCH": 'Type "{value}" into the search box.',
    "OPEN_SEARCH": "Click the search control to open the search box.",
    "FILL": 'Type "{value}" into the {target} field.',
    "SELECT": 'Select "{value}" in the {target} dropdown.',
    "OPTION": 'Click the "{value}" option.',
    "SUBMIT": "Press Enter to submit.",
    "SCROLL": 'Scroll to "{target}".',
    "JUMP": "Click the {target} section link.",
    "DO": "{target}",
}


def instruction(kind: str, target: str = "", value: str = "") -> str:
    return TEMPLATES[kind].format(target=" ".join(target.split()), value=" ".join(value.split()))
