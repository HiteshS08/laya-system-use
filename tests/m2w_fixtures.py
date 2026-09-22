import json

HTML = """<html backend_node_id="1">
<a backend_node_id="10"><text backend_node_id="11">NFL Scores</text></a>
<input backend_node_id="20"/>
<button backend_node_id="30"><text backend_node_id="31">Search</text></button>
<div backend_node_id="40"><text backend_node_id="41">Footer</text></div>
</html>"""


def cand(bid, tag, **attrs):
    return {"tag": tag, "backend_node_id": bid, "attributes": json.dumps({"backend_node_id": bid, **attrs})}


ALL = {
    "10": cand("10", "a"),
    "20": cand("20", "input", type="text", placeholder="Find", input_value=""),
    "30": cand("30", "button"),
    "40": cand("40", "div"),
}


def make_step(op="CLICK", gold="10", value=""):
    return {
        "action_uid": "u",
        "cleaned_html": HTML,
        "operation": {"op": op, "original_op": op, "value": value},
        "pos_candidates": [ALL[gold]],
        "neg_candidates": [c for k, c in ALL.items() if k != gold],
    }


def make_task(*steps):
    return {"annotation_id": "t1", "website": "site", "domain": "D", "confirmed_task": "Find NFL scores",
            "actions": list(steps)}
