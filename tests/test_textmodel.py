import json
from unittest.mock import Mock

import pytest

from jev_ultrafast import model, textmodel


def reply(content):
    return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 5}}


@pytest.mark.parametrize(
    "raw",
    ['{"text": "Zurich"}', '```json\n{"text": "Zurich"}\n```', 'Sure! {"text": "Zurich"} done',
     '<think>{"x": 1}</think>{"text": "Zurich"}'],
)
def test_extract_json_accepts_common_wrappers(raw):
    assert textmodel.extract_json(raw) == {"text": "Zurich"}


@pytest.mark.parametrize("raw", ["no json here", "{broken", "[1, 2]"])
def test_extract_json_rejects_non_objects(raw):
    with pytest.raises(ValueError):
        textmodel.extract_json(raw)


def test_complete_json_retries_once_on_invalid_output(monkeypatch):
    post = Mock(side_effect=[reply("nope"), reply('{"a": 1}')])
    monkeypatch.setattr(model, "post_json", post)
    output, meta = textmodel.complete_json("sys", "user")
    assert output == {"a": 1} and meta["attempts"] == 2 and post.call_count == 2


def test_complete_json_gives_up_after_two_invalid_outputs(monkeypatch):
    monkeypatch.setattr(model, "post_json", Mock(return_value=reply("nope")))
    with pytest.raises(ValueError, match="no valid JSON"):
        textmodel.complete_json("sys", "user")


def test_defaults_to_the_local_server_without_any_key(monkeypatch):
    for name in ("TEXT_MODEL_API_KEY", "TEXT_MODEL_BASE_URL", "TEXT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    post = Mock(return_value=reply('{"a": 1}'))
    monkeypatch.setattr(model, "post_json", post)
    textmodel.complete_json("sys", "user")
    url, key, body = post.call_args.args
    assert url == "http://127.0.0.1:8080/v1/chat/completions" and key == "local"
    assert body["temperature"] == 0 and body["messages"][1] == {"role": "user", "content": "user"}


def test_choose_option_requires_an_offered_option(monkeypatch):
    monkeypatch.setattr(model, "post_json", Mock(return_value=reply('{"option": "Blue"}')))
    assert textmodel.choose_option("goal", "Colour", ["Red", "Blue"]) == "Blue"
    monkeypatch.setattr(model, "post_json", Mock(return_value=reply('{"option": "Green"}')))
    with pytest.raises(ValueError, match="not an offered option"):
        textmodel.choose_option("goal", "Colour", ["Red", "Blue"])
    sent = json.loads(model.post_json.call_args.args[2]["messages"][1]["content"])
    assert sent["options"] == ["Red", "Blue"]
