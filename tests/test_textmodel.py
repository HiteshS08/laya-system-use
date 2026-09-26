import json
from unittest.mock import Mock

import httpx
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
    for name in ("TEXT_MODEL_API_KEY", "TEXT_MODEL_BASE_URL", "TEXT_MODEL", "TEXT_MODEL_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    post = Mock(return_value=reply('{"a": 1}'))
    monkeypatch.setattr(model, "post_json", post)
    textmodel.complete_json("sys", "user")
    url, key, body = post.call_args.args
    assert url == "http://127.0.0.1:8080/v1/chat/completions" and key == "local"
    assert body["temperature"] == 0 and body["messages"][1] == {"role": "user", "content": "user"}
    assert post.call_args.kwargs == {"timeout": 120.0}


def test_text_model_timeout_is_configurable_without_changing_policy_timeout(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_TIMEOUT_SECONDS", "75")
    post = Mock(return_value=reply('{"a": 1}'))
    monkeypatch.setattr(model, "post_json", post)
    textmodel.complete_json("sys", "user")
    assert post.call_args.kwargs == {"timeout": 75.0}
    with pytest.raises(ValueError, match="must be positive"):
        monkeypatch.setenv("TEXT_MODEL_TIMEOUT_SECONDS", "0")
        textmodel.complete_json("sys", "user")


def test_transport_timeout_reports_the_endpoint_and_error(monkeypatch):
    monkeypatch.setattr(model.CLIENT, "post", Mock(side_effect=httpx.ReadTimeout("read timed out")))
    with pytest.raises(RuntimeError, match=r"127\.0\.0\.1:8080.*ReadTimeout: read timed out"):
        model.post_json("http://127.0.0.1:8080/v1/chat/completions", "local", {}, timeout=120)


def test_choose_option_requires_an_offered_option(monkeypatch):
    monkeypatch.setattr(model, "post_json", Mock(return_value=reply('{"option": "Blue"}')))
    assert textmodel.choose_option("goal", "Colour", ["Red", "Blue"]) == "Blue"
    monkeypatch.setattr(model, "post_json", Mock(return_value=reply('{"option": "Green"}')))
    with pytest.raises(ValueError, match="not an offered option"):
        textmodel.choose_option("goal", "Colour", ["Red", "Blue"])
    sent = json.loads(model.post_json.call_args.args[2]["messages"][1]["content"])
    assert sent["options"] == ["Red", "Blue"]


def test_extra_fields_are_merged_into_the_request(monkeypatch):
    post = Mock(return_value=reply('{"a": 1}'))
    monkeypatch.setattr(model, "post_json", post)
    textmodel.complete_json("sys", "user", extra={"chat_template_kwargs": {"enable_thinking": False}})
    body = post.call_args.args[2]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0


def test_complete_text_returns_plain_content(monkeypatch):
    monkeypatch.setattr(model, "post_json", Mock(return_value=reply("<think>x</think>\nFIND Ada\n")))
    text, meta = textmodel.complete_text("sys", "user", max_tokens=32)
    assert text == "FIND Ada" and meta["usage"] == {"total_tokens": 5} and meta["attempts"] == 1


def test_complete_text_uses_the_compiler_model_when_set(monkeypatch):
    post = Mock(return_value=reply("FIND Ada"))
    monkeypatch.setattr(model, "post_json", post)
    monkeypatch.setenv("COMPILER_MODEL", "small-model")
    _, meta = textmodel.complete_text("sys", "user")
    assert post.call_args.args[2]["model"] == "small-model" and meta["model"] == "small-model"
