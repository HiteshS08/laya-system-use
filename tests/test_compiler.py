from evals.live_tasks import TASKS
from jev_ultrafast.compiler import COMPILER_SYSTEM, ProgramCache, compile_goal
from jev_ultrafast.program import fallback_program


def fake(*outputs):
    calls = []

    def complete(system, user, **kw):
        calls.append(user)
        out = outputs[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return out, {"latency_ms": 5, "usage": {"completion_tokens": 7}}
    return complete, calls


GOAL = "Find Alan Turing's article, then open the Turing Award article from it."


def test_compiles_goal_into_program():
    complete, calls = fake("FIND Alan Turing\nFIND Turing Award")
    program, meta = compile_goal(GOAL, complete=complete)
    assert [s.target for s in program.subgoals] == ["Alan Turing", "Turing Award"]
    assert meta["attempts"] == 1 and meta["source"] == "compiler" and len(calls) == 1 and GOAL in calls[0]
    assert meta["completion_tokens"] == 7


def test_retries_once_with_the_error_then_falls_back():
    complete, calls = fake("GO somewhere", "still bad")
    program, meta = compile_goal("Do a thing", complete=complete)
    assert program == fallback_program("Do a thing") and meta["attempts"] == 2 and "error" in meta
    assert "invalid" in calls[1]


def test_model_failure_falls_back():
    complete, _ = fake(RuntimeError("server down"))
    program, meta = compile_goal("Do a thing", complete=complete)
    assert program.source == "fallback" and "server down" in meta["error"]


def test_cache_hit_skips_the_model(tmp_path):
    cache = ProgramCache(tmp_path / "programs.json")
    complete, calls = fake("FIND Ada Lovelace")
    compile_goal("Find Ada", complete=complete, cache=cache)
    program, meta = compile_goal("  find ADA ", complete=complete, cache=ProgramCache(tmp_path / "programs.json"))
    assert len(calls) == 1 and meta["source"] == "cache" and program.source == "cache" and meta["attempts"] == 0


def test_fallback_programs_are_not_cached(tmp_path):
    cache = ProgramCache(tmp_path / "programs.json")
    complete, _ = fake("bad", "bad")
    compile_goal("Do a thing", complete=complete, cache=cache)
    assert cache.get("Do a thing") is None


def test_corrupt_cache_file_is_treated_as_empty(tmp_path):
    path = tmp_path / "programs.json"
    path.write_text("{not json")
    assert ProgramCache(path).get("Find Ada") is None


def test_prompt_examples_are_not_suite_goals():
    for task in TASKS.values():
        assert task.goal not in COMPILER_SYSTEM


def test_cache_key_changes_with_the_prompt_and_the_model(tmp_path, monkeypatch):
    from jev_ultrafast import compiler

    complete, calls = fake("FIND Ada", "FIND Ada", "FIND Ada")
    compile_goal("Find Ada", complete=complete, cache=ProgramCache(tmp_path / "p.json"))
    monkeypatch.setenv("COMPILER_MODEL", "other-model")
    compile_goal("Find Ada", complete=complete, cache=ProgramCache(tmp_path / "p.json"))
    monkeypatch.setattr(compiler, "COMPILER_SYSTEM", COMPILER_SYSTEM + "\nTask: x\nFIND x")
    compile_goal("Find Ada", complete=complete, cache=ProgramCache(tmp_path / "p.json"))
    assert len(calls) == 3
    assert compiler.cache_key("  find ADA ") == compiler.cache_key("Find Ada")
