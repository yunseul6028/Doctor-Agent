"""Eval harness: multi-directory case collection, subsets, per-set summaries, run comparison. No LLM calls."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval import compare as cmp  # noqa: E402
from eval import run_local  # noqa: E402

SAMPLE = ROOT / "data/sample_cases/synthetic_001.json"


def _make_sets(tmp_path: Path) -> tuple[Path, Path]:
    case = json.loads(SAMPLE.read_text(encoding="utf-8"))
    a, b = tmp_path / "cases_alpha", tmp_path / "beta_cases"
    for d, names in ((a, ["a1", "a2"]), (b, ["b1"])):
        d.mkdir()
        for n in names:
            (d / f"{n}.json").write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
    return a, b


def test_set_name():
    assert run_local.set_name(Path("data/cases_clinicalqa")) == "clinicalqa"
    assert run_local.set_name(Path("data/sample_cases")) == "sample"
    assert run_local.set_name(Path("data/cases_clinicalqa_aug")) == "clinicalqa_aug"


def test_collect_multi_dir_glob_and_dedup(tmp_path):
    a, b = _make_sets(tmp_path)
    got = run_local.collect_cases([str(a), str(b), str(a / "a1.json")])
    assert [(s, p.stem) for s, p in got] == [("alpha", "a1"), ("alpha", "a2"), ("beta", "b1")]
    globbed = run_local.collect_cases([str(tmp_path / "*")])
    assert sorted(p.stem for _, p in globbed) == ["a1", "a2", "b1"]


def test_select_sample_and_limit():
    items = list(range(20))
    s1 = run_local.select_cases(items, sample=5, seed=3)
    assert s1 == run_local.select_cases(items, sample=5, seed=3) and len(s1) == 5 and s1 == sorted(s1)
    assert run_local.select_cases(items, limit=3) == [0, 1, 2]
    assert run_local.select_cases(items, sample=5, seed=3, limit=2) == s1[:2]


def test_run_local_end_to_end_with_dummy(tmp_path):
    a, b = _make_sets(tmp_path)
    out = run_local.main(["--doctor", "dummy", "--patient", "keyword", "--judge", "none", "--no-view",
                          "--cases", str(a), str(b), "--workers", "2", "--label", "t", "--out", str(tmp_path / "res")])
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["label"] == "t" and "prompt_version" in d and "commit" in d
    assert {r["set"] for r in d["cases"]} == {"alpha", "beta"}
    assert [r["case"] for r in d["cases"]] == ["a1", "a2", "b1"]  # input order despite threads
    assert set(d["avg_by_set"]) == {"alpha", "beta"} and d["avg_by_set"]["alpha"]["n"] == 2
    assert set(d["avg"]) == {"accuracy", "efficiency", "safety", "case_checks"}
    assert all(r["path"].endswith(".json") for r in d["cases"])


def _row(case, set_, acc, turns=None, dx="x"):
    return {"case": case, "set": set_, "answer": "ans", "diagnosis": dx, "n_turns": 2,
            "scores": {"accuracy": acc, "efficiency": 0.5, "safety": None, "case_checks": None},
            "turns": turns or []}


def test_compare_flips_intersection_and_not_provided():
    np_turns = [{"type": "EXAM", "response": "이 진찰 결과는 제공되지 않습니다."}, {"type": "TEST", "response": "WBC 12000"},
                {"type": "ASK", "response": "제공되지 않습니다"}]
    base = {"_name": "a", "label": "base", "cases": [_row("c1", "s1", 1.0), _row("c2", "s1", 0.0), _row("c3", "s2", 0.5),
                                                      _row("only_a", "s2", 1.0)]}
    new = {"_name": "b", "label": "new", "cases": [_row("c1", "s1", 0.0, dx="y"), _row("c2", "s1", 1.0, np_turns),
                                                    _row("c3", "s2", 1.0)]}
    res = cmp.compare([base, new])
    assert res["n_shared"] == 3
    fl = res["flips"][0]
    assert [f["case"] for f in fl["right_to_wrong"]] == ["s1/c1"] and fl["right_to_wrong"][0]["after"] == "y"
    assert [f["case"] for f in fl["wrong_to_right"]] == ["s1/c2", "s2/c3"]  # 0.5 partial counts as wrong
    assert res["overall"][0]["accuracy"] == 0.5 and res["overall"][1]["accuracy"] == round(2 / 3, 3)
    assert res["overall"][1]["not_provided"] == 0.5 and res["overall"][0]["not_provided"] is None  # ASK ignored
    assert res["by_set"][1]["s2"]["n"] == 1
    text, md = cmp.to_text(res), cmp.to_markdown(res)
    assert "right→wrong 1, wrong→right 2" in text and md.startswith("Shared cases: 3") and "| s1 |" in md


def test_compare_old_results_without_set_match_by_case_name():
    old = {"_name": "old", "cases": [{k: v for k, v in _row("c1", None, 1.0).items() if k != "set"}]}
    new = {"_name": "new", "cases": [_row("c1", "s1", 1.0)]}
    res = cmp.compare([old, new])
    assert res["n_shared"] == 1 and res["flips"][0] == {"right_to_wrong": [], "wrong_to_right": []}


def test_compare_cli(tmp_path, capsys):
    for n, acc in (("run_1", 1.0), ("run_2", 0.0)):
        (tmp_path / f"{n}.json").write_text(json.dumps({"cases": [_row("c1", "s", acc)]}), encoding="utf-8")
    cmp.main(["--latest", "2", "--results", str(tmp_path), "--md"])
    assert "right→wrong 1" in capsys.readouterr().out
