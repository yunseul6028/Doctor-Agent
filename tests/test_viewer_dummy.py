import json

from eval import viewer


def test_viewer_hides_dummy_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "RESULTS", tmp_path)
    (tmp_path / "run_1.json").write_text(json.dumps({"doctor_model": "dummy", "cases": []}), encoding="utf-8")
    (tmp_path / "run_2.json").write_text(json.dumps({"doctor_model": "gpt-oss-20b", "cases": []}), encoding="utf-8")
    html = viewer.build().read_text(encoding="utf-8")
    assert '"run_2"' in html and '"run_1"' not in html
    html = viewer.build(include_dummy=True).read_text(encoding="utf-8")
    assert '"run_1"' in html
