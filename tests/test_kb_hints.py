import pytest

from doctor_agent.agent import kb_hints
from doctor_agent.agent.state import CaseState
from doctor_agent.config import AgentConfig
from doctor_agent.knowledge import kb

pytestmark = pytest.mark.skipif(not kb.available(), reason="data/kb not built")


def _state(pos=(), neg=(), ddx=(), initial="45세 남성. 주호소: 기침"):
    s = CaseState(initial_info=initial)
    s.findings.update([{"item": x, "status": "양성"} for x in pos] + [{"item": x, "status": "음성"} for x in neg], 1)
    s.ddx_ledger.update([{"dx": d, "p": p} for d, p in ddx])
    return s


RESP = ["기침", "발열", "흉통", "호흡곤란", "객담", "오한", "근육통"]


def test_no_candidate_hint_below_three_positives():
    assert kb_hints.candidate_hint(_state(RESP[:2]), set()) is None


def test_candidate_hint_thresholds_and_limits():
    seen: set = set()
    s = _state(RESP[:3], ddx=[("폐렴", 0.5)])
    h = kb_hints.candidate_hint(s, seen)
    assert h and h.startswith("고려해 볼 다른 질환 (참고용, 확진 근거 아님")
    assert len(h) <= kb_hints.MAX_CHARS
    assert h.count("; ") <= kb_hints.MAX_CANDIDATES - 1 and "[" in h  # ≤3 rows, with sources
    assert "폐렴(" not in h  # already in the DDx ledger
    assert kb_hints.candidate_hint(s, seen) is None  # same checkpoint: not repeated
    s.findings.update([{"item": x, "status": "양성"} for x in RESP[3:6]], 2)
    assert kb_hints.candidate_hint(s, seen)  # second checkpoint (6 positives)
    s.findings.update([{"item": "두통", "status": "양성"}], 3)
    assert kb_hints.candidate_hint(s, seen) is None  # at most two points per case


def test_discriminator_only_when_top2_close():
    far = _state(ddx=[("폐렴", 0.7), ("급성 기관지염", 0.3)])
    assert kb_hints.discriminator_hint(far, set()) is None
    seen: set = set()
    close = _state(ddx=[("폐렴", 0.5), ("급성 기관지염", 0.4)])
    h = kb_hints.discriminator_hint(close, seen)
    assert h and "폐렴" in h and "급성 기관지염" in h and len(h) <= kb_hints.MAX_CHARS
    feats = h.split("(참고용, 확진 근거 아님): ")[1].split(". 질문 예")[0]
    assert sum(p.split(" — ")[1].count(",") + 1 for p in feats.replace(" [KB]", "").split("; ")) <= kb_hints.MAX_FEATURES
    assert kb_hints.discriminator_hint(close, seen) is None  # once per pair


def test_sex_mismatch_warning():
    n = kb_hints.normalize_hint("자궁외 임신", "28세 남성. 주호소: 복통")
    assert n["code"].startswith("O00") and n["patient_sex"] == "남성" and "warning" in n
    ok = kb_hints.normalize_hint("자궁외 임신", "28세 여성. 주호소: 복통")
    assert ok["patient_sex"] == "여성" and "warning" not in ok
    assert "warning" not in kb_hints.normalize_hint("급성 충수염", "28세 남성")
    assert kb_hints.patient_sex("7살 여아, 발열") == "여성"
    assert kb_hints.patient_sex("주호소: 두통") == ""


def test_graceful_degradation(monkeypatch):
    monkeypatch.setattr(kb, "available", lambda: False)
    s = _state(RESP, ddx=[("폐렴", 0.5), ("급성 기관지염", 0.45)])
    assert kb_hints.step_hints(s, set()) == []
    assert kb_hints.normalize_hint("자궁외 임신", "28세 남성") == {}


def test_lookup_errors_are_swallowed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kb broken")
    for name in ("candidates", "discriminators", "normalize_diagnosis", "lookup"):
        monkeypatch.setattr(kb, name, boom)
    s = _state(RESP, ddx=[("폐렴", 0.5), ("급성 기관지염", 0.45)])
    assert kb_hints.candidate_hint(s, set()) is None
    assert kb_hints.discriminator_hint(s, set()) is None
    assert kb_hints.normalize_hint("폐렴", "") == {}


def test_use_kb_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_USE_KB", "0")
    assert AgentConfig().use_kb is False
    monkeypatch.delenv("AGENT_USE_KB")
    assert AgentConfig().use_kb is True


def test_single_test_result_match_is_enough(monkeypatch):
    from doctor_agent.agent import kb_hints
    from doctor_agent.knowledge import kb

    fake = [{"id": "X", "name_ko": "급성 췌장염", "name_en": "acute pancreatitis", "score": 9.0, "kcd": ["K85"],
             "matched": [{"id": "TF:lipase_high", "ko": "리파아제 상승", "finding": "리파아제 1850"}], "sources": ["curated"]},
            {"id": "Y", "name_ko": "위염", "name_en": "gastritis", "score": 5.0, "kcd": [],
             "matched": [{"id": "WD:1", "ko": "복통", "finding": "명치 통증"}], "sources": ["WD"]}]
    monkeypatch.setattr(kb, "candidates", lambda *a, **k: fake)
    monkeypatch.setattr(kb, "available", lambda: True)
    monkeypatch.setattr(kb_hints, "_positives", lambda state: ["a", "b", "c"])
    monkeypatch.setattr(kb_hints, "_negatives", lambda state: [])

    class S:
        class ddx_ledger:
            entries = []
    hint = kb_hints.candidate_hint(S(), set())
    assert hint and "급성 췌장염" in hint and "위염" not in hint


def test_candidate_hint_passes_sex_and_age(monkeypatch):
    from doctor_agent.agent import kb_hints
    from doctor_agent.knowledge import kb

    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return []
    monkeypatch.setattr(kb, "candidates", fake)
    monkeypatch.setattr(kb, "available", lambda: True)
    monkeypatch.setattr(kb_hints, "_positives", lambda state: ["a", "b", "c"])
    monkeypatch.setattr(kb_hints, "_negatives", lambda state: [])

    class S:
        initial_info = "28세 여성. 주호소: 복통"

        class ddx_ledger:
            entries = []
    kb_hints.candidate_hint(S(), set())
    assert seen.get("sex") == "여성" and seen.get("age") == 28
