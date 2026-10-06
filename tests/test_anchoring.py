"""Broad starting DDx and the premature-closure (anchoring) check (agent/anchoring.py). No LLM, no network."""
from doctor_agent.agent import anchoring as A
from doctor_agent.agent.anchoring import COMMON, MIN_TURNS, anchoring_check, initial_differential, render_for_prompt
from doctor_agent.agent.ledger import DdxLedger, Finding, FindingsLedger
from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.knowledge.clinical_rules import CATEGORY_NAMES

APPY = "급성 충수염"


def names(ddx):
    return [d["dx"] for d in ddx]


# ------------------------------------------------------------------------------------------ initial_differential
def test_chest_pain_has_cant_miss_and_common_causes():
    ddx = initial_differential("45세 남성. 주호소: 2시간 전 시작된 가슴 통증")
    assert 0 < len(ddx) <= A.MAX_DDX
    assert "급성 관상동맥 증후군" in names(ddx) and "대동맥 박리" in names(ddx)
    assert "위식도 역류질환" in names(ddx)
    assert {d["tag"] for d in ddx} <= {"위험", "흔함", "KB"}
    assert ddx[0]["tag"] == "위험"  # can't-miss first
    assert len(set(names(ddx))) == len(ddx)


def test_sex_filter_drops_female_only_diagnoses_for_men():
    ddx = initial_differential("30세 남성. 주호소: 하루 전부터 시작된 오른쪽 아랫배 통증")
    assert not any("임신" in n or "골반염" in n for n in names(ddx))
    assert APPY in names(ddx)


def test_neonate_gets_neonatal_causes_not_adult_ones():
    ddx = initial_differential("생후 4일 여아. 주호소: 어제부터 피부가 노래짐")
    n = names(ddx)
    assert "신생아 생리적 황달" in n
    assert "알코올성 간질환" not in n and "췌담도 악성 종양" not in n
    assert n[0].startswith("신생아")  # neonatal can't-miss first


def test_adult_drops_neonatal_items():
    ddx = initial_differential("60세 여성. 주호소: 2주 전부터 피부가 노랗게 변함")
    assert not any("신생아" in n for n in names(ddx))


def test_extra_category_covers_dizziness():
    ddx = initial_differential("31세 여성. 주호소: 오늘 새벽부터 방이 빙빙 도는 느낌")
    assert "양성 돌발성 체위성 현훈" in names(ddx)


def test_unknown_complaint_and_empty_input_do_not_crash():
    assert isinstance(initial_differential(""), list)
    assert isinstance(initial_differential("29세 남성. 주호소: 전반적으로 몸 상태가 나빠진 느낌"), list)


def test_kb_failure_degrades_to_no_kb_rows(monkeypatch):
    from doctor_agent.knowledge import kb
    monkeypatch.setattr(kb, "available", lambda: False)
    ddx = initial_differential("45세 남성. 주호소: 2시간 전 시작된 가슴 통증")
    assert ddx and all(d["tag"] != "KB" for d in ddx)


def test_render_is_short_and_grouped():
    ddx = initial_differential("45세 남성. 주호소: 2시간 전 시작된 가슴 통증")
    text = render_for_prompt(ddx)
    assert 0 < len(text) <= A.MAX_RENDER
    assert "반드시 배제" in text and "흔한 원인" in text
    assert render_for_prompt([]) == ""
    long = [{"dx": "가" * 60, "tag": "흔함"} for _ in range(8)]
    assert len(render_for_prompt(long)) <= A.MAX_RENDER


def test_common_table_is_sourced():
    for cat, (ref, rows) in COMMON.items():
        assert rows, cat
        assert cat in CATEGORY_NAMES or cat in A.EXTRA_CATEGORIES, cat
        if ref is not None:
            assert ref.pmid and ref.verified, cat


# ------------------------------------------------------------------------------------------ anchoring_check
def _state(snapshots, actions, top_for, top_against=(), p=0.7, initial="30세 남성. 주호소: 하루 전부터 시작된 오른쪽 아랫배 통증",
           responses=None):
    st = CaseState(initial_info=initial)
    responses = responses or ["우하복부가 아파요. 발열도 있어요."] * len(actions)
    for (typ, content, reason), snap, resp in zip(actions, snapshots, responses):
        st.turns.append(Turn(Action(ActionType(typ), content, reason), resp, snap))
    st.ddx_ledger = DdxLedger()
    st.ddx_ledger.update([{"dx": APPY, "p": p, "for": list(top_for), "against": list(top_against)},
                          {"dx": "급성 위장관염", "p": 0.2}])
    st.findings = FindingsLedger()
    return st


SNAP = [{"dx": APPY, "p": 0.6}, {"dx": "급성 위장관염", "p": 0.3}]
ASKS = [("ASK", "구토가 있나요?", ""), ("ASK", "설사가 있나요?", ""), ("ASK", "식욕은 어떤가요?", "")]


def test_not_before_min_turns():
    st = _state([SNAP] * 2, ASKS[:2], ["우하복부 통증", "발열"])
    assert anchoring_check(st, min_turns=3) is None
    three = _state([SNAP] * 3, ASKS, ["우하복부 통증", "발열"])
    assert anchoring_check(three, min_turns=3) is not None
    assert MIN_TURNS == 5 and anchoring_check(three) is None  # default gate (calibrated, docs/experiments.md)
    more = ASKS + [("ASK", "열은 몇 도였나요?", ""), ("ASK", "통증이 이동했나요?", "")]
    assert anchoring_check(_state([SNAP] * 5, more, ["우하복부 통증", "발열"])) is not None


def test_stable_untested_fires_with_devils_advocate_prompt():
    st = _state([SNAP] * 3, ASKS, ["우하복부 통증", "발열"])
    r = anchoring_check(st, min_turns=3)
    assert r is not None and r["dx"] == APPY
    assert "stable_untested" in r["reasons"]
    assert APPY in r["prompt_ko"] and "반론" in r["prompt_ko"] and "다른 진단" in r["prompt_ko"]
    assert r["why_ko"] and r["suggested_actions"]


def test_targeted_test_prevents_stable_untested():
    acts = ASKS[:2] + [("TEST", "복부 CT", "충수염 확인")]
    st = _state([SNAP] * 3, acts, ["우하복부 통증", "발열"])
    assert anchoring_check(st, min_turns=3) is None


def test_targeted_by_kb_test_label_without_the_name():
    acts = ASKS[:2] + [("TEST", "복부 초음파로 충수 비후 여부 확인", "")]
    st = _state([SNAP] * 3, acts, ["우하복부 통증", "발열"])
    assert anchoring_check(st, min_turns=3) is None


def test_recently_changed_top_is_not_anchoring():
    other = [{"dx": "급성 위장관염", "p": 0.6}, {"dx": APPY, "p": 0.3}]
    st = _state([other, other, SNAP], ASKS, ["우하복부 통증", "발열"])
    assert anchoring_check(st, min_turns=3) is None


def test_weak_support_fires_when_evidence_is_not_in_the_conversation():
    acts = ASKS[:2] + [("TEST", "복부 CT", "충수염 확인")]
    st = _state([SNAP] * 3, acts, ["맥버니 압통", "반발통"], p=0.8)
    r = anchoring_check(st, min_turns=3)
    assert r is not None and r["reasons"] == ["weak_support"]


def test_unverified_finding_does_not_ground_support():
    acts = ASKS[:2] + [("TEST", "복부 CT", "충수염 확인")]
    st = _state([SNAP] * 3, acts, ["우하복부 통증", "발열"], p=0.8)
    st.findings.items += [Finding("우하복부 통증", "양성", verified=False), Finding("발열", "양성", verified=False)]
    r = anchoring_check(st, min_turns=3)
    assert r is not None and "weak_support" in r["reasons"]


def test_contradicted_needs_grounded_concrete_findings():
    acts = ASKS[:2] + [("TEST", "복부 CT", "충수염 확인")]
    resp = ["우하복부가 아파요. 발열도 있어요.", "설사를 하루 10번 했어요.", "복부 CT에서 장벽 부종만 보입니다."]
    st = _state([SNAP] * 3, acts, ["우하복부 통증", "발열"], top_against=["잦은 설사", "CT상 장벽 부종"], responses=resp)
    r = anchoring_check(st, min_turns=3)
    assert r is not None and r["reasons"] == ["contradicted"]
    pending = _state([SNAP] * 3, acts, ["우하복부 통증", "발열"], top_against=["CT 결과 없음", "백혈구 아직 미확인"],
                     responses=resp)
    assert anchoring_check(pending, min_turns=3) is None


def test_low_probability_top_is_ignored():
    st = _state([SNAP] * 3, ASKS, ["우하복부 통증", "발열"], p=0.3)
    assert anchoring_check(st, min_turns=3) is None


def test_empty_state():
    assert anchoring_check(CaseState(initial_info="")) is None
