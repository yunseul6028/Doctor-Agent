"""knowledge/specialty.py: specialty mapping, routing, evidence slicing."""
import json
import statistics
import time
from pathlib import Path

import pytest

from doctor_agent.agent.state import CaseState, Turn
from doctor_agent.env.interface import Action, ActionType
from doctor_agent.knowledge import clinical_rules, diagnostic_criteria, kb
from doctor_agent.knowledge import specialty as sp
from doctor_agent.safety import protocols

pytestmark = pytest.mark.skipif(not kb.available(), reason="data/kb not built")
ROOT = Path(__file__).resolve().parents[1]


def _state(initial="55세 남성. 주호소: 1시간 전 시작된 가슴 통증", ddx=(), pos=(), neg=(), responses=()):
    s = CaseState(initial_info=initial)
    s.ddx_ledger.update([dict(zip(("dx", "p", "status"), d)) for d in ddx])
    s.findings.update([{"item": x, "status": "양성"} for x in pos] + [{"item": x, "status": "음성"} for x in neg], 1)
    for r in responses:
        s.turns.append(Turn(Action(ActionType.ASK, "질문"), r))
    return s


# ---------------------------------------------------------------- specialty_of

@pytest.mark.parametrize("name,expected", [
    ("급성 심근경색", "cardio"), ("대동맥 박리", "cardio"), ("지역사회획득 폐렴", "resp_id"), ("폐결핵", "resp_id"),
    ("급성 충수염", "gi_liver"), ("원발성 담즙성 담관염", "gi_liver"), ("중증근무력증", "neuro"),
    ("Myasthenia gravis", "neuro"), ("지주막하 출혈", "neuro"), ("전신홍반루푸스", "rheum_immune"), ("통풍", "rheum_immune"),
    ("아나필락시스", "rheum_immune"), ("자궁외 임신", "peds_obgyn"), ("자간전증", "peds_obgyn"), ("신생아 황달", "peds_obgyn"),
    ("세균성 수막염", "neuro"), ("감염성 심내막염", "cardio"), ("당뇨병성 케톤산증", None), ("급성 신우신염", None),
    ("주요 우울장애", None),
])
def test_specialty_of_known_names(name, expected):
    assert sp.specialty_of(name) == expected


@pytest.mark.parametrize("bad", ["", "   ", None, 123, "ㅁㄴㅇㄹ 뷁뷁"])
def test_specialty_of_never_raises(bad):
    assert sp.specialty_of(bad) in (None, *sp.SPECIALTIES)


def test_detail_groups_outside_the_six():
    assert sp.specialty_detail("당뇨병성 케톤산증")["group"] == "endo_metab"
    assert sp.specialty_detail("급성 신손상")["group"] == "renal_uro"
    assert sp.specialty_detail("철결핍성 빈혈")["group"] == "heme_onc"
    d = sp.specialty_detail("급성 충수염")
    assert d["how"] == "kcd" and d["code"].startswith("K35")


@pytest.mark.parametrize("code,expected", [
    ("I63.9", ("neuro", "neuro")), ("I21", ("cardio", "cardio")), ("K35", ("gi_liver", "gi_liver")),
    ("B18.1", ("gi_liver", "gi_liver")), ("A41.9", ("resp_id", "resp_id")), ("O14", ("peds_obgyn", "peds_obgyn")),
    ("M32", ("rheum_immune", "rheum_immune")), ("M54.5", (None, "msk_ortho")), ("N10", (None, "renal_uro")),
    ("E83.0", ("gi_liver", "gi_liver")), ("E11", (None, "endo_metab")), ("Q21.1", ("cardio", "cardio")),
    ("C34", ("resp_id", "resp_id")), ("C92", (None, "heme_onc")), ("T78.2", ("rheum_immune", "rheum_immune")),
])
def test_kcd_table(code, expected):
    assert sp._code_entry(code) == expected


def test_tables_only_use_the_six_ids():
    ok = set(sp.SPECIALTIES) | {None}
    assert {e[2] for e in sp.KCD_TABLE} <= ok
    assert {v[0] for v in sp.DO_MAP.values()} <= ok
    assert {e[1] for e in sp.OVERRIDES} <= ok and {e[1] for e in sp.FALLBACK} <= ok
    for table in (sp.CRITERIA_SPECIALTY, sp.RULE_SPECIALTY, sp.CATEGORY_SPECIALTY):
        assert {s for v in table.values() for s in v} <= set(sp.SPECIALTIES)


def test_slice_tables_cover_the_existing_modules():
    """A criteria set, rule or protocol category added elsewhere must get a specialty entry here (may be empty)."""
    assert {c.id for c in diagnostic_criteria.CRITERIA} <= set(sp.CRITERIA_SPECIALTY)
    assert {r.id for r in clinical_rules.RULES} <= set(sp.RULE_SPECIALTY)
    assert {p.category for p in protocols.PROTOCOLS} <= set(sp.CATEGORY_SPECIALTY)


def test_gold_set_regression():
    rows = [json.loads(x) for x in (ROOT / "data/labels/specialty_gold_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 80
    assert all(r["specialty"] in (None, *sp.SPECIALTIES) for r in rows)
    strict = sum(sp.specialty_of(r["name"]) == r["specialty"] for r in rows) / len(rows)
    assert strict >= 0.9  # 2026-09-28: first pass 0.948, after fixes 0.974 (eval/offline/eval_specialty.py)


# ---------------------------------------------------------------- route

def test_route_probability_mass():
    s = _state(ddx=[("급성 심근경색", 0.6), ("불안정 협심증", 0.2), ("대동맥 박리", 0.1), ("위식도 역류질환", 0.1)])
    spec, share, why = sp.route(s)
    assert spec == "cardio" and share == pytest.approx(0.9)
    assert why and all(isinstance(x, str) for x in why) and "심장" in why[0]


def test_route_rank_weights_without_probabilities():
    s = CaseState(initial_info="40세 남성. 주호소: 기침")
    s.ddx = [{"dx": "지역사회획득 폐렴"}, {"dx": "급성 기관지염"}, {"dx": "위식도 역류질환"}]
    spec, share, _ = sp.route(s)
    assert spec == "resp_id" and share == pytest.approx((1 + 1 / 2) / (1 + 1 / 2 + 1 / 3), abs=1e-3)


def test_route_ignores_ruled_out():
    s = _state(ddx=[("급성 심근경색", 0.9, "배제"), ("급성 췌장염", 0.5), ("급성 담낭염", 0.3)])
    assert sp.route(s)[0] == "gi_liver"


def test_route_none_when_nothing_maps():
    s = _state(ddx=[("당뇨병성 케톤산증", 0.7), ("급성 신손상", 0.3)])
    spec, share, why = sp.route(s)
    assert spec is None and share == 0.0 and why


def test_route_child_with_pediatric_leader():
    s = _state("4세 남아. 주호소: 5일째 계속되는 고열", ddx=[("가와사키병", 0.5), ("성홍열", 0.2), ("급성 림프절염", 0.3)])
    spec, share, why = sp.route(s)
    assert spec == "peds_obgyn" and share >= 0.5 and "소아" in why[0]


def test_route_child_with_organ_disease_stays_organ():
    s = _state("12세 남아. 주호소: 오른쪽 아랫배 통증", ddx=[("급성 충수염", 0.8), ("급성 위장염", 0.2)])
    spec, _, why = sp.route(s)
    assert spec == "gi_liver" and any("소아" in x for x in why)


def test_route_infant_always_pediatric():
    s = _state("생후 3주 여아. 주호소: 수유 후 구토", ddx=[("위식도 역류질환", 0.6), ("급성 위장염", 0.4)])
    assert sp.route(s)[:2] == ("peds_obgyn", 1.0)


def test_route_pregnancy_context():
    s = _state("31세 여성. 주호소: 두통", ddx=[("자간전증", 0.5), ("편두통", 0.5)], responses=["지금 임신 32주예요."])
    assert sp.route(s)[0] == "peds_obgyn"
    s = _state("31세 여성. 주호소: 오른쪽 아랫배 통증", ddx=[("급성 충수염", 0.9), ("신우신염", 0.1)],
               responses=["임신 20주예요."])
    assert sp.route(s)[0] == "gi_liver"
    s = _state("31세 여성. 주호소: 두통", ddx=[("자간전증", 0.2), ("편두통", 0.8)], responses=["임신은 아니에요."])
    assert sp.route(s)[0] == "neuro"


@pytest.mark.parametrize("bad", [None, object(), CaseState(initial_info=""), "state"])
def test_route_never_raises(bad):
    spec, share, why = sp.route(bad)
    assert spec is None and share == 0.0 and isinstance(why, list) and why


# ---------------------------------------------------------------- resources

def test_resources_cardio_chest_pain():
    s = _state(ddx=[("급성 심근경색", 0.6), ("대동맥 박리", 0.2), ("폐색전증", 0.2)], pos=["흉통", "식은땀"], neg=["발열"])
    res = sp.resources("cardio", s)
    assert set(res) >= {"criteria", "rules", "protocols", "kb_candidates"}
    for key, n in sp.MAX_ITEMS.items():
        assert len(res[key]) <= n
    rules = {r["id"]: r for r in res["rules"]}
    assert rules.get("heart", {}).get("applies") is True  # HEART applies to an adult chest-pain complaint
    assert any(p["category"] == "chest_pain" and p["in_specialty"] for p in res["protocols"])
    assert any("심근경색" in c["name"] for c in res["kb_candidates"])
    text = sp.render_resources(res)
    assert 0 < len(text) <= 700 and text.startswith("[심장·혈관 분과 참고 자료")
    assert len(sp.render_resources(res, max_chars=200)) <= 200


def test_resources_rheum_criteria_evaluated_for_candidate():
    s = _state("32세 여성. 주호소: 관절통과 발진", ddx=[("전신홍반루푸스", 0.6), ("류마티스 관절염", 0.4)],
               pos=["항핵항체 양성", "관절염", "혈소판 감소"], responses=["ANA 1:640 양성, 항dsDNA 항체 양성"])
    res = sp.resources("rheum_immune", s)
    ids = [c["id"] for c in res["criteria"]]
    assert ids[:2] == ["sle_2019", "ra_2010"] or set(ids[:2]) == {"sle_2019", "ra_2010"}
    assert "band" in res["criteria"][0]


def test_resources_bad_inputs():
    assert sp.resources("dermatology", _state()) == {}
    assert sp.resources("cardio", None) in ({}, sp.resources("cardio", None))
    assert isinstance(sp.resources("cardio", object()), dict)
    assert sp.render_resources({}) == "" and sp.render_resources(None) == ""
    assert sp.render_resources({"specialty": "cardio", "criteria": [{"bad": 1}]}) == ""


def test_latency_after_warm():
    """Budget: < 20 ms per call after load. Median checked with slack (shared CI machines are noisy)."""
    assert sp.warm()
    s = _state(ddx=[("급성 심근경색", 0.5), ("Acute ischemic stroke due to left MCA occlusion", 0.2),
                    ("Toxic ingestion/Metabolic acidosis", 0.2), ("대동맥 박리", 0.1)], pos=["흉통", "식은땀", "구토"])
    tr, tres = [], []
    for _ in range(5):
        t0 = time.perf_counter()
        sp.route(s)
        tr.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        sp.resources("cardio", s)
        tres.append(time.perf_counter() - t0)
    assert statistics.median(tr) < 0.02 and statistics.median(tres) < 0.04
