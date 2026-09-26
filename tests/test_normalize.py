"""Action-type normalisation (questions sent as EXAM/TEST) and DDx-name deduplication."""
import pytest

from doctor_agent.agent.ledger import DdxLedger
from doctor_agent.agent.policy import normalize_type
from doctor_agent.agent.text import same_dx
from doctor_agent.env.interface import Action, ActionType

QUESTIONS_AS_EXAM = [
    # real failures from local runs
    "환자에게 목 뻣뻣함(경부 강직)이 있는지, 그리고 두통이 발생했을 때 벼락두통(1시간 이내에 최고조에 달한 두통)이었는지 진찰하겠습니다.",
    "목이 뻣뻣하거나 고개를 숙일 때 통증이 심해지는지 확인하기 위해 경부 강 …",
    "두통이 언제부터 시작되었는지 확인",
    "이전에 비슷한 가슴 통증을 겪은 적이 있는지 확인",
    "최근 복용 중인 약물 확인",
    "흉통이 얼마나 지속되는지 확인",
    "환자분께 가족력 확인",
    "과거에 결핵을 앓았는지 확인",
    "숨이 차는 느낌이 있는지 확인",
    "평소 술을 드시는지 확인",
    "구토 증상이 있으신지 확인",
    "발열이 며칠째 지속됐는지 확인",
]
REAL_EXAMS = [
    "경부 강직 진찰",
    "Kernig 징후 확인",
    "복부 촉진",
    "양팔 혈압 측정",
    "신경학적 진찰(의식, 근력, 감각)",
    "우하복부 압통이 있는지 복부 촉진",
    "심잡음이 있는지 심장 청진",
    "Brudzinski sign 확인",
    "림프절 비대가 있는지 경부 진찰",
    "통증이 심해지는지 Kernig 징후로 확인",  # question + real exam → stays EXAM
    "안저 검사",
    "활력징후 측정",
]


@pytest.mark.parametrize("content", QUESTIONS_AS_EXAM)
def test_history_question_sent_as_exam_becomes_ask(content):
    assert normalize_type(Action(ActionType.EXAM, content)).type == ActionType.ASK


@pytest.mark.parametrize("content", REAL_EXAMS)
def test_real_exam_stays_exam(content):
    assert normalize_type(Action(ActionType.EXAM, content)).type == ActionType.EXAM


def test_tests_are_converted_only_without_a_test_name():
    for content in ["CBC, CRP", "흉부 X-ray", "혈액 배양 2세트", "뇌 CT", "심전도"]:
        assert normalize_type(Action(ActionType.TEST, content)).type == ActionType.TEST
    assert normalize_type(Action(ActionType.TEST, "과거에 검진에서 이상 소견을 들은 적이 있는지")).type == ActionType.TEST
    assert normalize_type(Action(ActionType.TEST, "당뇨 진단을 받은 적이 있는지 확인")).type == ActionType.ASK


def test_diagnose_untouched():
    a = Action(ActionType.DIAGNOSE, "환자에게 설명한 대로 급성 충수염")
    assert normalize_type(a) is a


@pytest.mark.parametrize("a,b", [
    ("다카야수 동맥염 (Takayasu arteritis)", "다카야수 동맥염"),
    ("타카야수 동맥염", "다카야수 동맥염"),
    ("타카야수동맥염(Takayasu)", "다카야수 동맥염 (Takayasu arteritis)"),
    ("롱 QT 증후군", "긴 QT 증후군"),
    ("QT 연장 증후군", "긴 QT 증후군"),
    ("대동맥 박리", "대동맥박리"),
    ("폐색전증 의심", "폐색전증"),
    ("Takayasu arteritis", "다카야수 동맥염 (Takayasu arteritis)"),
    ("대동맥류 (aortic aneurysm)", "흉부 대동맥류 (Aortic aneurysm)"),
    ("길랭-바레 증후군", "길랑바레 증후군"),
])
def test_same_dx(a, b):
    assert same_dx(a, b)


@pytest.mark.parametrize("a,b", [
    ("다카야수 동맥염", "거대세포 동맥염"),
    ("급성 충수염", "급성 담낭염"),
    ("1형 당뇨병", "2형 당뇨병"),
    ("세균성 수막염 (bacterial meningitis)", "바이러스성 수막염 (viral meningitis)"),
    ("긴 QT 증후군", "브루가다 증후군"),
    ("폐렴", "폐색전증"),
])
def test_different_dx(a, b):
    assert not same_dx(a, b)


def test_ddx_ledger_merges_variants_and_keeps_first_name():
    d = DdxLedger()
    d.update([{"dx": "다카야수 동맥염 (Takayasu arteritis)", "p": 0.4, "for": ["상지 혈압차"]},
              {"dx": "롱 QT 증후군", "p": 0.2}])
    d.update([{"dx": "타카야수 동맥염", "p": 0.7, "for": ["맥박 소실"]}, {"dx": "긴 QT 증후군", "p": 0.1}])
    d.update([{"dx": "다카야수 동맥염", "p": 0.8}])
    assert len(d.entries) == 2
    top = d.ranked()[0]
    assert top.dx == "다카야수 동맥염 (Takayasu arteritis)" and top.p == 0.8
    assert top.support == ["상지 혈압차", "맥박 소실"]
    assert d.entries[1].dx == "롱 QT 증후군" and d.entries[1].p == 0.1
