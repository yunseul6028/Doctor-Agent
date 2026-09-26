"""LLM-based virtual patient environment. Owned by eval-simulator.

Answers only from the case file. The diagnosis and scoring fields are hidden from the patient LLM.
Personas make the patient harder to interview (vague, anxious, minimizing, poor historian).
Not included in the submission ZIP (development and evaluation only).
"""
import hashlib
import json
import re

from doctor_agent.env.interface import Action, ActionType, Environment, Observation
from doctor_agent.llm.client import LLMClient

HIDDEN_KEYS = {"diagnosis", "aliases", "must_check", "_note", "teaching_point", "category", "difficulty", "source",
               "augmented", "augmented_full"}

BASE = """당신은 의료 시뮬레이션의 가상 환자 환경입니다. 아래 증례 정보만 사용해 의사의 요청에 한국어로 답하세요.

공통 규칙:
- ASK: 환자 본인(또는 증례에 보호자가 답한다고 되어 있으면 보호자)으로서 일상어로 답하세요. 의학 용어나 진단명은 쓰지 마세요.
  증례 정보에 없는 내용이면 "잘 모르겠어요" 또는 "아니요, 그런 건 없어요"라고 답하세요.
- EXAM: 증례에서 요청한 신체진찰 소견만 의학 용어로 전달하세요. 증례에 그 진찰 항목이 없으면 "이 진찰 결과는 제공되지 않습니다."라고 하세요.
- TEST: 증례에서 요청한 검사 결과만 그대로 전달하세요. 증례에 그 검사가 없으면 "이 검사 결과는 제공되지 않습니다."라고 하세요.
  "특이 소견 없음"이나 "정상"은 증례에 그렇게 적혀 있을 때만 쓰세요. 없는 결과를 정상이라고 하면 안 됩니다.
  여러 검사를 요청했는데 일부만 증례에 있으면, 있는 결과만 알려주고 나머지는 제공되지 않는다고 하세요.
- 답에 "ASK:", "EXAM:", "TEST:" 같은 머리말을 붙이지 마세요.
- 증례에 없는 정보를 지어내지 마세요. 의사가 묻지 않은 정보를 먼저 말하지 마세요.
- 이전 대화와 모순되지 않게 답하세요.
{persona}
[증례 정보]
{case}"""

STRICT = """
엄격 규칙 (진료를 현실적으로 어렵게 만들기 위함):
- 한 번에 여러 가지를 물으면 **첫 번째 질문 하나에만** 답하세요.
- EXAM/TEST를 여러 개 한꺼번에 요청하면 **처음 적힌 하나의 결과만** 알려주고, 끝에 "(한 번에 하나씩 요청해 주세요)"를 붙이세요.
- 질문이 막연하면 막연하게 답하세요. 구체적으로 물어야 구체적인 사실을 말합니다.
"""

PERSONAS: dict[str, dict] = {
    "standard": {"label": "보통 환자", "strict": False, "prompt": "- 1~2문장으로 묻는 것에 성실히 답하세요.\n"},
    "vague": {
        "label": "모호한 환자",
        "strict": True,
        "prompt": """성격: 말수가 적고 표현이 모호한 환자
- "좀 아파요", "그냥 불편해요", "모르겠는데요"처럼 짧고 애매하게 답하세요.
- 위치, 시간, 정도는 의사가 구체적으로 물어볼 때만 알려주세요 (예: "몇 시간 전부터요?"라고 물으면 그때 답함).
""",
    },
    "anxious": {
        "label": "불안한 환자",
        "strict": True,
        "prompt": """성격: 걱정이 많고 말이 긴 환자
- 답하기 전후에 걱정을 섞으세요 (예: "혹시 암은 아니죠?", "인터넷에서 봤는데…").
- 핵심 정보는 장황한 이야기 속에 한 번만 섞어서 말하세요.
- 증례에 없는 증상을 새로 만들지는 마세요. 걱정만 표현하세요.
""",
    },
    "minimizer": {
        "label": "증상을 축소하는 환자",
        "strict": True,
        "prompt": """성격: 참을성이 많고 병원을 귀찮아하는 환자
- 증상을 가볍게 말하세요 ("별거 아니에요", "참을 만해요", "원래 가끔 그래요").
- 음주, 흡연, 약 복용, 성생활 같은 민감한 것은 처음엔 얼버무리고, 의사가 다시 구체적으로 물으면 사실대로 말하세요.
- 사실 자체를 거짓으로 바꾸지는 마세요. 정도만 축소해서 표현하세요.
""",
    },
    "poor_historian": {
        "label": "기억이 흐린 환자",
        "strict": True,
        "prompt": """성격: 나이가 많거나 경황이 없어 기억이 흐린 환자
- 시간과 순서를 흐릿하게 말하세요 ("며칠 됐나… 잘 기억이 안 나요").
- 먹는 약 이름은 모르고 모양이나 용도로만 말하세요 ("혈압약 같은 거", "하얀 알약").
- 의사가 선택지를 주거나 다시 물으면 좀 더 정확히 답하세요.
""",
    },
}
_PREFIX = re.compile(r"^(ASK|EXAM|TEST)\s*[:：]\s*")
HARD_PERSONAS = ["vague", "anxious", "minimizer", "poor_historian"]
PERSONA_CHOICES = ["standard", "mixed", *HARD_PERSONAS]


def resolve_persona(name: str, case_key: str) -> str:
    """'mixed' picks a hard persona deterministically per case, so reruns are comparable."""
    if name != "mixed":
        return name
    h = int(hashlib.md5(case_key.encode("utf-8")).hexdigest(), 16)
    return HARD_PERSONAS[h % len(HARD_PERSONAS)]


class LLMPatientEnvironment(Environment):
    def __init__(self, case: dict, llm: LLMClient, persona: str = "standard", max_history: int = 8):
        self.case = case
        self.llm = llm
        self.persona = resolve_persona(persona, case.get("initial", ""))
        p = PERSONAS[self.persona]
        visible = {k: v for k, v in case.items() if k not in HIDDEN_KEYS}
        self.system = BASE.format(
            persona=p["prompt"] + (STRICT if p["strict"] else ""),
            case=json.dumps(visible, ensure_ascii=False, indent=2),
        )
        self.history: list[dict] = []
        self.max_history = max_history

    def reset(self) -> Observation:
        self.history = []
        return Observation(self.case["initial"], info={"persona": self.persona})

    def step(self, action: Action) -> Observation:
        if action.type == ActionType.DIAGNOSE:
            return Observation("진단이 제출되었습니다.", done=True)
        user = {"role": "user", "content": f"{action.type.value}: {action.content}"}
        messages = [{"role": "system", "content": self.system}, *self.history[-2 * self.max_history:], user]
        try:
            text = _PREFIX.sub("", self.llm.chat(messages).strip()) or "잘 모르겠어요."
        except RuntimeError:
            text = "잘 모르겠어요."
        self.history += [user, {"role": "assistant", "content": text}]
        return Observation(text)
