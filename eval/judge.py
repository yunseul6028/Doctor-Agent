"""LLM-judge diagnosis scoring. Owned by eval-simulator. Not included in the submission ZIP."""
import json
import re

from doctor_agent.llm.client import LLMClient

PROMPT = """당신은 의료 진단 채점자입니다. 예측 진단이 정답과 일치하는지 판정하세요.

정답 진단: {answer}
인정하는 다른 이름: {aliases}
예측 진단: {pred}

기준: 1.0 = 같은 질환 (동의어, 한국어/영어 표기 차이 인정), 0.5 = 같은 계열이지만 덜 구체적이거나 부분적으로 맞음, 0.0 = 다른 질환
JSON 한 줄로만 출력하세요. 이유는 한국어 한 문장으로: {{"score": 0.0, "reason": "..."}}"""


def judge_diagnosis(llm: LLMClient, case: dict, pred: str | None) -> dict:
    if not pred:
        return {"score": 0.0, "reason": "진단 없음"}
    msg = PROMPT.format(answer=case["diagnosis"], aliases=", ".join(case.get("aliases", [])) or "없음", pred=pred)
    try:
        raw = llm.chat([{"role": "user", "content": msg}])
        obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        return {"score": float(obj["score"]), "reason": str(obj.get("reason", ""))}
    except (RuntimeError, AttributeError, ValueError, KeyError, json.JSONDecodeError) as e:
        return {"score": None, "reason": f"채점 실패: {e}"}
