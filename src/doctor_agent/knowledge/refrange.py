"""One reader for printed reference ranges, shared by nlp/findings.py and knowledge/kb_tests.py (stdlib only).

A reference parenthesis after a measured value is read by how reference ranges are written, not by test:

    range            "(정상 0.4-4.0)", "(참고치: 13~60 U/L)"                  lo=0.4 hi=4.0 (both inclusive)
    upper limit      "(<500)", "(≤40)", "(정상 40 미만)", "(참고치 40 이하)", "(500미만)", "(less than 40)",
                     "(up to 40)", "(정상 상한 60)", "(상한치: 45 U/L)", "(ULN 60)", "(upper limit 500)"
    lower limit      "(>60)", "(≥12)", "(정상 12 이상)", "(60 초과)", "(above 12)", "(at least 12)",
                     "(정상 하한 12)", "(LLN 12)"
    bare limit       "(정상치 500)", "(참고치: 60)": one number under a reference label with no comparator; which side
                     it limits is not printed, so the caller decides from the analyte (ref.limit)
    direction words  "(정상 범위 초과)", "(경미한 상승)", "(mildly elevated)", "(감소)", "(H)", "(정상)", "(상승 없음)"

Strict limits ("<", "미만", "less than", "below"; ">", "초과", "above") put the limit value itself outside the normal
range; inclusive ones ("≤", "이하", "up to", "상한"; "≥", "이상", "at least", "하한") put it inside. A body that holds a
number the reader cannot place as a limit ("(정상 상한의 5배)", "(3백분위수 미만)") says nothing: its words are not read
as "normal". Multiples ("5배", "3x") are not limits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = r"(\d{1,3}(?:,\d{3})+(?![\d.])|\d+(?:\.\d+)?)"
UNIT = r"(%|[a-zμµ/.³^]+)"
_U = r"\s*" + UNIT + r"?"
# a number that is a multiple ("5배", "3x", "2 times"), a titre part or glued to a name ("s1", "c3") is not a limit
_NOT_LIMIT_AFTER = r"(?!\d|\.\d|\s*(?:배|x(?![a-z])|×|times|fold))"
_FREE = r"(?<![a-z\d.,])"

RANGE = re.compile(_FREE + _NUM + r"\s*[-~–]\s*" + _NUM + _U)
# upper limits: comparator / word before the number, or word after it; group "s" marks a strict limit
_UP_BEFORE = re.compile(
    r"(?:(?P<s>(?<![<>=])<(?!=)|less than|lower than|(?<![a-z])below|(?<![a-z])under)|≤|<=|=<|up to|upto|(?<![a-z])max(?:imum)?"
    r"|최대|(?:정상\s*)?상한(?:치|값)?(?:\s*(?:은|는|이|:|=))?|upper (?:reference )?limit(?: of normal)?|(?<![a-z])uln(?![a-z]))"
    r"\s*[:=]?\s*" + _NUM + _NOT_LIMIT_AFTER + _U)
_UP_AFTER = re.compile(_FREE + _NUM + _NOT_LIMIT_AFTER + _U + r"\s*(?:(?P<s>미만|under|below)|이하|or (?:less|below|lower))")
_LO_BEFORE = re.compile(
    r"(?:(?P<s>(?<![<>=])>(?!=)|greater than|more than|higher than|(?<![a-z])above|(?<![a-z])over)|≥|>=|=>|at least"
    r"|(?<![a-z])min(?:imum)?(?![a-z])|최소|(?:정상\s*)?하한(?:치|값)?(?:\s*(?:은|는|이|:|=))?"
    r"|lower (?:reference )?limit(?: of normal)?|(?<![a-z])lln(?![a-z]))"
    r"\s*[:=]?\s*" + _NUM + _NOT_LIMIT_AFTER + _U)
_LO_AFTER = re.compile(_FREE + _NUM + _NOT_LIMIT_AFTER + _U + r"\s*(?:(?P<s>초과|over|above)|이상|or (?:more|greater|higher|above))")
# one number right after a reference label, with no comparator: a limit whose side is not printed
_BARE = re.compile(r"(?:정상|참고|기준|normal|ref(?:erence)?)\s*(?:범위|수치|치|값|range|value|limit)?\s*[:=]?\s*" + _NUM
                   + _NOT_LIMIT_AFTER + _U + r"\s*$")
# a number standing on its own (not glued to a name like "s1" / "c3", not a multiple)
_FREE_NUM = re.compile(_FREE + r"\d+(?:\.\d+)?" + _NOT_LIMIT_AFTER)

# direction words of a parenthesis without a placed limit
NEG = re.compile(r"없|않|(?<![a-z])no(?![a-z])|(?<![a-z])not(?![a-z])")
HIGH = re.compile(r"초과|높|상승|상회|증가|항진|above|high|exceed|elevat|increas|raised|(?<![a-z])over(?![a-z])|↑")
LOW = re.compile(r"미만|낮|저하|감소|하회|결핍|부족|below|(?<![a-z])low|decreas|reduc|under|↓")
OUT = re.compile(r"벗어|이상|abnormal|out of|outside")
IN = re.compile(r"정상|normal|범위\s*내|within|wnl")
_FLAG = re.compile(r"^\s*(?:(?P<h>h|hh)|(?P<l>l|ll))\s*[*!]?\s*$")


@dataclass
class Ref:
    lo: float | None = None
    hi: float | None = None
    lo_strict: bool = False  # "> 60", "60 초과": 60 itself is below the normal range
    hi_strict: bool = False  # "< 500", "500 미만": 500 itself is above the normal range
    unit: str = ""           # unit printed with the limits ("" when none is printed), as written
    limit: float | None = None  # a bare limit ("정상치 500"): the side is the analyte's abnormal side
    says: str = ""           # "normal" / "high" / "low" / "abnormal" from words, when no limit is placed

    @property
    def bounded(self) -> bool:
        return self.lo is not None or self.hi is not None

    @property
    def empty(self) -> bool:
        return not self.bounded and self.limit is None and not self.says


def _f(s: str) -> float:
    return float(s.replace(",", ""))


def read(body: str) -> Ref:
    """Read the body of one reference parenthesis (text inside the brackets)."""
    b = (body or "").lower().replace("µ", "μ")
    ref = Ref()
    r = RANGE.search(b)
    if r:
        ref.lo, ref.hi, ref.unit = _f(r.group(1)), _f(r.group(2)), r.group(3) or ""
        return ref
    for rx_list, side in (((_UP_BEFORE, _UP_AFTER), "hi"), ((_LO_BEFORE, _LO_AFTER), "lo")):
        for rx in rx_list:
            m = rx.search(b)
            if m:
                v, unit = _f(m.group(1) if rx in (_UP_AFTER, _LO_AFTER) else m.group(2)), \
                    (m.group(2) if rx in (_UP_AFTER, _LO_AFTER) else m.group(3)) or ""
                if side == "hi":
                    ref.hi, ref.hi_strict = v, bool(m.group("s"))
                else:
                    ref.lo, ref.lo_strict = v, bool(m.group("s"))
                ref.unit = ref.unit or unit
                break
    if ref.bounded:
        return ref
    m = _BARE.search(b)
    if m:
        ref.limit, ref.unit = _f(m.group(1)), m.group(2) or ""
        return ref
    if _FREE_NUM.search(b):
        return ref  # a number that is not a limit: the words around it say nothing
    fl = _FLAG.match(b)
    if fl:
        ref.says = "high" if fl.group("h") else "low"
    elif NEG.search(b):
        ref.says = "normal" if (HIGH.search(b) or LOW.search(b) or OUT.search(b) or IN.search(b)) else ""
    elif HIGH.search(b) and LOW.search(b):
        ref.says = "abnormal"
    elif HIGH.search(b):
        ref.says = "high"
    elif LOW.search(b):
        ref.says = "low"
    elif OUT.search(b):
        ref.says = "abnormal"
    elif IN.search(b) and not re.search(r"\d", b):  # "(정상 상한의 5배)" is not a normal statement
        ref.says = "normal"
    return ref


def above(v: float, hi: float, strict: bool, mult: float = 1.0) -> bool:
    """Is v above an upper limit (× mult)? A strict limit ("< 500") counts the limit itself as above."""
    lim = hi * mult
    return v > lim or (strict and mult == 1.0 and v == lim)


def below(v: float, lo: float, strict: bool) -> bool:
    return v < lo or (strict and v == lo)


def direction(ref: Ref, v: float, side: str = "") -> str:
    """"high" / "low" / "normal" / "" for a value printed in the reference's unit. side ("high" | "low" | ""): the
    analyte's only abnormal side, used for a bare limit and for "abnormal" words; "" when both sides are possible."""
    if ref.bounded:
        if ref.hi is not None and above(v, ref.hi, ref.hi_strict):
            return "high"
        if ref.lo is not None and below(v, ref.lo, ref.lo_strict):
            return "low"
        return "normal"
    if ref.limit is not None:
        if side == "high":
            return "high" if v > ref.limit else "normal"
        if side == "low":
            return "low" if v < ref.limit else "normal"
        return ""
    if ref.says == "abnormal":
        return side
    return ref.says


def unit_gap(v: float, bound: float) -> bool:
    """A value two orders of magnitude away from the printed limit: when only one of them carries a unit, the two
    may be in different units ("D-dimer 1.2 (정상 <500 ng/mL)" is 1.2 μg/mL), so the printed comparison is not
    trusted on its own."""
    v, bound = abs(v), abs(bound)
    return bound > 0 and (v * 100 < bound or v > bound * 100)
