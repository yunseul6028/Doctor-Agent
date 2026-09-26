"""Safety rules. Content is owned by clinical-strategist.

Maps chief-complaint keywords to dangerous differentials that must be ruled out. If a DIAGNOSE comes before they are checked, add a hint to the policy.
Now backed by safety/protocols.py (per-category can't-miss diagnoses + minimum checks, with citations).
"""
from doctor_agent.knowledge.clinical_rules import CATEGORY_KEYWORDS
from doctor_agent.safety.protocols import PROTOCOLS, cant_miss_for

# Kept for backward compatibility: "|"-separated synonyms → can't-miss diagnoses (derived from PROTOCOLS)
RED_FLAGS: dict[str, list[str]] = {
    "|".join(CATEGORY_KEYWORDS[p.category]): list(p.cant_miss) for p in PROTOCOLS
}


def red_flags_for(text: str) -> list[str]:
    return cant_miss_for(text)
