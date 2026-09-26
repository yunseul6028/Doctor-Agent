"""Clean up gpt-oss (harmony format) output that leaks into `message.content`.

gpt-oss writes an `analysis` channel (chain of thought) and a `final` channel (the answer). A correctly configured
server (vLLM/OpenAI-compatible) puts the analysis in `message.reasoning_content` / `message.reasoning` and only the
final text in `content`, but misconfigured parsers leak the raw markers, e.g.

    <|start|>assistant<|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>{...}<|return|>

or, with special tokens stripped by the detokenizer: ``analysis...assistantfinal{...}``.
"""
import re

_TOKEN = re.compile(r"<\|(start|end|message|channel|return|call|constrain|endoftext|startoftext)\|>")
_SEGMENT = re.compile(r"<\|channel\|>\s*(analysis|commentary|final)(?:(?!<\|message\|>|<\|channel\|>).)*<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>|<\|channel\|>|$)", re.S)
_STRIPPED_PREFIX = re.compile(r"^\s*analysis", re.I)


def split_harmony(text: str | None) -> tuple[str, str]:
    """(final_text, analysis_text) from a possibly harmony-leaked string. Plain text comes back unchanged as final."""
    text = text or ""
    if "<|" in text and "|>" in text:
        segments = _SEGMENT.findall(text)
        if segments:
            final = [body for ch, body in segments if ch == "final"]
            analysis = [body for ch, body in segments if ch != "final"]
            if final:
                return _TOKEN.sub("", final[-1]).strip(), "\n".join(_TOKEN.sub("", a).strip() for a in analysis)
            # only analysis/commentary channels: no final answer (e.g. cut off by max_tokens)
            return "", "\n".join(_TOKEN.sub("", a).strip() for a in analysis)
        return re.sub(r"\s*<\|[a-z_]+\|>\s*", "\n", text).strip(), ""
    if _STRIPPED_PREFIX.match(text) and (idx := text.lower().rfind("assistantfinal")) != -1:
        return text[idx + len("assistantfinal"):].strip(), _STRIPPED_PREFIX.sub("", text[:idx]).strip()
    return text.strip(), ""


def clean_content(text: str | None) -> str:
    return split_harmony(text)[0]
