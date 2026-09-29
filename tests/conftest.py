import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "tests")]

# Specialist sub-agents make extra LLM calls that would consume the scripted outputs of the older tests (which pin the
# call sequence). They are off by default in tests; tests/test_subagents.py switches them on explicitly
# (cfg.use_subagents = True). Set before any AgentConfig is built (its defaults read the environment).
os.environ["AGENT_USE_SUBAGENTS"] = "0"


def pytest_configure(config):
    # Wall-clock tests use tests/perf.py (best-of-N + load-scaled slack; PERF_STRICT=1 for the strict budgets).
    config.addinivalue_line("markers", "perf: wall-clock latency/size budget (PERF_STRICT=1 for strict limits; "
                                       "-m 'not perf' to skip)")
