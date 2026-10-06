"""Specialist sub-agents: the same doctor LLM in a different role with a different evidence slice, called only when
triggered (see docs/architecture.md "Specialist sub-agents"). base = interface contract, runner = one guarded call,
orchestrator = triggers / caps / logging (one SubagentManager per case). Content modules (consult.py, advocate.py and
knowledge/specialty.py) are imported lazily by the orchestrator."""
