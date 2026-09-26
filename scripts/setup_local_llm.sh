#!/usr/bin/env bash
# Runs gpt-oss-20b locally with Ollama (dev only; the evaluation server provides its own fixed LLM).
# 16GB Mac: ~13GB model, so close other apps before running. Q: is the Ollama build the same revision as the official HF one? → recheck against the official API before submitting.
set -euo pipefail
command -v ollama >/dev/null || brew install ollama
pgrep -x ollama >/dev/null || (ollama serve >/tmp/ollama.log 2>&1 &) && sleep 2
ollama pull gpt-oss:20b
echo "export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=gpt-oss:20b LLM_API_KEY=ollama"
