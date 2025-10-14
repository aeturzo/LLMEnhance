# backend/api/answerer_ctx.py
from __future__ import annotations

import os
import re
from typing import List, Dict

# ---------------------------------------------------------------------
# Configuration (via env vars)
# ---------------------------------------------------------------------
# OPENAI_API_KEY            -> use OpenAI
# OPENROUTER_API_KEY        -> use OpenRouter (OpenAI-compatible)
# GEN_MODEL                 -> chat model name (default: "gpt-4o-mini")
# MAX_CTX_CHARS             -> max total chars of concatenated context (default: 12000)
# MAX_PASSAGES              -> max number of passages to include (default: 6)
# LLM_DISABLED=1            -> force fallback (no network calls)
# ---------------------------------------------------------------------

SYSTEM = (
    "You must answer ONLY using the provided context.\n"
    "If the context is insufficient, respond exactly with: Insufficient context.\n"
    "When you quote or rely on a span, add a citation like [doc_id]."
)

_CITATION_RE = re.compile(r"\[[^\[\]\n]{1,64}\]")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y")

# ---------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------

def format_context(passages: List[Dict]) -> str:
    """
    Format retrieved passages into a compact, ID-tagged context block.
    Each passage:
      [<id>] <title>
      <text>
    """
    if not passages:
        return ""

    max_passages = _env_int("MAX_PASSAGES", 6)
    max_ctx_chars = _env_int("MAX_CTX_CHARS", 12_000)

    chunks: List[str] = []
    used = 0
    for p in passages[:max_passages]:
        pid = str(p.get("id") or "doc")
        title = str(p.get("title") or pid)
        text = (p.get("text") or "")[:2000]  # per-pass chunk cap
        block = f"[{pid}] {title}\n{text}".strip()
        # stop if we would exceed total budget
        if used + len(block) + 2 > max_ctx_chars:
            break
        chunks.append(block)
        used += len(block) + 2
    return "\n\n".join(chunks)

# ---------------------------------------------------------------------
# LLM call (robust + optional)
# ---------------------------------------------------------------------

def call_llm(prompt: str, max_tokens: int = 256) -> str:
    """
    Best-effort chat completion. Tries OpenAI first, then OpenRouter.
    Falls back to a deterministic message if no keys/SDK present.
    """
    if _env_bool("LLM_DISABLED", False):
        raise RuntimeError("LLM disabled by env")

    model = os.getenv("GEN_MODEL", "gpt-4o-mini")
    # Try modern OpenAI SDK
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
    except Exception:
        pass

    # Try OpenRouter (OpenAI-compatible)
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
    except Exception:
        pass

    # Legacy OpenAI sdk (pre-2024)
    try:
        import openai  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai.api_key = api_key
            out = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return (out["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        pass

    # If we reach here, no network-backed LLM is available
    raise RuntimeError("No LLM backend available (no API key or SDK missing)")

# ---------------------------------------------------------------------
# Answering with context
# ---------------------------------------------------------------------

def _ensure_citation(ans: str, passages: List[Dict]) -> str:
    """If the model forgot to cite, append the top doc id as a minimal cite."""
    if not ans or _CITATION_RE.search(ans):
        return ans
    if passages:
        pid = str(passages[0].get("id") or "doc")
        return f"{ans.strip()} [{pid}]"
    return ans

def answer_with_context(question: str, passages: List[Dict]) -> str:
    """
    Build a strict prompt with the provided context and ask the LLM.
    If no LLM is available, return a safe fallback.
    """
    if not passages:
        return "Insufficient context."

    ctx = format_context(passages)
    prompt = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"

    try:
        out = call_llm(prompt, max_tokens=256).strip()
        if not out:
            return "Insufficient context."
        # Enforce rule: only use provided context; if the model hedges, normalize.
        if "insufficient context" in out.lower():
            return "Insufficient context."
        return _ensure_citation(out, passages)
    except Exception:
        # Fallback: return the top snippet with a citation
        top = passages[0]
        snippet = (top.get("text") or "")[:220].strip()
        pid = str(top.get("id") or "doc")
        return f"{snippet} [{pid}]"



#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