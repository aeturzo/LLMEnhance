# backend/api/answerer_ctx.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

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
    "For factual attribute or numeric questions, copy the exact source phrase; "
    "do not paraphrase values, units, symbols, or dash characters.\n"
    "When you quote or rely on a span, add a citation like [doc_id]."
)

_CITATION_RE = re.compile(r"\[[^\[\]\n]{1,64}\]")
_DOC_FIELD_Q_RE = re.compile(
    r"^\s*according to\s+([A-Za-z0-9_.-]+),\s*what is the\s+(.+?)\?\s*$",
    re.IGNORECASE,
)
_INLINE_LABEL_RE = re.compile(r"([A-Za-z][A-Za-z0-9 /().,+&%_-]{1,80}):\s*")

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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def answerer_config() -> Dict[str, Any]:
    model = os.getenv("GEN_MODEL", "gpt-4o-mini")
    llm_disabled = _env_bool("LLM_DISABLED", False)
    if os.getenv("OPENAI_API_KEY"):
        configured_provider = "openai"
    elif os.getenv("OPENROUTER_API_KEY"):
        configured_provider = "openrouter"
    else:
        configured_provider = "none"
    return {
        "configured_provider": configured_provider,
        "configured_model": model,
        "llm_disabled": llm_disabled,
    }


def _extract_responses_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(resp, "output", None) or []
    chunks: List[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())
    return "\n".join(chunks).strip()


def _usage_payload(resp) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    prompt_tokens = (
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or 0
    )
    completion_tokens = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or 0
    )
    total_tokens = getattr(usage, "total_tokens", None) or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }

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


def _norm_field(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _split_inline_fields(line: str) -> List[Tuple[str, str]]:
    matches = []
    for m in _INLINE_LABEL_RE.finditer(line):
        if m.group(1).strip().lower() in {"http", "https"}:
            continue
        prefix = line[:m.start()].rstrip()
        if prefix and prefix[-1] == ":":
            continue
        matches.append(m)
    out: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        label = re.sub(r"\s+", " ", match.group(1)).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
        value = re.sub(r"\s+", " ", line[start:end]).strip(" \t-*;")
        if not label or not value:
            continue
        if "|" in value:
            value = value.split("|", 1)[0].strip()
        if value:
            out.append((label, value))
    return out


def _answer_simple_doc_field(question: str, passages: List[Dict]) -> str | None:
    m = _DOC_FIELD_Q_RE.match(question or "")
    if not m:
        return None
    source_id, field = m.group(1), m.group(2)
    field_key = _norm_field(field)
    candidates: List[Tuple[str, str]] = []
    for p in passages:
        pid = str(p.get("id") or "")
        doc_id = str(p.get("doc_id") or "")
        if source_id not in {pid, doc_id}:
            continue
        for raw_line in (p.get("text") or "").splitlines():
            for label, value in _split_inline_fields(raw_line):
                if _norm_field(label) == field_key:
                    candidates.append((value.strip(" .;"), pid or doc_id or source_id))
    distinct = []
    seen = set()
    for value, pid in candidates:
        key = _norm_field(value)
        if key and key not in seen:
            seen.add(key)
            distinct.append((value, pid))
    if len(distinct) != 1:
        return None
    value, pid = distinct[0]
    return f"{value} [{pid}]"

# ---------------------------------------------------------------------
# LLM call (robust + optional)
# ---------------------------------------------------------------------

def call_llm_detailed(prompt: str, max_tokens: int = 256) -> Tuple[str, Dict[str, Any]]:
    """
    Best-effort chat completion. Tries OpenAI first, then OpenRouter.
    Falls back to a deterministic message if no keys/SDK present.
    """
    if _env_bool("LLM_DISABLED", False):
        raise RuntimeError("LLM disabled by env")

    model = os.getenv("GEN_MODEL", "gpt-4o-mini")
    errors: List[str] = []
    # Try modern OpenAI SDK
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(
                api_key=api_key,
                timeout=_env_float("OPENAI_TIMEOUT", 60.0),
                max_retries=_env_int("OPENAI_MAX_RETRIES", 0),
            )
            try:
                if _env_bool("OPENAI_RESPONSES_DISABLED", False):
                    raise RuntimeError("Responses API disabled by env")
                resp = client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "low"},
                )
                text = _extract_responses_text(resp)
                if text:
                    return text, {
                        "llm_attempted": True,
                        "llm_used": True,
                        "provider": "openai",
                        "model": model,
                        "api": "responses",
                        "path": "llm",
                        **_usage_payload(resp),
                    }
            except Exception as exc:
                errors.append(f"openai.responses: {exc}")

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip(), {
                "llm_attempted": True,
                "llm_used": True,
                "provider": "openai",
                "model": model,
                "api": "chat_completions",
                "path": "llm",
                **_usage_payload(resp),
            }
    except Exception as exc:
        errors.append(f"openai.chat: {exc}")

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
            return (resp.choices[0].message.content or "").strip(), {
                "llm_attempted": True,
                "llm_used": True,
                "provider": "openrouter",
                "model": model,
                "api": "chat_completions",
                "path": "llm",
            }
    except Exception as exc:
        errors.append(f"openrouter.chat: {exc}")

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
            return (out["choices"][0]["message"]["content"] or "").strip(), {
                "llm_attempted": True,
                "llm_used": True,
                "provider": "openai",
                "model": model,
                "api": "legacy_chat_completions",
                "path": "llm",
            }
    except Exception as exc:
        errors.append(f"openai.legacy_chat: {exc}")

    # If we reach here, no network-backed LLM is available
    reason = "; ".join(errors) if errors else "No LLM backend available (no API key or SDK missing)"
    raise RuntimeError(reason)


def call_llm(prompt: str, max_tokens: int = 256) -> str:
    text, _ = call_llm_detailed(prompt, max_tokens=max_tokens)
    return text

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

def answer_with_context_detailed(question: str, passages: List[Dict]) -> Dict[str, Any]:
    """
    Build a strict prompt with the provided context and ask the LLM.
    If no LLM is available, return a safe fallback.
    """
    trace = {
        **answerer_config(),
        "llm_attempted": False,
        "llm_used": False,
        "provider": None,
        "model": None,
        "api": None,
        "path": "no_passages",
        "passage_count": len(passages or []),
    }

    if not passages:
        trace["reason"] = "No passages were provided."
        return {"answer": "Insufficient context.", "trace": trace}

    disable_pre_llm_direct = _env_bool(
        "AUTO_COMPOSE_DISABLE_PRE_LLM_DIRECT_FALLBACK",
        _env_bool("AUTO_COMPOSE_DISABLE_DIRECT_FALLBACK", False),
    )
    deterministic = None if disable_pre_llm_direct else _answer_simple_doc_field(question, passages)
    if deterministic:
        trace["path"] = "deterministic_doc_field"
        trace["reason"] = "Answered by exact field extraction from cited context."
        return {"answer": deterministic, "trace": trace}

    ctx = format_context(passages)
    prompt = f"Question: {question}\n\nContext:\n{ctx}\n\nAnswer:"

    try:
        out, llm_trace = call_llm_detailed(prompt, max_tokens=256)
        trace.update(llm_trace)
        if not out:
            trace["reason"] = "The LLM returned an empty response."
            return {"answer": "Insufficient context.", "trace": trace}
        # Enforce rule: only use provided context; if the model hedges, normalize.
        if "insufficient context" in out.lower():
            trace["path"] = "llm_insufficient_context"
            trace["reason"] = "The LLM reported insufficient context."
            return {"answer": "Insufficient context.", "trace": trace}
        return {"answer": _ensure_citation(out, passages), "trace": trace}
    except Exception as exc:
        if _env_bool("AUTO_COMPOSE_FAIL_ON_LLM_ERROR", False):
            trace["llm_attempted"] = (
                trace["configured_provider"] != "none" and not trace["llm_disabled"]
            )
            trace["path"] = "llm_error"
            trace["reason"] = str(exc)
            return {"answer": f"LLM_ERROR: {exc}", "trace": trace}
        # Fallback: return the top snippet with a citation
        top = passages[0]
        snippet = (top.get("text") or "")[:220].strip()
        pid = str(top.get("id") or "doc")
        trace["llm_attempted"] = (
            trace["configured_provider"] != "none" and not trace["llm_disabled"]
        )
        trace["path"] = "snippet_fallback"
        trace["reason"] = str(exc)
        trace["fallback_source_id"] = pid
        return {"answer": f"{snippet} [{pid}]", "trace": trace}


def answer_with_context(question: str, passages: List[Dict]) -> str:
    return answer_with_context_detailed(question, passages)["answer"]
