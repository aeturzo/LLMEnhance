from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.answerer_ctx import _answer_simple_doc_field, answer_with_context_detailed, answerer_config
from backend.api.solve import _attach_confidence, _pick_query, _retrieve_context, _safe_features, _snippet
from backend.services import memory_service
from backend.services.carbon_query_service import is_carbon_query, solve_carbon_query
from backend.services.symbolic_reasoning_service import answer_symbolic, sym_fire_flags

router = APIRouter()
SUPPORTED_AUTO_DOMAINS = ("auto", "battery", "lexmark", "viessmann")


class SolveAutoRequest(BaseModel):
    query: Optional[str] = Field(default=None)
    q: Optional[str] = Field(default=None)
    product: Optional[str] = None
    domain: Optional[str] = "auto"
    session: Optional[str] = "frontend-session"
    top_k_search: Optional[int] = 4
    top_k_memory: Optional[int] = 3


def _is_memory_like_query(query: str) -> bool:
    q = (query or "").lower()
    phrases = (
        "did i say",
        "what did i say",
        "what did i tell",
        "remember",
        "remind me",
        "preferred packaging",
        "preferred supplier",
        "my note",
        "our note",
    )
    if any(phrase in q for phrase in phrases):
        return True
    if ("packaging" in q or "supplier" in q) and any(token in q for token in ("prefer", "prefers", "preferred")):
        return True
    if q.startswith("what packaging") or q.startswith("which packaging"):
        return True
    return False


def _derive_memory_answer(query: str, memory_text: str, passage_id: str) -> Optional[str]:
    q = (query or "").lower()
    text = (memory_text or "").strip()
    if not text:
        return None

    packaging_match = re.search(
        r"preferred packaging is\s+(.*?)(?:\s+and\s+the\s+preferred supplier is|[.]+$|$)",
        text,
        flags=re.IGNORECASE,
    )
    supplier_match = re.search(
        r"preferred supplier is\s+(.*?)(?:[.]+$|$)",
        text,
        flags=re.IGNORECASE,
    )
    product_match = re.search(r"for\s+([A-Za-z0-9_-]+)", text, flags=re.IGNORECASE)
    product_label = product_match.group(1) if product_match else "the product"

    if "packaging" in q and packaging_match:
        packaging = packaging_match.group(1).strip(" .")
        return f"The preferred packaging for {product_label} is {packaging} [{passage_id}]"

    if "supplier" in q and supplier_match:
        supplier = supplier_match.group(1).strip(" .")
        return f"The preferred supplier for {product_label} is {supplier} [{passage_id}]"

    if packaging_match and ("prefer" in q or "preferred" in q or "prefers" in q):
        packaging = packaging_match.group(1).strip(" .")
        return f"The preferred packaging for {product_label} is {packaging} [{passage_id}]"

    return None


def _infer_product_name(query: str) -> Optional[str]:
    text = (query or "").strip()
    if not text:
        return None

    lexmark_match = re.search(r"\blexmark\s+mx431adn\b", text, flags=re.IGNORECASE)
    if lexmark_match:
        return "lexmark_mx431adn"

    direct_match = re.search(r"\b(ProductV\d+|Product[A-Za-z0-9_-]+|PrinterL\d+)\b", text)
    if direct_match:
        return direct_match.group(1)

    return None


def _normalize_auto_domain(domain: Optional[str]) -> str:
    dom = (domain or "auto").strip().lower()
    return dom if dom in SUPPORTED_AUTO_DOMAINS else "auto"


def _infer_domain(query: str, product: Optional[str], selected_domain: Optional[str]) -> str:
    chosen = _normalize_auto_domain(selected_domain)
    if chosen != "auto":
        return chosen

    product_name = (product or "").strip()
    if re.match(r"^PrinterL\d+$", product_name):
        return "lexmark"
    if re.match(r"^ProductV\d+$", product_name):
        return "viessmann"
    if re.match(r"^Product[A-Za-z0-9_-]+$", product_name):
        return "battery"

    text = (query or "").lower()
    if "viessmann" in text or re.search(r"\bproductv\d+\b", text):
        return "viessmann"
    if "lexmark" in text or re.search(r"\bprinterl\d+\b", text):
        return "lexmark"
    return "battery"


def _is_compliance_like_query(query: str) -> bool:
    q = (query or "").lower()
    phrases = (
        "compliance",
        "standard",
        "standards",
        "requirements",
        "apply to",
        "applies to",
    )
    return any(phrase in q for phrase in phrases)


def _is_document_like_query(query: str, product: Optional[str]) -> bool:
    text = f"{query or ''} {product or ''}".lower()
    if re.search(r"\b(?:battery|lexmark|viessmann)_seed_\d{4}\b", text):
        return True
    phrases = (
        "passport",
        "recorded in",
        "documented in",
        "stated in",
        "from the",
        "together with the",
        "dpp issuer",
        "gtin",
        "manufacture date",
        "declaration of conformity",
    )
    return any(phrase in text for phrase in phrases)


def _derive_symbolic_answer(query: str, symbolic_text: str, passage_id: str, product_name: Optional[str]) -> Optional[str]:
    q = (query or "").lower()
    text = (symbolic_text or "").strip()
    if not text:
        return None

    match = re.search(
        r"(?:standards?|compliance)\s+for\s+[^:]+:\s*(.*?)(?:\.\s*required steps:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    raw_items = [item.strip(" .") for item in match.group(1).split(",")]
    preferred_tokens = ("EN ", "IEC", "ISO", "RoHS", "REACH", "UN ", "WEEE", "CE")
    standards = [
        item
        for item in raw_items
        if item and any(token.lower() in item.lower() for token in preferred_tokens)
    ]
    if not standards:
        standards = [item for item in raw_items if item]
    if not standards:
        return None

    label = product_name or "the product"
    if "two" in q and len(standards) >= 2:
        return f"Two compliance standards for {label} are {standards[0]} and {standards[1]} [{passage_id}]"
    return f"One compliance standard for {label} is {standards[0]} [{passage_id}]"


def _answer_mentions_memory_content(answer: str) -> bool:
    lowered = (answer or "").lower()
    tokens = ("packaging", "supplier", "corrugated", "greencells")
    return any(token in lowered for token in tokens)


def _answer_mentions_compliance_content(answer: str) -> bool:
    lowered = (answer or "").lower()
    tokens = ("standard", "compliance", "[sym:", "en ", "iec", "rohs", "reach", "un ")
    return any(token in lowered for token in tokens)


def _compact_answer_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _post_llm_trace(trace: dict[str, Any], path: str, reason: str, source_id: str) -> dict[str, Any]:
    corrected = {
        **trace,
        "path": path,
        "reason": reason,
        "post_llm_correction": True,
        "fallback_source_id": source_id,
    }
    if path in {
        "llm_then_memory_direct",
        "llm_then_memory_symbolic_direct",
        "llm_then_symbolic_direct",
    }:
        # The LLM was attempted, but its answer was discarded and replaced by
        # deterministic evidence. Keep llm_attempted while reporting the final
        # answer provenance accurately.
        corrected["llm_used"] = False
        corrected["path"] = path.removeprefix("llm_then_")
    return corrected


def _symbolic_direct_values(symbolic_text: str, query: str = "") -> list[str]:
    standards: list[str] = []
    steps: list[str] = []
    components: list[str] = []
    misc: list[str] = []
    for raw_line in (symbolic_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("evidence:"):
            continue
        symbolic_match = re.search(
            r"(?:standards?|compliance)\s+for\s+[^:]+:\s*(.*?)(?:\.\s*required steps:|$)",
            line,
            flags=re.IGNORECASE,
        )
        if symbolic_match:
            tail = symbolic_match.group(1)
            standards.extend(part.strip(" .") for part in tail.split(",") if part.strip(" ."))

        steps_match = re.search(r"required steps:\s*(.*)$", line, flags=re.IGNORECASE)
        if steps_match:
            tail = steps_match.group(1)
            steps.extend(part.strip(" .") for part in tail.split(",") if part.strip(" ."))
            continue

        if line.lower().startswith("components:") and ":" in line:
            tail = line.split(":", 1)[1]
            components.extend(part.strip(" .") for part in tail.split(",") if part.strip(" ."))
        elif not symbolic_match and "|" not in line:
            misc.append(line.strip(" ."))
    m = re.search(r"Evidence:\s*(.*)", symbolic_text or "", flags=re.IGNORECASE | re.DOTALL)
    if m:
        for triple in m.group(1).split(";"):
            if "|" in triple:
                pred = triple.split("|", 2)[1].strip().lower() if triple.count("|") >= 2 else ""
                obj = triple.rsplit("|", 1)[-1].strip(" .")
                if not obj:
                    continue
                if "step" in pred:
                    steps.append(obj)
                elif "component" in pred:
                    components.append(obj)
                elif "compliance" in pred or "conform" in pred or "standard" in pred:
                    standards.append(obj)
                else:
                    misc.append(obj)

    q = (query or "").lower()
    if "step" in q or "verification" in q or "test" in q:
        values = steps or misc
    elif "component" in q:
        values = components or misc
    elif "standard" in q or "compliance" in q or "conforms" in q:
        values = standards or misc
    else:
        values = standards + steps + components + misc

    distinct: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _compact_answer_text(value)
        if key and key not in seen:
            seen.add(key)
            distinct.append(value)
    return distinct


def _answer_mentions_symbolic_value(answer: str, symbolic_text: str, query: str = "") -> bool:
    compact_answer = _compact_answer_text(answer)
    return any(_compact_answer_text(value) in compact_answer for value in _symbolic_direct_values(symbolic_text, query))


def _symbolic_query_direct_answer(
    query: str,
    symbolic_text: str,
    passage_id: str,
    product_name: Optional[str],
) -> Optional[str]:
    values = _symbolic_direct_values(symbolic_text, query)
    if not values:
        return None
    label = product_name or "the product"
    q = (query or "").lower()
    if "step" in q or "verification" in q or "test" in q:
        return f"One required step for {label} is {values[0]} [{passage_id}]"
    if "component" in q:
        return f"One documented component of {label} is {values[0]} [{passage_id}]"
    if "standard" in q or "compliance" in q or "conforms" in q:
        return f"One compliance standard for {label} is {values[0]} [{passage_id}]"
    return f"{values[0]} [{passage_id}]"


def _compound_doc_field_answer(query: str, passages: list[dict[str, Any]]) -> Optional[str]:
    q = query or ""
    patterns = (
        r"\b(?:and|with|together with)\s+the\s+(.+?)\s+(?:recorded in|documented in|stated in)\s+(?:the\s+)?((?:battery|lexmark|viessmann)_seed_\d{4})",
        r"\b(?:and|with|together with)\s+the\s+(.+?)\s+from\s+(?:the\s+)?((?:battery|lexmark|viessmann)_seed_\d{4})\s+(?:passport|record)",
        r"\bthe\s+(.+?)\s+(?:recorded|documented|stated)\s+in\s+(?:its\s+product\s+passport|the\s+product\s+passport)",
    )
    source_id = None
    label = None
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            label = match.group(1).strip(" .")
            if len(match.groups()) >= 2:
                source_id = match.group(2).strip()
            break
    if label and not source_id:
        ids = re.findall(r"\b(?:battery|lexmark|viessmann)_seed_\d{4}\b", q, flags=re.IGNORECASE)
        if len(set(ids)) == 1:
            source_id = ids[0]
    if not label or not source_id:
        return None
    simple_q = f"According to {source_id}, what is the {label}?"
    return _answer_simple_doc_field(simple_q, passages)


def _memory_passages(query: str, session: str, top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    hits: List[Any] = []
    try:
        hits = memory_service.retrieve(session_id=session, query=query, top_k=top_k)
    except Exception:
        hits = []

    top_score = None
    if hits:
        try:
            top_score = float(getattr(hits[0], "score", 0.0) or 0.0)
        except Exception:
            top_score = None

    step = {
        "source": "MEM",
        "score": top_score,
        "session": session,
        "hit_count": len(hits),
        "included": False,
    }
    passages: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, hit in enumerate(hits[:top_k], start=1):
        passage = {
            "id": f"mem:{session}:{index}",
            "title": f"Memory {index}",
            "text": hit.content,
            "score": getattr(hit, "score", None),
            "source": "memory",
            "domain": "memory",
        }
        passages.append(passage)
        sources.append({
            "type": "memory",
            "id": passage["id"],
            "title": passage["title"],
            "score": getattr(hit, "score", None),
            "snippet": _snippet(hit.content),
        })
    return passages, step, sources


def _symbolic_passage(query: str, product: Optional[str], session: str, domain: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    product_name = (product or "").strip()
    step = {
        "source": "SYM",
        "product": product_name or None,
        "domain": domain,
        "included": False,
        "fired": False,
    }
    if not product_name:
        step["reason"] = "No product supplied for symbolic reasoning."
        return [], step, []

    sym = None
    try:
        if sym_fire_flags(query, product_name, domain=domain):
            sym = answer_symbolic(query, product_name, session, domain=domain)
    except Exception:
        sym = None

    if not sym or not getattr(sym, "text", ""):
        step["reason"] = f"No symbolic answer produced for domain '{domain}'."
        return [], step, []

    evidence = getattr(sym, "evidence", None) or []
    evidence_text = "; ".join(f"{s} | {p} | {o}" for (s, p, o) in evidence[:6])
    text = getattr(sym, "text", "").strip()
    if evidence_text:
        text = f"{text}\nEvidence: {evidence_text}"

    passage = {
        "id": f"sym:{product_name}",
        "title": f"Symbolic reasoning for {product_name}",
        "text": text,
        "score": 1.0 if getattr(sym, "proved", False) else 0.75,
        "source": "symbolic",
        "domain": domain,
    }
    step.update({
        "included": False,
        "fired": bool(getattr(sym, "fired", True)),
        "proved": getattr(sym, "proved", None),
        "refuted": getattr(sym, "refuted", None),
        "sym_trace": {
            "product": product_name,
            "evidence_count": len(evidence),
            "triples": [f"{s} | {p} | {o}" for (s, p, o) in evidence[:6]],
        },
    })
    source = {
        "type": "symbolic",
        "id": passage["id"],
        "title": passage["title"],
        "score": passage["score"],
        "snippet": _snippet(text),
    }
    return [passage], step, [source]


def _search_passages(query: str, product: Optional[str], top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    passages = _retrieve_context(query, product, top_k=top_k)
    top_score = None
    if passages:
        try:
            top_score = float(passages[0].get("score"))
        except Exception:
            top_score = None
    step = {
        "source": "SEARCH",
        "score": top_score,
        "k": len(passages),
        "doc_ids": [p.get("id") for p in passages[: min(5, len(passages))]],
        "included": False,
    }
    sources = []
    for passage in passages[: min(5, len(passages))]:
        sources.append({
            "type": "search",
            "id": passage.get("id"),
            "title": passage.get("title"),
            "score": passage.get("score"),
            "snippet": _snippet(passage.get("text", "")),
        })
    return passages, step, sources


def _symbolic_direct_answer(
    query: str,
    sym_passages: list[dict[str, Any]],
    product_name: Optional[str],
) -> Optional[str]:
    if not sym_passages:
        return None

    direct = _derive_symbolic_answer(
        query,
        sym_passages[0]["text"],
        sym_passages[0]["id"],
        product_name,
    )
    if direct:
        return direct

    text = (sym_passages[0].get("text") or "").strip()
    if text:
        return f"{text} [{sym_passages[0]['id']}]"
    return None


def solve_auto_query(
    query: str,
    product: Optional[str] = None,
    domain: str = "auto",
    session: str = "frontend-session",
    top_k_search: int = 4,
    top_k_memory: int = 3,
) -> Dict[str, Any]:
    text = (query or "").strip()
    explicit_product = (product or "").strip()
    inferred_product = _infer_product_name(text) if not explicit_product else None
    product_name = explicit_product or inferred_product
    product_name = product_name.strip() if product_name else None
    selected_domain = _normalize_auto_domain(domain)
    effective_domain = _infer_domain(text, product_name, selected_domain)
    if not text:
        raise HTTPException(status_code=422, detail="Missing query/q")

    if is_carbon_query(text, mode="AUTO_COMPOSE"):
        return solve_carbon_query(query=text, product=product_name, session=session)

    feats = _safe_features(text, product_name, session)
    memory_like = _is_memory_like_query(text)

    mem_passages, mem_step, mem_sources = _memory_passages(text, session, max(1, int(top_k_memory or 3)))
    sym_passages, sym_step, sym_sources = _symbolic_passage(text, product_name, session, effective_domain)
    search_passages, search_step, search_sources = _search_passages(text, product_name, max(1, int(top_k_search or 4)))
    feats["sym_fired"] = int(bool(sym_step.get("fired")))

    mem_score = float(mem_step.get("score") or 0.0)
    search_score = float(search_step.get("score") or 0.0)
    mem_threshold = 0.35
    search_threshold = 0.50
    memory_dominant = bool(mem_passages) and mem_score >= mem_threshold and (
        mem_score >= search_score + 0.10 or (mem_score >= 0.60 and search_score < 0.50)
    )

    include_mem = bool(mem_passages) and (memory_like or mem_score >= mem_threshold)
    include_sym = bool(sym_passages)
    include_search = bool(search_passages)
    compliance_like = _is_compliance_like_query(text)
    document_like = _is_document_like_query(text, product_name)

    if (memory_like or memory_dominant) and include_mem:
        include_search = bool(search_passages) if document_like else search_score >= search_threshold
    elif memory_like and not include_mem:
        include_search = bool(search_passages) if document_like else bool(product_name) and search_score >= search_threshold
    elif include_sym and include_search:
        include_search = True if document_like else search_score >= 0.35

    if include_sym and compliance_like:
        include_search = bool(search_passages) if document_like else False
        if not memory_like and not memory_dominant:
            include_mem = False

    if not include_mem and not include_sym and not include_search and search_passages:
        include_search = True

    mem_step["included"] = include_mem
    if include_mem:
        mem_step["reason"] = "Memory matched the question strongly enough to contribute context."
    elif mem_passages:
        mem_step["reason"] = "Memory was checked but not included because the query looked better served by other evidence."
    else:
        mem_step["reason"] = "No memory entries matched this session."

    sym_step["included"] = include_sym
    if include_sym:
        sym_step["reason"] = "Symbolic reasoning produced domain evidence for the requested product."

    search_step["included"] = include_search
    if include_search:
        search_step["reason"] = "Search evidence was included in the composed context."
    elif search_passages:
        search_step["reason"] = "Search evidence was checked but filtered out to avoid irrelevant context."
    else:
        search_step["reason"] = "No search passages were retrieved."

    ordered_passages: list[dict[str, Any]] = []
    ordered_sources: list[dict[str, Any]] = []
    if memory_like or memory_dominant:
        if include_mem:
            ordered_passages.extend(mem_passages)
            ordered_sources.extend(mem_sources)
        if include_sym:
            ordered_passages.extend(sym_passages)
            ordered_sources.extend(sym_sources)
        if include_search:
            ordered_passages.extend(search_passages)
            ordered_sources.extend(search_sources)
    else:
        if include_sym:
            ordered_passages.extend(sym_passages)
            ordered_sources.extend(sym_sources)
        if include_search:
            ordered_passages.extend(search_passages)
            ordered_sources.extend(search_sources)
        if include_mem:
            ordered_passages.extend(mem_passages)
            ordered_sources.extend(mem_sources)

    steps = [mem_step, sym_step, search_step, {
        "source": "COMPOSE",
        "included_passage_ids": [passage.get("id") for passage in ordered_passages],
        "query_features": feats,
        "memory_like_query": memory_like,
        "memory_dominant": memory_dominant,
        "document_like_query": document_like,
        "session": session,
        "selected_domain": selected_domain,
        "effective_domain": effective_domain,
        "effective_product": product_name,
        "product_inferred_from_query": inferred_product,
    }]

    answer_trace = {
        **answerer_config(),
        "llm_attempted": False,
        "llm_used": False,
        "provider": None,
        "model": None,
        "api": None,
        "path": "no_passages",
        "passage_count": len(ordered_passages),
    }

    if ordered_passages:
        answer_result = answer_with_context_detailed(text, ordered_passages)
        answer = answer_result["answer"]
        answer_trace = answer_result["trace"]
        llm_failed = str(answer_trace.get("path") or "") in {"llm_error", "snippet_fallback"}
        memory_answer = _derive_memory_answer(text, mem_passages[0]["text"], mem_passages[0]["id"]) if include_mem else None
        symbolic_answer = (
            _derive_symbolic_answer(text, sym_passages[0]["text"], sym_passages[0]["id"], product_name)
            if include_sym else None
        )
        needs_compliance = include_sym and compliance_like
        needs_memory = include_mem and (memory_like or memory_dominant)
        symbolic_direct = _symbolic_direct_answer(text, sym_passages, product_name) if include_sym else None
        symbolic_query_direct = (
            _symbolic_query_direct_answer(text, sym_passages[0]["text"], sym_passages[0]["id"], product_name)
            if include_sym else None
        )
        doc_direct = None
        if include_search and document_like:
            doc_direct = _answer_simple_doc_field(text, search_passages)
            if not doc_direct:
                doc_direct = _compound_doc_field_answer(text, search_passages)
        strict_llm_final = os.getenv("AUTO_COMPOSE_DISABLE_DIRECT_FALLBACK", "").strip().lower() in {"1", "true", "yes", "y"}
        if llm_failed and os.getenv("AUTO_COMPOSE_FAIL_ON_LLM_ERROR", "").strip().lower() in {"1", "true", "yes", "y"}:
            pass
        elif not strict_llm_final:
            if (
                doc_direct
                and not (include_mem or include_sym)
                and _compact_answer_text(doc_direct.rsplit("[", 1)[0]) not in _compact_answer_text(answer)
            ):
                answer = doc_direct
                answer_trace = _post_llm_trace(
                    answer_trace,
                    "llm_then_deterministic_doc_field",
                    "The LLM answer omitted the exact extracted document field value.",
                    search_passages[0]["id"],
                )
            if (
                memory_answer
                and symbolic_answer
                and needs_memory
                and needs_compliance
                and (
                    answer.lower().startswith("insufficient context")
                    or not _answer_mentions_memory_content(answer)
                    or not _answer_mentions_compliance_content(answer)
                )
            ):
                answer = f"{memory_answer.rstrip('.')} {symbolic_answer}"
                answer_trace = _post_llm_trace(
                    answer_trace,
                    "llm_then_memory_symbolic_direct",
                    "The LLM answer missed required memory or symbolic evidence, so exact retrieved evidence was composed.",
                    f"{mem_passages[0]['id']}+{sym_passages[0]['id']}",
                )
            if include_mem and (memory_like or memory_dominant):
                fallback_answer = memory_answer
                if fallback_answer and (
                    ("packaging" in text.lower() or "supplier" in text.lower()) and not include_sym and not include_search
                    or answer.lower().startswith("insufficient context")
                    or not _answer_mentions_memory_content(answer)
                ):
                    answer = fallback_answer
                    answer_trace = _post_llm_trace(
                        answer_trace,
                        "llm_then_memory_direct",
                        "The LLM answer missed the exact retrieved memory value.",
                        mem_passages[0]["id"],
                    )
            if (
                symbolic_direct
                and needs_compliance
                and not needs_memory
                and (
                    answer.lower().startswith("insufficient context")
                    or not _answer_mentions_compliance_content(answer)
                )
            ):
                answer = symbolic_direct
                answer_trace = _post_llm_trace(
                    answer_trace,
                    "llm_then_symbolic_direct",
                    "The LLM answer missed the exact symbolic compliance evidence.",
                    sym_passages[0]["id"],
                )
            elif (
                (symbolic_query_direct or symbolic_direct)
                and include_sym
                and not needs_memory
                and not document_like
                and (
                    answer.lower().startswith("insufficient context")
                    or not _answer_mentions_symbolic_value(answer, sym_passages[0]["text"], text)
                )
            ):
                answer = symbolic_query_direct or symbolic_direct
                answer_trace = _post_llm_trace(
                    answer_trace,
                    "llm_then_symbolic_direct",
                    "The LLM answer missed the exact symbolic lookup value.",
                    sym_passages[0]["id"],
                )
            if (
                (symbolic_query_direct or symbolic_direct)
                and include_sym
                and needs_memory
                and not _answer_mentions_symbolic_value(answer, sym_passages[0]["text"], text)
            ):
                if memory_answer:
                    answer = f"{memory_answer.rstrip('.')} {symbolic_query_direct or symbolic_direct}"
                else:
                    answer = f"{answer.rstrip('.')} Exact symbolic evidence: {symbolic_query_direct or symbolic_direct}"
                answer_trace = _post_llm_trace(
                    answer_trace,
                    "llm_then_symbolic_append",
                    "The LLM answer missed the query-specific symbolic value, so exact symbolic evidence was appended.",
                    sym_passages[0]["id"],
                )
            if doc_direct and _compact_answer_text(doc_direct.rsplit("[", 1)[0]) not in _compact_answer_text(answer):
                if include_mem or include_sym:
                    answer = f"{answer.rstrip('.')} Exact document value: {doc_direct}"
                else:
                    answer = doc_direct
                answer_trace = _post_llm_trace(
                    answer_trace,
                    "llm_then_deterministic_doc_field",
                    "The LLM answer omitted the exact extracted document field value.",
                    search_passages[0]["id"],
                )
    else:
        answer = "Insufficient context."
        answer_trace["reason"] = "No passages were available after orchestration."

    steps.append({
        "source": "ANSWER",
        "answer_trace": answer_trace,
    })

    out = {
        "answer": answer,
        "answer_trace": answer_trace,
        "steps": steps,
        "sources": ordered_sources,
        "mode": "AUTO_COMPOSE",
        "session": session,
        "domain": effective_domain,
        "product": product_name,
        "auto_trace": {
            "selected_domain": selected_domain,
            "effective_domain": effective_domain,
            "memory_like_query": memory_like,
            "memory_dominant": memory_dominant,
            "features": feats,
            "effective_product": product_name,
            "product_inferred_from_query": inferred_product,
            "included_source_types": [
                source_type
                for source_type, included in (
                    ("memory", include_mem),
                    ("symbolic", include_sym),
                    ("search", include_search),
                )
                if included
            ],
        },
    }
    out["confidence"] = _attach_confidence("MEMSYM", out)
    return out


@router.post("/solve_auto")
def solve_auto(req: SolveAutoRequest):
    text = _pick_query(req)
    return solve_auto_query(
        query=text,
        product=req.product,
        domain=req.domain or "auto",
        session=req.session or "frontend-session",
        top_k_search=int(req.top_k_search or 4),
        top_k_memory=int(req.top_k_memory or 3),
    )
