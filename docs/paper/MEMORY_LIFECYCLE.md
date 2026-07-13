# Memory lifecycle audit

Audit date: 2026-07-13. Scope: the current prototype implementation in
`backend/services/memory_service.py` and `scripts/seed_memory.py`.

## Result

The lifecycle sentence in COMPASS §4.2 is not supported by the current code.
The implementation is a session-scoped semantic text memory, not a validated,
product-versioned fact store. The paper should be corrected rather than adding
production storage semantics immediately before the frozen evaluation.

| Paper claim | Finding | Implementation evidence |
|---|---|---|
| Extraction → normalization → provenance linking → validation before storage; rejected facts are not stored | **CORRECTED.** `add_memory` accepts any session/content pair and performs no validation. The common seeding path discards its `meta` argument when `add_memory` exists. | `MemoryService.add_memory`; `scripts/seed_memory.py::_add_mem` |
| Every stored fact has provenance, validation state, product identity, and timestamp | **CORRECTED.** Persisted entries contain `session_id`, `content`, and `timestamp` only. They have no provenance ID, validation flag, product ID, fact ID, or supersession ID. | `MemoryEntry.as_dict` |
| Facts are immutable; corrections supersede rather than overwrite; no deletion | **CORRECTED.** `_persist` rewrites the JSONL file, and `flush_session` and `reset_storage` delete entries. There is no update/supersession API or version relation. | `MemoryService._persist`, `flush_session`, `reset_storage` |
| Retrieval is scoped by product identity and cannot cross products | **CORRECTED.** Retrieval is filtered by `session_id`. It is isolated when callers use distinct sessions, but product identity is not stored or independently enforced. | `MemoryService.retrieve` |

## Actual pipeline and schema

1. A caller supplies free text and a session ID.
2. The service embeds the text, appends an in-memory row, and persists all
   current metadata rows to `memory.meta.jsonl`.
3. Retrieval embeds a query, filters candidates to the requested session, and
   ranks those candidates by cosine similarity.
4. A caller may flush one session or reset all storage.

Persisted schema:

```json
{
  "session_id": "string",
  "content": "string",
  "timestamp": "ISO-8601 string"
}
```

## Replacement text for §4.2

> The prototype memory stores timestamped text entries and retrieves them by
> semantic similarity within the current session. Session filtering prevents
> retrieval across separately assigned sessions, but the current store does
> not independently enforce product identity, provenance validation, immutable
> supersession, or correction history; these are requirements for a production
> DPP deployment rather than properties evaluated in this study.

## Engineering decision

Do not retrofit validation, provenance, product versioning, and immutable
supersession immediately before the frozen rerun. That would materially change
the evaluated system and risks corrupting or invalidating existing memory-based
results. If implemented later, it should use a versioned fact identifier,
product identifier, provenance record, validation status, `supersedes` link,
and append-only persistence, with migration tests against a copied store.
