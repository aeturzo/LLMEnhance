from __future__ import annotations

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# --- Strictly require the symbolic reasoner ---------------------------------
from backend.services.symbolic_reasoning_service import build_reasoner, SUPPORTED_DOMAINS  # must exist

# --- Routers ----------------------------------------------------------------
from backend.api.solve import router as solve_router
from backend.api.solve_auto import router as solve_auto_router
from backend.api.solve_rl import router as rl_router
from backend.api.carbon import router as carbon_router

# If you added seeding endpoints earlier, uncomment these:
from backend.api.memory import router as memory_router
from backend.api.ingest import router as ingest_router  # (rename from docs.py to avoid UI /docs clash)

# --- Logging ----------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

# --- App --------------------------------------------------------------------
app = FastAPI(
    title="Hybrid LLM Backend",
    version="0.1.0",
    description="Neural + Memory + Symbolic API",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lifecycle --------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    log = logging.getLogger("backend.main")
    log.info("Initializing symbolic reasoners…")
    reasoners = {}
    for domain in SUPPORTED_DOMAINS:
        reasoners[domain] = build_reasoner(run_owl_rl=True, domain=domain)
    app.state.reasoners = reasoners
    app.state.reasoner = reasoners["battery"]
    log.info("Symbolic reasoners initialized for domains: %s", ", ".join(sorted(reasoners.keys())))

# --- Health -----------------------------------------------------------------
@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "message": "See /docs for the Swagger UI."}

# --- Routes -----------------------------------------------------------------
app.include_router(solve_router)      # POST /solve
app.include_router(solve_auto_router) # POST /solve_auto
app.include_router(rl_router)       
app.include_router(carbon_router)     # POST /carbon/calculate
app.include_router(memory_router)     # POST /memory/put   (optional seeding)
app.include_router(ingest_router)     # POST /ingest/put   (optional seeding)
