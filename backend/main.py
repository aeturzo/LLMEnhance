from __future__ import annotations

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# --- Strictly require the symbolic reasoner ---------------------------------
from backend.services.symbolic_reasoning_service import build_reasoner  # must exist

# --- Routers ----------------------------------------------------------------
from backend.api.solve import router as solve_router

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
    log.info("Initializing symbolic reasoner…")
    # Reasoner is required; if this raises, let it crash so you see the real error.
    app.state.reasoner = build_reasoner(run_owl_rl=True)  # adjust arg name if yours differs
    log.info("Symbolic reasoner initialized.")

# --- Health -----------------------------------------------------------------
@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "message": "See /docs for the Swagger UI."}

# --- Routes -----------------------------------------------------------------
app.include_router(solve_router)        # POST /solve
app.include_router(memory_router)     # POST /memory/put   (optional seeding)
app.include_router(ingest_router)     # POST /ingest/put   (optional seeding)
