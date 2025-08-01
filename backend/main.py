from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import routes
from backend.services import search_service

app = FastAPI(title="DPP Semantic Search API", version="0.1.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    routes.router,
    prefix="/api",
    tags=["Semantic Search"],
)

@app.on_event("startup")
async def on_startup():
    pass
