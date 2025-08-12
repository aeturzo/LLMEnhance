from fastapi import APIRouter

api = APIRouter(prefix="/api")

# Include each sub-router safely
try:
    from . import routes
    api.include_router(routes.router)
except ImportError:
    pass

try:
    from . import symbolic
    api.include_router(symbolic.router)
except ImportError:
    pass

try:
    from . import solve
    api.include_router(solve.router)
except ImportError:
    pass

__all__ = ["api"]
