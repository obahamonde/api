from fastapi import FastAPI

from .router import app as api


def create_app() -> FastAPI:
    """
    Create FastAPI application
    """
    app = FastAPI(
        title="AI Assistants API", description="API for AI Assistants", version="0.0.1"
    )

    app.include_router(api, prefix="/api")
    return app
