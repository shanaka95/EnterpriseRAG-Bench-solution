"""
FastAPI application entry point.
Hierarchical Soft-Clustering RAG Pipeline.
"""
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.core.config import settings
from app.core.database import engine, Base
from app.api.routes import router as api_router
from app.api.sse import router as sse_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Path to the built frontend
FRONTEND_BUILD_DIR = Path(os.environ.get("FRONTEND_BUILD_DIR", "/app/frontend/build"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: create tables on startup, try to load FAISS index."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    # Create database tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified.")

    # Try to load FAISS index for fast ANN retrieval
    try:
        from app.ml.faiss_index import load_index, get_index_stats
        if load_index():
            stats = get_index_stats()
            logger.info(f"FAISS index loaded: {stats}")
        else:
            logger.info("No FAISS index found. Use /api/v1/build_faiss to create one.")
    except Exception as e:
        logger.warning(f"FAISS not available: {e}")

    logger.info(f"Frontend build dir: {FRONTEND_BUILD_DIR} (exists={FRONTEND_BUILD_DIR.exists()})")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS - allow all for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers FIRST (higher priority than catch-all)
app.include_router(api_router)
app.include_router(sse_router)


# Serve React frontend - mount AFTER API routers
if FRONTEND_BUILD_DIR.exists():
    # Mount /static to serve /static/js/..., /static/css/..., etc.
    app.mount("/static", StaticFiles(directory=str(FRONTEND_BUILD_DIR / "static")), name="static")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        favicon_path = FRONTEND_BUILD_DIR / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(str(favicon_path))
        return FileResponse(str(FRONTEND_BUILD_DIR / "index.html"))

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Serve the React app's index.html at root."""
        index_path = FRONTEND_BUILD_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return HTMLResponse("<h1>Frontend not built</h1>")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(request: Request, full_path: str):
        """
        Catch-all route for SPA routing.
        Only serves index.html if no API route matched.
        """
        # Check for a specific file in the build directory
        file_path = FRONTEND_BUILD_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))

        # Fall back to index.html for SPA client-side routing
        index_path = FRONTEND_BUILD_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))

        return HTMLResponse("<h1>Not found</h1>", status_code=404)
else:
    @app.get("/")
    def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
        }
