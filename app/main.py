"""
FaceRelis - Modern Face Recognition Application

Main FastAPI application with clean architecture
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging import setup_logging, get_logger
from .api.routes import files, clustering, tasks
from .models.schemas import DriveInfoResponse

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    logger.info("Starting FaceRelis application")

    # Startup tasks
    yield

    # Shutdown tasks
    logger.info("Shutting down FaceRelis application")


# Create FastAPI application
app = FastAPI(
    title="FaceRelis",
    description="Modern Face Recognition and Clustering Application",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include API routes
app.include_router(
    files.router,
    prefix="/api",
    tags=["files"]
)

app.include_router(
    clustering.router,
    prefix="/api",
    tags=["clustering"]
)

app.include_router(
    tasks.router,
    prefix="/api",
    tags=["tasks"]
)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve main application page"""
    try:
        index_path = static_path / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
        else:
            return HTMLResponse(content="<h1>FaceRelis</h1><p>Application is starting...</p>")
    except Exception as e:
        logger.error(f"Failed to serve index page: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/drives", response_model=List[DriveInfoResponse])
async def get_drives():
    """Get available drives"""
    try:
        from ..utils.file_utils import get_logical_drives

        drives = get_logical_drives()
        return [
            DriveInfoResponse(
                name=drive["name"],
                path=drive["path"]
            )
            for drive in drives
        ]
    except Exception as e:
        logger.error(f"Failed to get drives: {e}")
        raise HTTPException(status_code=500, detail="Failed to get drives")


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon"""
    return {"detail": "No favicon available"}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info"
    )
