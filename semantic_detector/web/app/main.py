"""FastAPI application main file."""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn

from semantic_detector.web.app.core.config import settings
from semantic_detector.web.app.api.routes import router

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Language fingerprinting and embedding analysis system"
)

# Include routers
app.include_router(router)

# Get paths relative to this file
from pathlib import Path

# static/ and templates/ are in semantic_detector/web/
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"
PUBLIC_DIR = BASE_DIR / "public"
TEMPLATES_DIR = BASE_DIR / "templates"

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# Mount public files (icons, images, etc.)
if PUBLIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint."""
    # Get base URL for Open Graph tags
    base_url = str(request.base_url).rstrip('/')
    return templates.TemplateResponse("index.html", {
        "request": request,
        "base_url": base_url
    })


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # Create necessary directories
    project_root = Path(__file__).parent.parent.parent.parent
    uploads_dir = project_root / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    print(f"[STARTED] {settings.app_name} v{settings.app_version}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("[SHUTDOWN] Shutting down...")


if __name__ == "__main__":
    uvicorn.run(
        "semantic_detector.web.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

