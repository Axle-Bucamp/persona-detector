"""Web app module entry point."""

import uvicorn
from semantic_detector.web.app.core.config import settings

if __name__ == '__main__':
    uvicorn.run(
        "semantic_detector.web.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )

