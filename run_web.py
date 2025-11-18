"""Run the web application."""

import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"[INFO] Starting Semantic Detector web app on http://localhost:{port}")
    print(f"[INFO] Press Ctrl+C to stop")
    
    try:
        uvicorn.run(
            "semantic_detector.web.app.main:app",
            host="0.0.0.0",
            port=port,
            reload=False
        )
    except OSError as e:
        if "10048" in str(e) or "address already in use" in str(e).lower():
            print(f"\n[ERROR] Port {port} is already in use!")
            print(f"[INFO] Try using a different port: python run_web.py 8080")
        else:
            raise

