"""Test app imports."""

try:
    print("Testing imports...")
    from app.main import app
    print("[OK] app.main imported successfully")
    
    from app.core.detector import detector_service
    print("[OK] detector_service imported successfully")
    
    from app.models.schemas import AnalysisRequest
    print("[OK] Pydantic schemas imported successfully")
    
    from app.api.routes import router
    print("[OK] API routes imported successfully")
    
    print("\n[SUCCESS] All imports successful! App should work.")
except Exception as e:
    print(f"[ERROR] Import error: {e}")
    import traceback
    traceback.print_exc()

