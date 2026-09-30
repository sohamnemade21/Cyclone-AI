import os
import sys
import uvicorn
from backend.config import settings

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print(f">>> STARTING {settings.APP_NAME.upper()} SERVER")
    print("=" * 70)
    print(f">> Environment : {settings.ENVIRONMENT}")
    print(f">> Host        : {settings.HOST}")
    print(f">> Port        : {settings.PORT}")
    print(f">> Workers     : {settings.WORKERS}")
    print(f">> Auto-Reload : {settings.RELOAD}")
    print(f">> URL         : http://{settings.HOST if settings.HOST != '0.0.0.0' else '127.0.0.1'}:{settings.PORT}")
    print(f">> API Docs    : http://{settings.HOST if settings.HOST != '0.0.0.0' else '127.0.0.1'}:{settings.PORT}/docs")
    print("=" * 70)

    uvicorn.run(
        "backend.app:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS if not settings.RELOAD else 1,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL
    )
