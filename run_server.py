import uvicorn
import os
import sys

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 65)
    print(">>> STARTING CYCLONEGUARD AI FULL-STACK SERVER")
    print("=" * 65)
    print(">> URL: http://localhost:8000")
    print(">> API Docs: http://localhost:8000/docs")
    print(">> Region: Bay of Bengal & Indian Ocean Cyclone Intelligence")
    print("=" * 65)
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)

