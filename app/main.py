from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

# NOW import everything else that needs DATABASE_URL
from app.api.routes import upload, labeling, export, augmentation, datasets
from app.utils.file_utils import ensure_dirs, EXPORTS_DIR

app = FastAPI(title="Dataset Management Backend")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite and CRA default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure storage directories exist on startup
ensure_dirs()

# Initialize & create DB tables (will use DATABASE_URL env var if provided)
from app.core import Base, engine, ensure_additional_columns
Base.metadata.create_all(bind=engine)
# Ensure any newly-added JSON columns exist (safe / idempotent helper)
try:
    ensure_additional_columns(engine)
except Exception:
    pass

# Mount storage directory for static file access
from app.utils.file_utils import STORAGE_ROOT
app.mount("/storage", StaticFiles(directory=STORAGE_ROOT), name="storage")

# Mount exports directory for static file serving
app.mount("/files", StaticFiles(directory=EXPORTS_DIR), name="files")

# Include routers
app.include_router(datasets.router, tags=["Datasets"])
app.include_router(upload.router, tags=["Upload & Analysis"])
app.include_router(augmentation.router, tags=["Augmentation"])
app.include_router(labeling.router, tags=["Labeling"])
app.include_router(export.router, tags=["Export"])


@app.get("/")
async def root():
    return {"message": "Welcome to the Dataset Management Backend API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
