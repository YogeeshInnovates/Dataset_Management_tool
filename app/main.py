from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers import upload, labeling, export
from app.utils.file_utils import ensure_dirs, EXPORTS_DIR

app = FastAPI(title="Dataset Management Backend")

# Ensure storage directories exist on startup
ensure_dirs()

# Mount exports directory for static file serving
app.mount("/files", StaticFiles(directory=EXPORTS_DIR), name="files")

# Include routers
app.include_router(upload.router, tags=["Upload & Analysis"])
app.include_router(labeling.router, tags=["Labeling"])
app.include_router(export.router, tags=["Export"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Dataset Management Backend API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
