import os
import shutil
import zipfile
import uuid
from pathlib import Path

STORAGE_ROOT = Path("dataset_backend/app/storage")
UPLOADS_DIR = STORAGE_ROOT / "uploads"
PROCESSED_DIR = STORAGE_ROOT / "processed"
ANALYSIS_DIR = STORAGE_ROOT / "analysis"
EXPORTS_DIR = STORAGE_ROOT / "exports"

def ensure_dirs():
    """Ensure all required storage directories exist."""
    for directory in [UPLOADS_DIR, PROCESSED_DIR, ANALYSIS_DIR, EXPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def generate_session_id() -> str:
    """Generate a unique session UUID."""
    return str(uuid.uuid4())

def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip file to a specific directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def cleanup_session(session_id: str):
    """Remove all files associated with a session ID."""
    for directory in [UPLOADS_DIR, PROCESSED_DIR, ANALYSIS_DIR, EXPORTS_DIR]:
        session_path = directory / session_id
        if session_path.exists():
            shutil.rmtree(session_path)

def create_zip_archive(parent_dir: Path, folder_to_zip: str, output_zip_path: Path):
    """
    Create a zip archive where folder_to_zip is the root folder inside the ZIP.
    - parent_dir: The directory containing the folder to zip.
    - folder_to_zip: The name of the folder within parent_dir to be zipped.
    - output_zip_path: The full path where the .zip file should be saved.
    """
    base_name = str(output_zip_path).replace(".zip", "")
    shutil.make_archive(
        base_name=base_name,
        format='zip',
        root_dir=parent_dir,
        base_dir=folder_to_zip
    )
    return output_zip_path
