import os
import shutil
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.utils.file_utils import generate_session_id, UPLOADS_DIR, PROCESSED_DIR, ANALYSIS_DIR, EXPORTS_DIR, extract_zip, create_zip_archive
from app.services.validator import DatasetValidator
from app.services.analyzer import DatasetAnalyzer
from app.services.format_converter import FormatConverter
from app.services.splitter import DatasetSplitter
from app.models.schemas import UploadResponse, DatasetInternalFormat

router = APIRouter()

@router.post("/upload-dataset", response_model=UploadResponse)
async def upload_dataset(
    images_zip: UploadFile = File(...),
    labels_zip: UploadFile = File(...),
    format_type: str = Form(...)
):
    session_id = generate_session_id()
    
    # Paths for this session
    session_upload_dir = UPLOADS_DIR / session_id
    session_processed_dir = PROCESSED_DIR / session_id
    session_analysis_dir = ANALYSIS_DIR / session_id
    session_export_dir = EXPORTS_DIR / session_id
    
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    session_processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded zips
    images_zip_path = session_upload_dir / "images.zip"
    labels_zip_path = session_upload_dir / "labels.zip"
    
    with open(images_zip_path, "wb") as buffer:
        shutil.copyfileobj(images_zip.file, buffer)
    with open(labels_zip_path, "wb") as buffer:
        shutil.copyfileobj(labels_zip.file, buffer)
        
    # Extract
    extract_images_dir = session_upload_dir / "images"
    extract_labels_dir = session_upload_dir / "labels"
    extract_images_dir.mkdir(exist_ok=True)
    extract_labels_dir.mkdir(exist_ok=True)
    
    try:
        extract_zip(images_zip_path, extract_images_dir)
        extract_zip(labels_zip_path, extract_labels_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid zip file: {str(e)}")

    # Normalize format_type
    format_type = format_type.strip().lower().replace(" ", "_").replace("-", "_")
    if format_type == "pascal" or "pascal_voc" in format_type or format_type == "voc":
        format_type = "pascal_voc"
    
    supported_formats = ["yolo", "coco", "pascal_voc", "roboflow", "voc"]
    if format_type not in supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format_type}. Supported: {supported_formats}")

    # Validate
    validator = DatasetValidator(extract_images_dir, extract_labels_dir)
    report, annotations, stem_to_image, stem_to_label, class_names = validator.validate()
    
    print(f"DEBUG: Session {session_id} - Matched: {len(annotations)}, Orphans: {report.orphan_labels}")

    # Save internal format
    internal_format_path = session_processed_dir / "annotations.json"
    with open(internal_format_path, "w") as f:
        json.dump([ann.dict() for ann in annotations], f, indent=4)
        
    # Analyze
    analyzer = DatasetAnalyzer(session_id, session_analysis_dir)
    summary = analyzer.analyze(annotations, report)
    
    # Export (Initial)
    # Perform split ONLY on matched annotations
    train_anns, val_anns, test_anns = DatasetSplitter.split(annotations)
    
    # Create a root folder for the ZIP contents
    dataset_folder_name = f"dataset_{session_id}"
    export_output_dir = session_export_dir / dataset_folder_name
    export_output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": train_anns,
        "valid": val_anns,
        "test": test_anns
    }

    total_copied_images = 0

    if format_type in ["yolo", "roboflow"]:
        for split_name, split_anns in splits.items():
            if not split_anns: continue
            
            split_dir = export_output_dir / split_name
            images_out = split_dir / "images"
            labels_out = split_dir / "labels"
            images_out.mkdir(parents=True, exist_ok=True)
            labels_out.mkdir(parents=True, exist_ok=True)

            # 1. Convert labels
            FormatConverter.to_yolo(split_anns, split_dir, report.class_ids_found)
            
            # 2. Physically copy images
            for ann in split_anns:
                stem = ann.image_name
                src_img = stem_to_image.get(stem)
                if src_img and src_img.exists():
                    shutil.copy2(src_img, images_out / src_img.name)
                    total_copied_images += 1

        FormatConverter.generate_data_yaml(
            export_output_dir / "data.yaml", 
            report.class_ids_found, 
            "train/images", "valid/images", "test/images",
            custom_names=class_names
        )
        if format_type == "roboflow":
            FormatConverter.generate_roboflow_metadata(export_output_dir, "Custom Dataset")
            
    elif format_type == "coco":
        for split_name, split_anns in splits.items():
            if not split_anns: continue
            
            split_dir = export_output_dir / split_name
            images_out = split_dir / "images"
            images_out.mkdir(parents=True, exist_ok=True)
            
            # 1. Generate COCO JSON
            coco_data = FormatConverter.to_coco(split_anns, split_dir, report.class_ids_found)
            with open(split_dir / "annotations.json", "w") as f:
                json.dump(coco_data, f, indent=4)
            
            # 2. Physically copy images
            for ann in split_anns:
                stem = ann.image_name
                src_img = stem_to_image.get(stem)
                if src_img and src_img.exists():
                    shutil.copy2(src_img, images_out / src_img.name)
                    total_copied_images += 1

    elif format_type == "pascal_voc":
        # Structure: JPEGImages/, Annotations/, ImageSets/Main/
        images_out = export_output_dir / "JPEGImages"
        images_out.mkdir(parents=True, exist_ok=True)
        
        # 1. Convert all matched pairs to XMLs
        FormatConverter.to_pascal_voc(annotations, export_output_dir)
        
        # 2. Generate ImageSets/Main
        FormatConverter.generate_voc_imagesets(export_output_dir, splits)
        
        # 3. Physically copy all matched images to JPEGImages/
        for ann in annotations:
            stem = ann.image_name
            src_img = stem_to_image.get(stem)
            if src_img and src_img.exists():
                shutil.copy2(src_img, images_out / src_img.name)
                total_copied_images += 1

    # Verify count consistency
    if total_copied_images != len(annotations):
        raise HTTPException(status_code=500, detail=f"Export Failed: Expected {len(annotations)} images, but only {total_copied_images} were processed.")

    # Zip the root folder
    zip_path = session_export_dir / f"{session_id}.zip"
    create_zip_archive(session_export_dir, dataset_folder_name, zip_path)
    
    return UploadResponse(
        dataset_id=session_id,
        validation_report=report,
        analysis_summary=summary,
        csv_file_path=str(session_analysis_dir / "dataset_statistics.csv"),
        download_url=f"/download/{session_id}"
    )
