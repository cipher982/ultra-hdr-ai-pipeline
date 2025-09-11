"""
HDR Web Service - FastAPI application for HDR photo restoration.

Upload SDR images and get HDR versions back with visual preview.
"""

import os
import tempfile
import uuid
import asyncio
from pathlib import Path
from typing import Optional
import shutil

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np

# Import our HDR pipeline
from hdr.gainmap_pipeline import run_gainmap_pipeline, GainMapPipelineConfig, GainMapPipelineError
from hdr.models import ModelError
from hdr.exporters.direct_ultrahdr import UltraHDRDirectError
from hdr_web.utils.preview import create_hdr_preview

app = FastAPI(title="HDR Photo Restorer", description="Restore HDR to AI-edited photos")

# Setup templates and static files
templates = Jinja2Templates(directory="hdr_web/templates")
app.mount("/static", StaticFiles(directory="hdr_web/static"), name="static")

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Temporary storage for processed files (in project directory)
TEMP_DIR = Path("temp_web")
TEMP_DIR.mkdir(exist_ok=True)

# Clean up old files on startup
def cleanup_old_files():
    """Remove files older than 1 hour"""
    import time
    cutoff = time.time() - 3600  # 1 hour ago
    
    for file_path in TEMP_DIR.glob("*"):
        if file_path.stat().st_mtime < cutoff:
            try:
                file_path.unlink()
            except:
                pass

def cleanup_job_files(job_id: str, keep_input: bool = False):
    """Clean up files for a specific job"""
    patterns = [f"{job_id}_hdr.jpg", f"{job_id}_preview.jpg"]
    if not keep_input:
        patterns.append(f"{job_id}_input.jpg")
    
    for pattern in patterns:
        file_path = TEMP_DIR / pattern
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass

def cleanup_preview_files(preview_id: str):
    """Clean up preview files immediately"""
    patterns = [f"preview_{preview_id}_input.jpg", f"preview_{preview_id}_output.jpg"]
    
    for pattern in patterns:
        file_path = TEMP_DIR / pattern
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass

# Disabled auto-cleanup for debugging
# cleanup_old_files()

# Background cleanup task (disabled)
# async def background_cleanup():
#     """Periodically clean up old files"""
#     while True:
#         await asyncio.sleep(300)  # Every 5 minutes
#         cleanup_old_files()

# Start background task (disabled)  
@app.on_event("startup")
async def startup_event():
    """Startup event - auto-cleanup disabled for debugging"""
    print("HDR Web Service started - auto-cleanup disabled for debugging")


@app.post("/admin/cleanup")
async def manual_cleanup():
    """Manual cleanup endpoint for development/debugging"""
    try:
        cleanup_old_files()
        return {"status": "success", "message": "Temporary files cleaned up"}
    except Exception as e:
        return {"status": "error", "message": f"Cleanup failed: {e}"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main upload page"""
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/process")
async def process_image(
    request: Request,
    file: UploadFile = File(...),
    strength: float = Form(1.0)
):
    """Process uploaded image and return results page"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported")
    
    # Check file size (read content to check)
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    if len(content) < 1000:  # Minimum viable image size
        raise HTTPException(status_code=400, detail="File too small or corrupted")
    
    # Reset file position for later use
    await file.seek(0)
    
    # Generate unique ID for this processing job
    job_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file
        input_path = TEMP_DIR / f"{job_id}_input.jpg"
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create output paths
        hdr_output_path = TEMP_DIR / f"{job_id}_hdr.jpg"
        preview_path = TEMP_DIR / f"{job_id}_preview.jpg"
        
        # Configure HDR pipeline (using working direct exporter only)
        config = GainMapPipelineConfig(
            model_type="auto",
            export_quality=95,
            max_side=4096,
            save_intermediate=False,
            strict_mode=True
        )
        
        # Process with HDR pipeline
        result = run_gainmap_pipeline(str(input_path), str(hdr_output_path), config)
        
        # Validate HDR output BEFORE serving to user
        if not hdr_output_path.exists():
            raise HTTPException(status_code=500, detail="HDR processing failed - output not created")
        
        hdr_size = hdr_output_path.stat().st_size
        
        # Validate HDR structure instead of just size (HDR might be smaller due to compression)
        from hdr.gainmap_pipeline import validate_ultrahdr_structure
        validation = validate_ultrahdr_structure(str(hdr_output_path))
        
        if not validation['valid']:
            errors = '; '.join(validation['errors'])
            raise HTTPException(status_code=500, detail=f"HDR processing failed - {errors}")
        
        # Ensure file is substantial but not unreasonably large
        if hdr_size < 10000:  # At least 10KB
            raise HTTPException(status_code=500, detail="HDR processing failed - output file too small") 
        
        if hdr_size > 50 * 1024 * 1024:  # Max 50MB
            raise HTTPException(status_code=500, detail="HDR processing failed - output file too large")
        
        # Validate JPEG header
        with open(hdr_output_path, 'rb') as f:
            header = f.read(4)
            if not header.startswith(b'\xff\xd8'):
                raise HTTPException(status_code=500, detail="HDR processing failed - invalid JPEG format")
        
        # Generate web preview that shows HDR effect
        create_hdr_preview(
            str(input_path), 
            str(preview_path),
            strength=strength
        )
        
        # Return results page
        response = templates.TemplateResponse("result.html", {
            "request": request,
            "job_id": job_id,
            "original_filename": file.filename,
            "model_name": result.model_name,
            "original_size": input_path.stat().st_size,
            "hdr_size": hdr_size,
            "has_hdr": True,  # We validated it exists above
            "strength": strength
        })
        
        # No auto-cleanup - keep files for debugging and manual inspection
        
        return response
        
    except GainMapPipelineError as e:
        # Pipeline-specific errors with user-friendly messages
        error_msg = str(e)
        if "AI model failed" in error_msg:
            raise HTTPException(status_code=500, detail="AI processing failed. Please try again with a different image.")
        elif "not created" in error_msg:
            raise HTTPException(status_code=500, detail="HDR processing failed. The image may be too complex or corrupted.")
        else:
            raise HTTPException(status_code=500, detail=f"HDR processing error: {error_msg}")
    
    except UltraHDRDirectError as e:
        raise HTTPException(status_code=500, detail="HDR export failed. Please try again.")
    
    except ModelError as e:
        raise HTTPException(status_code=500, detail="AI model is temporarily unavailable. Please try again later.")
    
    except MemoryError:
        raise HTTPException(status_code=413, detail="Image too large to process. Please use a smaller image.")
    
    except Exception as e:
        # Log the actual error for debugging but return user-friendly message
        import traceback
        print(f"Unexpected error: {e}")
        print(f"Error type: {type(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/image/{job_id}/{image_type}")
async def serve_image(job_id: str, image_type: str):
    """Serve processed images"""
    
    # Validate image type and build file path
    if image_type == "original":
        file_path = TEMP_DIR / f"{job_id}_input.jpg"
    elif image_type == "preview":
        file_path = TEMP_DIR / f"{job_id}_preview.jpg"
    else:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path)


@app.post("/preview")
async def generate_live_preview(
    file: UploadFile = File(...),
    strength: float = Form(1.0)
):
    """Generate real-time HDR preview for slider changes"""
    
    # Validate file size for preview (smaller limit for real-time processing)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB limit for previews
        raise HTTPException(status_code=413, detail="File too large for preview. Maximum: 10MB")
    
    if len(content) < 1000:
        raise HTTPException(status_code=400, detail="File too small or corrupted")
    
    await file.seek(0)
    
    try:
        # Generate temp job ID for preview
        preview_id = str(uuid.uuid4())
        input_path = TEMP_DIR / f"preview_{preview_id}_input.jpg"
        preview_path = TEMP_DIR / f"preview_{preview_id}_output.jpg"
        
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate enhanced preview quickly
        create_hdr_preview(
            str(input_path), 
            str(preview_path),
            strength=strength
        )
        
        # Return the preview image (no auto-cleanup)
        response = FileResponse(preview_path, media_type="image/jpeg")
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {e}")


@app.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download processed Ultra HDR JPEG file"""
    
    # Only support HDR JPEG downloads
    if file_type != "hdr":
        raise HTTPException(status_code=400, detail="Only HDR JPEG downloads supported")
    
    file_path = TEMP_DIR / f"{job_id}_hdr.jpg"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="HDR file not found")
    
    # Final validation before download
    file_size = file_path.stat().st_size
    if file_size < 50000:  # Ultra HDR should be at least 50KB
        raise HTTPException(status_code=500, detail="HDR file appears corrupted (too small)")
    
    response = FileResponse(
        file_path, 
        filename="hdr_enhanced.jpg",
        media_type="image/jpeg"
    )
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "HDR Photo Restorer"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)