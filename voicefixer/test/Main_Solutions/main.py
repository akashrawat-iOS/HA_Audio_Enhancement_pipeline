
import sys
import os
import io
import uuid
import logging
import tempfile
from pathlib import Path

# Suppress noisy Streamlit ScriptRunContext warnings when running outside Streamlit
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
import librosa
from pipeline_core import process_audio_pipeline

MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_DURATION_SECONDS = 120  # 2 minutes max


app = FastAPI(title="VoiceFixer Audio Enhancement API")

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@app.post("/process/")
async def process_audio(file: UploadFile = File(...)):
    try:
        # 1️⃣ Validate content type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="Only audio files are allowed")

        # 2️⃣ Read file into memory
        audio_bytes = await file.read()

        # 3️⃣ Empty file check
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # 4️⃣ Size limit check
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB"
            )

        # 5️⃣ Validate audio format + duration
        tmp_path = None
        try:
            # Save to temp file with original extension so librosa can detect format
            suffix = Path(file.filename).suffix if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            duration = len(y) / sr

            if duration > MAX_DURATION_SECONDS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio too long. Maximum allowed duration is {MAX_DURATION_SECONDS} seconds"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audio validation error: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid audio format: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # 6️⃣ Run pipeline in threadpool
        logger.info(f"Processing audio file: {file.filename}")
        result = await run_in_threadpool(
            process_audio_pipeline,
            audio_bytes,
            file.filename
        )

        # 7️⃣ Save enhanced audio to disk if accepted, return JSON with download URL
        decision = result.get("decision")
        metrics = result.get("metrics", {})
        download_url = None
        filename_out = None

        if decision == "Accepted":
            # Save enhanced audio to disk with a unique filename
            filename_out = f"enhanced_{uuid.uuid4().hex}.wav"
            output_dir = "enhanced_outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename_out)
            with open(output_path, "wb") as f:
                f.write(result["processed_bytes"])
            download_url = f"/download/{filename_out}"

        logger.info(f"Processing complete. Decision: {decision}")
        return JSONResponse(
            content={
                "decision": decision,
                "metrics": metrics,
                "download_url": download_url
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /process endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
# Add download endpoint

@app.get("/download/{filename:path}")
def download_file(filename: str):
    # Handle cases where filename includes '/download/' prefix
    if filename.startswith("/download/"):
        filename = filename[len("/download/"):]
    elif filename.startswith("download/"):
        filename = filename[len("download/"):]
    
    output_dir = "enhanced_outputs"
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="audio/wav"
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "voicefixer-api"}