"""
pipeline_models.py
==================
Single source of truth for all model singletons used across the pipeline.

This module owns the PipelineModels dataclass and the factory function
that loads every model exactly once.

Design principles:
  - All models are immutable after construction (read-only inference state).
  - No Streamlit imports or @st.cache_resource decorators.
  - Can be imported by pipeline_core.py (FastAPI backend) and, in future,
    any other non-Streamlit caller.
  - The Streamlit module continues to own its own @st.cache_resource globals
    for its own UI; pipeline_core.py uses this module instead.

Thread safety:
  - `load_pipeline_models()` is called ONCE at module import of pipeline_core.py.
  - After construction the PipelineModels instance is never mutated.
  - All inference calls (transcribe, restore_inmem, predict) are stateless
    per-call, safe for concurrent requests via FastAPI threadpool.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple

import torch
import whisper
import enchant

from voicefixer import VoiceFixer
from scoreq import Scoreq

logger = logging.getLogger(__name__)

# Re-export the constants used by pipeline functions so callers only need
# to import from this module (or from the original for the thresholds).
from voicefixer_input_validation_gate_refined import (
    ANALYSIS_SR,
    VOICEFIXER_SR,
    DEVICE,
)


@dataclass
class PipelineModels:
    """
    Explicit container for every model singleton used in the pipeline.

    By making all dependencies explicit:
      - There are no hidden module globals.
      - Each function that needs a model receives it as an argument.
      - The caller (pipeline_core.py or Streamlit) controls which model
        instance is used — no aliasing, no duplication.

    Fields
    ------
    whisper_model : whisper.Whisper
        Loaded Whisper ASR model. Used by transcribe_audio().
    voicefixer : VoiceFixer
        VoiceFixer enhancement model. Used in Stage 2.
    scoreq_natural : Scoreq
        ScoreQ-NR (natural domain). Used for degraded input audio.
    scoreq_synthetic : Scoreq
        ScoreQ-NR (synthetic domain). Used for VoiceFixer output.
    dictionary : enchant.Dict | None
        PyEnchant dictionary for lexical coverage. None if unavailable.
    vad_model : Any | None
        Silero VAD model. None if unavailable.
    vad_get_speech_timestamps : callable | None
        Silero get_speech_timestamps function. Paired with vad_model.
    nisqa_available : bool
        True if NISQA weights were found and nisqa package is installed.
    nisqa_weights_path : str | None
        Absolute path to nisqa.tar weights file, or None.
    """
    whisper_model: Any
    voicefixer: VoiceFixer
    scoreq_natural: Scoreq
    scoreq_synthetic: Scoreq
    dictionary: Optional[Any]
    vad_model: Optional[Any]
    vad_get_speech_timestamps: Optional[Any]
    nisqa_available: bool
    nisqa_weights_path: Optional[str]


def load_pipeline_models() -> PipelineModels:
    """
    Load all models once and return a PipelineModels instance.

    Called at module import of pipeline_core.py.
    Never called per-request.

    Returns
    -------
    PipelineModels
        Fully initialised model container. Fields may be None for
        optional models that are unavailable in the environment.
    """
    # ------------------------------------------------------------------
    # Whisper
    # ------------------------------------------------------------------
    logger.info("Loading Whisper model (base)...")
    whisper_model = whisper.load_model("base", device=DEVICE)
    logger.info(f"✓ Whisper loaded on {DEVICE}")

    # ------------------------------------------------------------------
    # VoiceFixer
    # ------------------------------------------------------------------
    logger.info("Loading VoiceFixer model...")
    vf = VoiceFixer()
    logger.info("✓ VoiceFixer loaded")

    # ------------------------------------------------------------------
    # ScoreQ-NR (dual domain)
    # ------------------------------------------------------------------
    logger.info("Loading ScoreQ-NR (natural domain)...")
    scoreq_natural = Scoreq(mode="nr", data_domain="natural")
    logger.info("✓ ScoreQ natural loaded")

    logger.info("Loading ScoreQ-NR (synthetic domain)...")
    scoreq_synthetic = Scoreq(mode="nr", data_domain="synthetic")
    logger.info("✓ ScoreQ synthetic loaded")

    # ------------------------------------------------------------------
    # PyEnchant dictionary
    # ------------------------------------------------------------------
    dictionary = None
    for lang in ("en_US", "en_GB"):
        try:
            dictionary = enchant.Dict(lang)
            logger.info(f"✓ PyEnchant dictionary loaded ({lang})")
            break
        except enchant.errors.DictNotFoundError:
            continue
    if dictionary is None:
        logger.warning("⚠️ No English dictionary available — lexical validation disabled")

    # ------------------------------------------------------------------
    # Silero VAD
    # ------------------------------------------------------------------
    vad_model = None
    vad_get_speech_timestamps = None
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        vad_model = load_silero_vad(onnx=False)
        vad_get_speech_timestamps = get_speech_timestamps
        logger.info("✓ Silero VAD loaded")
    except Exception as e:
        logger.warning(f"⚠️ Silero VAD not available: {e}")

    # ------------------------------------------------------------------
    # NISQA — only weights path stored; model is file-based and created
    # per call inside _assess_nisqa_local (it must read a wav file path).
    # ------------------------------------------------------------------
    nisqa_available = False
    nisqa_weights_path = None
    try:
        from nisqa.NISQA_model import nisqaModel  # noqa: F401 — just validate import
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            "nisqa.tar",
            os.path.join(script_dir, "nisqa.tar"),
            os.path.join(script_dir, "..", "nisqa.tar"),
        ]
        for loc in candidates:
            if os.path.exists(loc):
                nisqa_weights_path = loc
                nisqa_available = True
                logger.info(f"✓ NISQA weights found at {loc}")
                break
        if not nisqa_available:
            logger.warning(f"⚠️ NISQA weights not found. Tried: {candidates}")
    except ImportError:
        logger.warning("⚠️ NISQA package not installed — perceptual quality checks disabled")

    return PipelineModels(
        whisper_model=whisper_model,
        voicefixer=vf,
        scoreq_natural=scoreq_natural,
        scoreq_synthetic=scoreq_synthetic,
        dictionary=dictionary,
        vad_model=vad_model,
        vad_get_speech_timestamps=vad_get_speech_timestamps,
        nisqa_available=nisqa_available,
        nisqa_weights_path=nisqa_weights_path,
    )
