"""
pipeline_core.py
================
Pure backend for the VoiceFixer validation/enhancement pipeline.

No Streamlit imports. No hidden globals. All model dependencies are
explicit: loaded once into a PipelineModels instance at module import,
then passed explicitly to every function that needs them.

Architecture
------------
  pipeline_models.py  →  PipelineModels dataclass + load_pipeline_models()
  pipeline_core.py    →  owns the singleton, defines all pipeline functions
                         with explicit model parameters, exposes
                         process_audio_pipeline() to FastAPI

Why this fixes the hidden-global problem
-----------------------------------------
The original Streamlit module owns:
    whisper_model, dictionary, vad_model, vad_utils,
    scoreq_natural, scoreq_synthetic, nisqa_model, nisqa_available
as module-level globals decorated with @st.cache_resource.

Previously, importing validate_input_audio / transcribe_audio / etc.
from that module silently used those Streamlit-owned globals — so any
model loaded here was a duplicate that was never actually used.

The fix: re-implement every function that touches a global so it accepts
a PipelineModels argument instead. The Streamlit module's globals are
never accessed from this module.

Thread safety
-------------
  - PipelineModels is constructed once at module import, never mutated.
  - All inference calls (transcribe, restore_inmem, predict) are
    stateless per call — safe for concurrent FastAPI requests via threadpool.
  - All temp files are per-request (NamedTemporaryFile with unique names).
  - No shared mutable state between requests.
"""

import io
import os
import logging
import tempfile
from typing import Optional, Dict, List, Tuple

import numpy as np
import librosa
import torch
import soundfile as sf

# Import pure threshold/config constants only — no functions that touch globals
from voicefixer_input_validation_gate_refined import (
    # Thresholds (plain constants, no hidden state)
    ANALYSIS_SR,
    VOICEFIXER_SR,
    MIN_WORD_DENSITY,
    MIN_WORD_DENSITY_SHORT,
    MIN_DICTIONARY_COVERAGE,
    MIN_ABSOLUTE_WORDS,
    MIN_TRANSCRIPT_LENGTH,
    SHORT_CLIP_THRESHOLD,
    DOMAIN_ALLOWLIST,
    NON_LEXICAL_WORDS,
    RMS_ENERGY_MIN,
    VOICED_FRAME_RATIO_MIN,
    MIN_VOICED_DURATION,
    VAD_THRESHOLD,
    ENABLE_LANGUAGE_DETECTION,
    REQUIRED_LANGUAGE,
    MAX_INPUT_BANDWIDTH_HZ,
    MIN_TRANSCRIPTION_CONFIDENCE,
    MAX_NO_SPEECH_PROB,
    NISQA_MOS_DELTA_MIN,
    NISQA_COL_MIN,
    NISQA_DIS_MIN,
    NISQA_MOS_MIN,
    # Pure utility functions — no global access
    normalize,
    save_wav,
    check_acoustic_speech_presence,
    analyze_audio_quality,
    calculate_bandwidth,
    calculate_snr,
    calculate_temporal_roughness,
    calculate_risk_score,
    calculate_benefit_score,
    decide_enhancement_quality,
    # Interpretation helpers (pure functions)
    interpret_confidence,
    interpret_no_speech_prob,
    interpret_speech_rate,
    interpret_dictionary_coverage,
    interpret_bandwidth,
    interpret_temporal_roughness,
)

from pipeline_models import (
    PipelineModels,
    load_pipeline_models,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single model singleton — loaded ONCE at module import.
# Every function below receives `models` explicitly; nothing is global.
# ---------------------------------------------------------------------------
_models: PipelineModels = load_pipeline_models()


# ===========================================================================
# Layer 1 — Refactored functions with explicit model parameters
# (These replace the originals that used hidden globals.)
# ===========================================================================

def _transcribe_audio(
    audio: np.ndarray,
    models: PipelineModels,
) -> Tuple[List[Dict], str, str, float, float]:
    """
    Transcribe audio using Whisper.

    Replaces transcribe_audio() from the original module.
    Uses models.whisper_model instead of the module-global whisper_model.

    Returns
    -------
    (word_list, transcript, detected_language, confidence, no_speech_prob)
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        save_wav(tmp.name, audio, ANALYSIS_SR)
        tmp_path = tmp.name
    try:
        result = models.whisper_model.transcribe(
            tmp_path,
            word_timestamps=True,
            fp16=False,
            temperature=0.0,
            beam_size=5,
            best_of=5,
        )
    finally:
        os.unlink(tmp_path)

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None:
                words.append({
                    "word": w["word"].strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })

    detected_language = result.get("language", "unknown")
    segments = result.get("segments", [])
    if segments:
        avg_logprobs = [s.get("avg_logprob", -999) for s in segments if "avg_logprob" in s]
        confidence = float(np.mean(avg_logprobs)) if avg_logprobs else -999.0
        no_speech_prob = float(segments[0].get("no_speech_prob", 0.0))
    else:
        confidence = -999.0
        no_speech_prob = 1.0

    return words, result.get("text", ""), detected_language, confidence, no_speech_prob


def _detect_speech_segments(
    audio: np.ndarray,
    sr: int,
    models: PipelineModels,
) -> Tuple[List[Dict], float, float]:
    """
    Detect speech segments using Silero VAD.

    Replaces detect_speech_segments() from the original module.
    Uses models.vad_model / models.vad_get_speech_timestamps instead of
    the module-globals vad_model / vad_utils.

    Returns
    -------
    (speech_segments, total_voiced_duration, voiced_percentage)
    """
    if models.vad_model is None or models.vad_get_speech_timestamps is None:
        logger.warning("Silero VAD not available — returning zero voiced duration")
        return [], 0.0, 0.0

    try:
        audio_16k = (
            librosa.resample(audio, orig_sr=sr, target_sr=16000)
            if sr != 16000
            else audio
        )
        audio_tensor = torch.from_numpy(audio_16k).float()

        speech_timestamps = models.vad_get_speech_timestamps(
            audio_tensor,
            models.vad_model,
            threshold=VAD_THRESHOLD,
            sampling_rate=16000,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        speech_segments: List[Dict] = []
        total_voiced_samples = 0
        for ts in speech_timestamps:
            start_sec = ts["start"] / 16000.0
            end_sec = ts["end"] / 16000.0
            speech_segments.append({"start": start_sec, "end": end_sec,
                                     "duration": end_sec - start_sec})
            total_voiced_samples += ts["end"] - ts["start"]

        total_voiced_duration = total_voiced_samples / 16000.0
        total_duration = len(audio) / sr
        voiced_percentage = total_voiced_duration / total_duration if total_duration > 0 else 0.0

        return speech_segments, total_voiced_duration, voiced_percentage

    except Exception as e:
        logger.error(f"VAD detection failed: {e}")
        return [], 0.0, 0.0


def _validate_input_audio(
    audio: np.ndarray,
    duration: float,
    models: PipelineModels,
) -> Tuple[bool, Optional[str], Dict]:
    """
    Pre-screen input audio quality before VoiceFixer processing.

    Replaces validate_input_audio() from the original module.
    Uses models.whisper_model (via _transcribe_audio) and models.dictionary
    instead of the module-globals whisper_model and dictionary.

    Identical logic to the original — all thresholds and rejection rules
    are preserved exactly.
    """
    validation_info: Dict = {
        "word_count": 0,
        "words_per_sec": 0.0,
        "dictionary_coverage": None,
        "valid_words": [],
        "invalid_words": [],
        "transcript": "",
        "is_short_clip": duration < SHORT_CLIP_THRESHOLD,
        "acoustic_fallback_used": False,
        "acoustic_info": None,
        "bandwidth_hz": 0.0,
        "snr_db": 0.0,
        "temporal_roughness": 0.0,
        "detected_language": "unknown",
        "transcription_confidence": 0.0,
        "no_speech_prob": 0.0,
        "speech_segments": [],
        "voiced_duration": 0.0,
        "voiced_percentage": 0.0,
    }

    is_short_clip = duration < SHORT_CLIP_THRESHOLD

    # VAD — uses models.vad_model explicitly
    speech_segments, voiced_duration, voiced_percentage = _detect_speech_segments(
        audio, ANALYSIS_SR, models
    )
    validation_info["speech_segments"] = speech_segments
    validation_info["voiced_duration"] = voiced_duration
    validation_info["voiced_percentage"] = voiced_percentage

    # Acoustic metrics (pure functions, no globals)
    validation_info["bandwidth_hz"] = calculate_bandwidth(audio, ANALYSIS_SR)
    validation_info["snr_db"] = calculate_snr(audio, ANALYSIS_SR, speech_segments)
    validation_info["temporal_roughness"] = calculate_temporal_roughness(audio, ANALYSIS_SR)

    # Hard bandwidth reject
    if validation_info["bandwidth_hz"] > MAX_INPUT_BANDWIDTH_HZ:
        return (
            False,
            f"Input bandwidth too high: {validation_info['bandwidth_hz']:.0f} Hz "
            f"(maximum allowed: {MAX_INPUT_BANDWIDTH_HZ:.0f} Hz)",
            validation_info,
        )

    # Transcription — uses models.whisper_model explicitly
    words, transcript, detected_lang, confidence, no_speech_prob = _transcribe_audio(
        audio, models
    )
    word_count = len(words)

    validation_info["detected_language"] = detected_lang
    validation_info["transcription_confidence"] = confidence
    validation_info["no_speech_prob"] = no_speech_prob
    validation_info["word_count"] = word_count
    validation_info["transcript"] = transcript

    # VAD minimum voiced duration
    if voiced_duration < MIN_VOICED_DURATION:
        return (
            False,
            f"Insufficient speech content: only {voiced_duration:.2f}s of voiced audio "
            f"(minimum: {MIN_VOICED_DURATION}s)",
            validation_info,
        )

    # Acoustic fallback for zero-word cases
    if word_count < MIN_ABSOLUTE_WORDS:
        has_signal, acoustic_info = check_acoustic_speech_presence(audio, ANALYSIS_SR)
        validation_info["acoustic_fallback_used"] = True
        validation_info["acoustic_info"] = acoustic_info
        if not has_signal and duration > 2.0:
            return (
                False,
                f"No speech detected: {word_count} words, no acoustic signal "
                f"(RMS={acoustic_info['rms_energy']:.4f})",
                validation_info,
            )

    # Multi-failure rejection logic
    validation_failures: List[str] = []

    words_per_sec = word_count / duration if duration > 0 else 0.0
    validation_info["words_per_sec"] = words_per_sec
    effective_threshold = MIN_WORD_DENSITY_SHORT if is_short_clip else MIN_WORD_DENSITY

    low_word_count = word_count < MIN_ABSOLUTE_WORDS and duration > 2.0
    low_word_density = words_per_sec < effective_threshold
    if low_word_count or low_word_density:
        details = []
        if low_word_count:
            details.append(f"count={word_count}<{MIN_ABSOLUTE_WORDS}")
        if low_word_density:
            details.append(f"density={words_per_sec:.2f}<{effective_threshold:.2f} wps")
        validation_failures.append(f"Transcript sparsity ({', '.join(details)})")

    # Dictionary coverage — uses models.dictionary explicitly
    if models.dictionary and word_count >= MIN_TRANSCRIPT_LENGTH:
        word_list = transcript.lower().split()
        valid_words: List[str] = []
        invalid_words: List[str] = []
        for word in word_list:
            clean = "".join(c for c in word if c.isalpha())
            if not clean:
                continue
            if len(clean) == 1 and clean not in {"i", "a"}:
                continue
            if clean in NON_LEXICAL_WORDS or clean in DOMAIN_ALLOWLIST:
                valid_words.append(clean)
                continue
            if models.dictionary.check(clean):
                valid_words.append(clean)
            else:
                invalid_words.append(clean)

        total_checked = len(valid_words) + len(invalid_words)
        dict_coverage = len(valid_words) / total_checked if total_checked > 0 else 0.0
        validation_info["dictionary_coverage"] = dict_coverage
        validation_info["valid_words"] = valid_words
        validation_info["invalid_words"] = invalid_words
        if dict_coverage < MIN_DICTIONARY_COVERAGE:
            validation_failures.append(f"Low dictionary coverage ({dict_coverage:.1%})")

    if confidence < MIN_TRANSCRIPTION_CONFIDENCE:
        validation_failures.append(
            f"Low transcription confidence ({confidence:.3f} < {MIN_TRANSCRIPTION_CONFIDENCE})"
        )
    if no_speech_prob > MAX_NO_SPEECH_PROB:
        validation_failures.append(
            f"High no-speech probability ({no_speech_prob:.1%} > {MAX_NO_SPEECH_PROB:.1%})"
        )

    if len(validation_failures) >= 2:
        return False, "Multiple validation failures: " + "; ".join(validation_failures), validation_info
    if len(validation_failures) == 1 and word_count < MIN_ABSOLUTE_WORDS and duration > 3.0:
        return False, validation_failures[0] + " (high confidence rejection)", validation_info

    return True, None, validation_info


def _assess_nisqa(
    audio_path: str,
    models: PipelineModels,
    audio_label: str = "audio",
) -> Optional[Dict]:
    """
    Assess perceptual quality using NISQA (file-based API).

    Replaces assess_nisqa() from the original module.
    Uses models.nisqa_available / models.nisqa_weights_path instead of
    module-globals nisqa_available / nisqa_model.
    """
    if not models.nisqa_available or not models.nisqa_weights_path:
        return None
    try:
        from nisqa.NISQA_model import nisqaModel
        args = {
            "mode": "predict_file",
            "pretrained_model": models.nisqa_weights_path,
            "num_workers": 0,
            "bs": 1,
            "ms_channel": None,
            "deg": audio_path,
            "output_dir": None,
        }
        file_model = nisqaModel(args)
        file_model.predict()
        row = file_model.ds_val.df.iloc[0]
        return {
            "model": "NISQA",
            "mos": float(row["mos_pred"]),
            "noi": float(row["noi_pred"]),
            "dis": float(row["dis_pred"]),
            "col": float(row["col_pred"]),
            "loud": float(row["loud_pred"]),
            "label": audio_label,
        }
    except Exception as e:
        logger.error(f"NISQA error for {audio_label}: {e}")
        return None


def _assess_scoreq(
    audio: np.ndarray,
    sr: int,
    models: PipelineModels,
    is_synthetic: bool = False,
    audio_label: str = "audio",
) -> Optional[Dict]:
    """
    Assess speech quality using ScoreQ-NR (file-based API).

    Replaces assess_scoreq() from the original module.
    Uses models.scoreq_natural / models.scoreq_synthetic instead of
    module-globals scoreq_natural / scoreq_synthetic.

    is_synthetic=False → natural domain  (degraded input)
    is_synthetic=True  → synthetic domain (VoiceFixer output)
    """
    try:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            tmp_path = tmp.name
        try:
            model = models.scoreq_synthetic if is_synthetic else models.scoreq_natural
            score = model.predict(tmp_path)
            return {
                "model": "ScoreQ-NR",
                "score": float(score),
                "domain": "synthetic" if is_synthetic else "natural",
                "label": audio_label,
            }
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"ScoreQ-NR error for {audio_label}: {e}")
        return None


# ===========================================================================
# Layer 2 — Main pipeline entry point
# ===========================================================================

def process_audio_pipeline(audio_bytes: bytes, filename: str = "input.wav") -> dict:
    """
    Full 3-stage backend pipeline.

    Stage 1 — Input Validation
        _validate_input_audio() with models.whisper_model, models.dictionary,
        models.vad_model — NO hidden globals.

    Stage 2 — VoiceFixer Enhancement
        normalize → resample to 44.1 kHz → models.voicefixer.restore_inmem
        → normalize → resample back to 16 kHz.

    Stage 3 — Output Quality Gate
        _validate_input_audio() on enhanced (re-uses models — same singletons)
        _assess_nisqa() with models.nisqa_weights_path
        _assess_scoreq() with models.scoreq_natural / scoreq_synthetic
        decide_enhancement_quality() — Tier 1–6 hierarchical logic

    Final audio
        Accepted → vf_out at VOICEFIXER_SR (44.1 kHz), matches Streamlit ZIP
        Rejected → original bytes, unchanged

    Returns
    -------
    {
        "decision": "Accepted" | "Rejected (Input Validation)" | "Rejected",
        "processed_bytes": bytes,
        "metrics": dict
    }
    """
    original_path: Optional[str] = None
    enhanced_path: Optional[str] = None
    input_temp_path: Optional[str] = None

    try:
        # Load audio - save to temp file with proper extension for format detection
        from pathlib import Path
        suffix = Path(filename).suffix if filename else ".wav"
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            input_temp_path = tmp.name
        
        raw_audio, sr = librosa.load(input_temp_path, sr=None, mono=True)
        duration = len(raw_audio) / sr

        raw_audio_16k = (
            librosa.resample(raw_audio, orig_sr=sr, target_sr=ANALYSIS_SR)
            if sr != ANALYSIS_SR
            else raw_audio
        )

        # ------------------------------------------------------------------
        # STAGE 1 — Input Validation
        # ------------------------------------------------------------------
        is_valid, rejection_reason, validation_info = _validate_input_audio(
            raw_audio_16k, duration, _models
        )
        metrics: Dict = {
            "stage1_valid": is_valid,
            "stage1_reason": rejection_reason,
            "validation_info": validation_info,
        }

        if not is_valid:
            return {
                "decision": "Rejected (Input Validation)",
                "processed_bytes": audio_bytes,
                "metrics": metrics,
            }

        # ------------------------------------------------------------------
        # STAGE 2 — VoiceFixer Enhancement
        # ------------------------------------------------------------------
        vf_audio = normalize(raw_audio)
        if sr != VOICEFIXER_SR:
            vf_audio = librosa.resample(vf_audio, orig_sr=sr, target_sr=VOICEFIXER_SR)

        vf_out = normalize(
            _models.voicefixer.restore_inmem(
                vf_audio, mode=0, cuda=torch.cuda.is_available()
            )
        )
        # Resample to 16 kHz for Stage 3 assessment (matches Streamlit `processed`)
        processed = librosa.resample(vf_out, orig_sr=VOICEFIXER_SR, target_sr=ANALYSIS_SR)

        # ------------------------------------------------------------------
        # STAGE 3 — Output Quality Gate
        # ------------------------------------------------------------------

        # Temp files required by NISQA file API — per-request, not shared
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_wav(tmp.name, raw_audio_16k, ANALYSIS_SR)
            original_path = tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_wav(tmp.name, processed, ANALYSIS_SR)
            enhanced_path = tmp.name

        # Re-validate enhanced audio — feeds Tier 1 (dict) and Tier 2 (words)
        processed_duration = len(processed) / ANALYSIS_SR
        _, _, validation_info_enh = _validate_input_audio(processed, processed_duration, _models)

        # NISQA — feeds Tier 3
        nisqa_orig = _assess_nisqa(original_path, _models, "original")
        nisqa_enh = _assess_nisqa(enhanced_path, _models, "enhanced")

        # ScoreQ-NR dual-domain — feeds Tier 4
        scoreq_orig_result = _assess_scoreq(
            raw_audio_16k, ANALYSIS_SR, _models, is_synthetic=False, audio_label="original"
        )
        scoreq_enh_result = _assess_scoreq(
            processed, ANALYSIS_SR, _models, is_synthetic=True, audio_label="enhanced"
        )
        scoreq_orig_score = scoreq_orig_result["score"] if scoreq_orig_result else None
        scoreq_enh_score = scoreq_enh_result["score"] if scoreq_enh_result else None

        # Tier 1–6 hierarchical decision
        decision = decide_enhancement_quality(
            nisqa_orig,
            nisqa_enh,
            scoreq_orig_score,
            scoreq_enh_score,
            validation_info,
            validation_info_enh,
        )

        metrics["stage3"] = {
            "nisqa_orig": nisqa_orig,
            "nisqa_enh": nisqa_enh,
            "scoreq_orig": scoreq_orig_score,
            "scoreq_enh": scoreq_enh_score,
            "decision_rejected": decision["rejected"],
            "decision_reasons": decision.get("reasons", []),
            "decision_warnings": decision.get("warnings", []),
            "risk_score": decision.get("risk_score"),
            "benefit_score": decision.get("benefit_score"),
            "validation_info_enh": validation_info_enh,
        }

        # ------------------------------------------------------------------
        # Final audio selection
        # ------------------------------------------------------------------
        if decision["rejected"]:
            return {
                "decision": "Rejected",
                "processed_bytes": audio_bytes,
                "metrics": metrics,
            }

        # Write enhanced audio to temp file then read as bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            enhanced_temp_path = tmp.name
        
        try:
            # Ensure vf_out is 1D array
            if vf_out.ndim > 1:
                vf_out = vf_out.flatten()
            sf.write(enhanced_temp_path, vf_out.astype(np.float32), VOICEFIXER_SR, subtype='PCM_16')
            
            with open(enhanced_temp_path, 'rb') as f:
                enhanced_bytes = f.read()
        finally:
            if os.path.exists(enhanced_temp_path):
                os.unlink(enhanced_temp_path)
                
        return {
            "decision": "Accepted",
            "processed_bytes": enhanced_bytes,
            "metrics": metrics,
        }

    finally:
        for path in (input_temp_path, original_path, enhanced_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
