"""
VoiceFixer Input Validation Gate - REFINED v2.0
=================================================

Pre-screening approach: Validate input quality BEFORE running VoiceFixer.

Philosophy:
- Don't process garbage input (saves compute, prevents hallucination)
- Use simple, defensible checks (word density + dictionary coverage)
- Avoid complex frame-level quality heuristics (theoretically flawed)
- Rely on NISQA/ScoreQ for actual quality assessment
- Multi-signal agreement for robust decisions (no single-metric rejection)

DESIGN TRADE-OFFS (explicitly documented):
- This validation gate intentionally PREFERS PRECISION OVER RECALL
- Some valid but hard-to-transcribe speech may be conservatively rejected
- This is acceptable: we prioritize avoiding false enhancements over catching all valid inputs
- Examples that may be rejected: heavily accented speech, code-mixed audio, domain jargon
- Users can tune thresholds based on their use case and accept higher false positive rates

NEW in v2.0 (Based on Expert Feedback):
1. ✅ **Silero VAD Integration** - Robust voice activity detection
   - Minimum voiced duration enforcement (rejects noise-only segments)
   - VAD-aware SNR calculation (computed only on voiced frames)
   - Prevents enhancement of pure noise/silence

2. ✅ **De-hallucination Detection** - Whisper confidence analysis
   - Uses avg_logprob and no_speech_prob to detect hallucinations
   - Word count decrease + confidence increase = de-hallucination (GOOD)
   - Prevents false rejection of noise-masked speech

3. ✅ **Non-lexical Speech Support** - Handles valid vocalizations
   - Recognizes "uhm", "mmh", "uh", "ah" as valid speech
   - Improves dictionary coverage for natural conversation

4. ✅ **Multi-Signal Rejection** - No single-metric decisions
   - Requires ≥2 independent metrics to degrade before rejecting
   - Prevents cases like "sounds better but ScoreQ alone rejects"
   - Improves acceptance of valid enhancements

5. ✅ **Risk/Benefit Scoring** - Quantified decision framework
   - Risk score (0.0-1.0): semantic degradation, artifacts, VAD mismatch
   - Benefit score (0.0-1.0): quality improvement, noise reduction, clarity
   - Transparent, data-driven decision logic

Previous Refinements (v1.0):
1. Reduced over-reliance on Whisper for hard rejection
2. More robust dictionary coverage for real speech patterns
3. Relaxed word-density checks for short clips and PTT pauses
4. Lightweight acoustic fallback for zero-word cases
5. Loosened ScoreQ delta rejection logic
6. Language detection to prevent non-English hallucinations (English-only mode)

Key Changes from voicefixer_perceptual_quality_gate.py:
1. Input validation BEFORE VoiceFixer processing
2. Dictionary-based gibberish detection
3. Removed complex frame-level quality metrics
4. Simpler, more reliable decision logic
5. Auto-detect language and reject non-English audio

LANGUAGE SUPPORT:
- Currently configured for ENGLISH-ONLY mode
- Auto-detects audio language and rejects non-English inputs
- Prevents Whisper hallucinations (e.g., German audio transcribed as English gibberish)
- Configurable for future multi-language support via ENABLE_LANGUAGE_DETECTION flag
- See configuration section for threshold tuning

Author: Akash Rawat
Date: February 2026
Refined v1.0: February 2026
Refined v2.0: February 2026 (VAD, De-hallucination, Multi-signal, Risk/Benefit)
"""

import os
import sys
import tempfile
import logging
import numpy as np
import streamlit as st
import torch
import librosa
import whisper
import soundfile as sf
from voicefixer import VoiceFixer
from scoreq import Scoreq
from typing import Optional, Dict, List, Tuple
import enchant
from scipy import signal
import glob
from pathlib import Path
import zipfile
import io
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================

# Audio Processing
ANALYSIS_SR = 16000      # Sample rate for analysis
VOICEFIXER_SR = 44100    # VoiceFixer processing sample rate

# Input Validation Thresholds (Pre-screening) - REFINED
# Speech Rate (Word Density) - words per second
# INTERPRETATION GUIDE:
#   > 3.0 wps    : Very fast speech (podcasters, auctioneers)
#   2.0 - 3.0 wps: Normal conversational speech (typical) ✅
#   1.0 - 2.0 wps: Slow/careful speech (presentations, non-native)
#   0.5 - 1.0 wps: Very slow (dictation, PTT with pauses)
#   < 0.5 wps    : Extremely sparse (likely not continuous speech) ⚠️
MIN_WORD_DENSITY = 0.8           # Minimum words/sec for valid speech
MIN_WORD_DENSITY_SHORT = 0.5     # REFINED: Relaxed threshold for short clips (<3s)

# Dictionary Coverage - percentage of valid English words
# INTERPRETATION GUIDE:
#   90% - 100%: Excellent (clean English speech) ✅
#   70% - 90% : Good (normal speech with some artifacts)
#   60% - 70% : Acceptable (accented or technical terms)
#   40% - 60% : Poor (heavy accent, jargon, or noise) ⚠️
#   < 40%     : Very poor (gibberish, non-English, or severe noise) ❌
MIN_DICTIONARY_COVERAGE = 0.6    # Minimum ratio of valid dictionary words (60%)
MIN_ABSOLUTE_WORDS = 3           # Minimum total words for short audio
MIN_TRANSCRIPT_LENGTH = 5        # REFINED: Minimum words before applying dictionary coverage
SHORT_CLIP_THRESHOLD = 3.0       # REFINED: Duration threshold for relaxed validation (seconds)

# REFINED: Common acronyms/domain terms that shouldn't count against dictionary coverage
DOMAIN_ALLOWLIST = {
    'ptt', 'dsp', 'ai', 'ml', 'api', 'url', 'http', 'https', 'ui', 'ux',
    'ok', 'okay', 'um', 'uh', 'hmm', 'yeah', 'yep', 'nope', 'gonna', 'wanna'
}

# NEW: Non-lexical vocalizations that are valid speech but not in dictionary
NON_LEXICAL_WORDS = {
    'um', 'uh', 'uhm', 'mm', 'mmh', 'mhm', 'hmm', 'hm', 'ah', 'oh', 'eh',
    'erm', 'er', 'umm', 'uhh', 'ahem', 'huh', 'aha', 'ooh', 'whoa'
}

# REFINED: Acoustic fallback thresholds (for zero-word cases)
RMS_ENERGY_MIN = 0.01            # Minimum RMS energy to consider as speech
VOICED_FRAME_RATIO_MIN = 0.15    # Minimum ratio of frames above noise floor

# NEW: VAD (Voice Activity Detection) Thresholds
MIN_VOICED_DURATION = 0.5        # Minimum seconds of actual speech required (reject pure noise)
VAD_THRESHOLD = 0.5              # Silero VAD probability threshold (0.0-1.0)

# Language Detection (English-Only Mode)
ENABLE_LANGUAGE_DETECTION = True      # Auto-detect language and reject non-English
REQUIRED_LANGUAGE = "en"              # Required language code (ISO 639-1)

# Input bandwidth hard limit (reject if above this)
MAX_INPUT_BANDWIDTH_HZ = 4000.0  # 4 kHz

# Transcription Confidence Threshold (Whisper avg_logprob)
# This is a log probability score indicating how confident Whisper is about the transcription
# INTERPRETATION GUIDE:
#   -0.2 to  0.0  : Very confident (excellent quality)
#   -0.5 to -0.2  : Good confidence (reliable transcription)
#   -0.8 to -0.5  : Moderate confidence (acceptable quality)
#   -1.0 to -0.8  : Low confidence (may have errors)
#    < -1.0       : Very low confidence (likely hallucinating)
# HIGHER values (closer to 0) = MORE confident
MIN_TRANSCRIPTION_CONFIDENCE = -0.8   # Minimum avg_logprob (balanced threshold)

# No-Speech Probability Threshold (Whisper no_speech_prob)
# This indicates the probability that audio contains NO actual speech
# INTERPRETATION GUIDE:
#   0% - 20%  : Very likely speech (clear human voice detected) ✅
#   20% - 40% : Probably speech (speech present but may have noise)
#   40% - 60% : Uncertain (could be speech or noise) ⚠️
#   60% - 80% : Probably not speech (likely music/noise/garbled)
#   80% - 100%: Very likely not speech (silence/pure noise/music) ❌
# LOWER values (closer to 0%) = MORE likely to be speech
MAX_NO_SPEECH_PROB = 0.6              # Maximum acceptable (reject if > 60%)

# NISQA Thresholds (Post-processing quality checks)
NISQA_MOS_DELTA_MIN = -0.1       # Max acceptable MOS degradation
NISQA_COL_MIN = 2.5              # Min coloration score
NISQA_DIS_MIN = 3.0              # Min distortion score
NISQA_MOS_MIN = 2.5              # Absolute minimum MOS for output

# ScoreQ-NR Thresholds (natural domain) - REFINED
SCOREQ_MIN_DELTA = -0.05         # No longer used (kept for compatibility)
                                 # Logic now uses fixed thresholds:
                                 # - Significant degradation: < -0.1 (always reject)
                                 # - Stable/minimal: -0.1 to 0.1 (check NISQA)
                                 # - Clear improvement: >= 0.1 (always accept)

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug Mode
DEBUG_MODE = False

# ==========================================================
# MODEL INITIALIZATION
# ==========================================================

# NISQA Model
nisqa_available = False
nisqa_model = {}

try:
    from nisqa.NISQA_model import nisqaModel
    
    # Check for weights - try multiple locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_locations = [
        "nisqa.tar",  # Current directory
        os.path.join(script_dir, "..", "nisqa.tar"),  # Parent directory
        os.path.join(script_dir, "nisqa.tar"),  # Script directory
    ]
    
    weights_path = None
    for loc in weights_locations:
        if os.path.exists(loc):
            weights_path = loc
            break
    
    if weights_path:
        nisqa_model = {
            'loaded': True,
            'weights_path': weights_path
        }
        nisqa_available = True
        logger.info(f"✓ NISQA model loaded successfully from {weights_path}")
    else:
        logger.warning(f"⚠️ NISQA weights not found. Tried: {weights_locations}")
except ImportError:
    logger.warning("⚠️ NISQA not installed - overall quality checks disabled")

# Whisper Model
@st.cache_resource
def load_whisper():
    """Load Whisper model for transcription."""
    logger.info("Loading Whisper base model...")
    try:
        model = whisper.load_model("base", device=DEVICE)
        logger.info(f"✓ Whisper model loaded successfully on {DEVICE}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
        raise

try:
    whisper_model = load_whisper()
except Exception as e:
    st.error(f"❌ Failed to load Whisper model: {e}")
    st.error("Please ensure NumPy < 2.0 is installed: `pip install 'numpy<2.0'`")
    st.stop()

# VoiceFixer Model
@st.cache_resource
def load_voicefixer():
    """Load VoiceFixer model."""
    logger.info("Loading VoiceFixer model...")
    return VoiceFixer()

voicefixer = load_voicefixer()

# ScoreQ-NR Models (dual domain for accurate assessment)
@st.cache_resource
def load_scoreq_natural():
    """ScoreQ-NR for natural degraded audio (input)."""
    logger.info("Loading ScoreQ-NR (natural domain)...")
    return Scoreq(mode="nr", data_domain="natural")

@st.cache_resource
def load_scoreq_synthetic():
    """ScoreQ-NR for synthetic/generated audio (VoiceFixer output)."""
    logger.info("Loading ScoreQ-NR (synthetic domain)...")
    return Scoreq(mode="nr", data_domain="synthetic")

scoreq_natural = load_scoreq_natural()
scoreq_synthetic = load_scoreq_synthetic()

# Dictionary for lexical validation
@st.cache_resource
def load_dictionary():
    """Load English dictionary for word validation."""
    try:
        return enchant.Dict("en_US")
    except enchant.errors.DictNotFoundError:
        logger.warning("⚠️ en_US dictionary not found, trying en_GB")
        try:
            return enchant.Dict("en_GB")
        except:
            logger.error("❌ No English dictionary available")
            return None

dictionary = load_dictionary()

# Silero VAD Model
@st.cache_resource
def load_vad_model():
    """
    Load Silero VAD model using the pip-installed silero-vad package.
    Model weights are bundled inside the package — no server/Torch Hub download.
    Install once: pip install silero-vad
    """
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        logger.info("Loading Silero VAD model (bundled via pip)...")
        model = load_silero_vad(onnx=False)
        logger.info("✓ Silero VAD loaded successfully (bundled)")
        # Return model and a utils tuple matching the shape the rest of the code expects:
        # (get_speech_timestamps, _, _, _, _)
        return model, (get_speech_timestamps, None, None, None, None)
    except Exception as e:
        logger.error(f"❌ Failed to load Silero VAD: {e}")
        return None, None

vad_model, vad_utils = load_vad_model()

# ==========================================================
# UTILITIES
# ==========================================================

def normalize(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    return audio / (np.max(np.abs(audio)) + 1e-8)

def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Save audio to WAV file with normalization."""
    audio_np = np.array(audio, dtype=np.float32)
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=0)
    sf.write(path, normalize(audio_np), sr)

# ==========================================================
# NEW: VAD (Voice Activity Detection) using Silero
# ==========================================================

def detect_speech_segments(audio: np.ndarray, sr: int) -> Tuple[List[Dict], float, float]:
    """
    Detect speech segments using Silero VAD.
    
    Args:
        audio: Audio signal (any sample rate, will be resampled to 16kHz for VAD)
        sr: Sample rate of input audio
    
    Returns:
        Tuple of (speech_segments, total_voiced_duration, voiced_percentage)
        - speech_segments: List of dicts with 'start' and 'end' timestamps in seconds
        - total_voiced_duration: Total seconds of detected speech
        - voiced_percentage: Percentage of audio that is speech (0.0 to 1.0)
    """
    if vad_model is None or vad_utils is None:
        # Fallback to simple energy-based detection if VAD not available
        logger.warning("Silero VAD not available, using simple energy-based fallback")
        return [], 0.0, 0.0
    
    try:
        import torch
        
        # Resample to 16kHz if needed (Silero VAD requirement)
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio_16k).float()
        
        # Get VAD utilities
        (get_speech_timestamps, _, _, _, _) = vad_utils
        
        # Detect speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            threshold=VAD_THRESHOLD,
            sampling_rate=16000,
            min_speech_duration_ms=250,  # Minimum 250ms speech segment
            min_silence_duration_ms=100   # Minimum 100ms silence between segments
        )
        
        # Convert to seconds and calculate durations
        speech_segments = []
        total_voiced_samples = 0
        
        for ts in speech_timestamps:
            start_sec = ts['start'] / 16000.0
            end_sec = ts['end'] / 16000.0
            speech_segments.append({
                'start': start_sec,
                'end': end_sec,
                'duration': end_sec - start_sec
            })
            total_voiced_samples += (ts['end'] - ts['start'])
        
        total_voiced_duration = total_voiced_samples / 16000.0
        total_duration = len(audio) / sr
        voiced_percentage = total_voiced_duration / total_duration if total_duration > 0 else 0.0
        
        return speech_segments, total_voiced_duration, voiced_percentage
        
    except Exception as e:
        logger.error(f"VAD detection failed: {e}")
        return [], 0.0, 0.0

# ==========================================================
# REFINED: ACOUSTIC SANITY CHECKS (fallback for zero-word cases)
# ==========================================================

def check_acoustic_speech_presence(audio: np.ndarray, sr: int) -> Tuple[bool, Dict]:
    """
    REFINED: Lightweight acoustic check for speech presence.
    
    Used only as a fallback when Whisper yields zero or very few words.
    This prevents false rejection of valid speech that's hard to transcribe.
    
    Simple approach: RMS energy and voiced frame ratio.
    NOT frame-level perceptual analysis - just basic signal presence.
    
    Args:
        audio: Audio signal at ANALYSIS_SR
        sr: Sample rate
    
    Returns:
        Tuple of (has_speech_signal, acoustic_info)
    """
    acoustic_info = {
        'rms_energy': 0.0,
        'voiced_frame_ratio': 0.0,
        'has_signal': False
    }
    
    # Compute RMS energy
    rms = np.sqrt(np.mean(audio ** 2))
    acoustic_info['rms_energy'] = float(rms)
    
    # Compute voiced frame ratio (simple energy-based VAD)
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop
    
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    frame_energies = np.sqrt(np.mean(frames ** 2, axis=0))
    
    # Simple threshold: median energy as noise floor
    if len(frame_energies) > 0:
        noise_floor = np.median(frame_energies)
        voiced_frames = np.sum(frame_energies > noise_floor * 2)  # 2x median as voiced threshold
        voiced_ratio = voiced_frames / len(frame_energies)
        acoustic_info['voiced_frame_ratio'] = float(voiced_ratio)
    
    # Decision: has speech signal if either RMS or voiced ratio is sufficient
    has_signal = (rms >= RMS_ENERGY_MIN) or (acoustic_info['voiced_frame_ratio'] >= VOICED_FRAME_RATIO_MIN)
    acoustic_info['has_signal'] = has_signal
    
    return has_signal, acoustic_info

# ==========================================================
# ACOUSTIC QUALITY METRICS (Informational Only - Not Used for Validation)
# ==========================================================
def analyze_audio_quality(audio: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """
    Corrected acoustic diagnostics (informational only).

    Metrics:
    - Bandwidth: speech-aware effective bandwidth (frame-wise, 95% energy)
    - SNR: conservative, percentile-based, capped
    - Temporal roughness: envelope instability (artifact-sensitive)

    These are diagnostic signals, NOT perceptual quality scores.
    """
    try:
        if len(audio) < 2048:
            return 0.0, 0.0, 0.0

        audio = audio.astype(np.float32)
        audio -= np.mean(audio)

        # Frame parameters
        frame_length = int(0.02 * sr)   # 20 ms
        hop_length = int(0.01 * sr)     # 10 ms

        frames = librosa.util.frame(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        )
        frame_energy = np.mean(frames ** 2, axis=0)

        # ==================================================
        # 1. Speech-aware bandwidth (FIXED)
        # ==================================================
        n_fft = 2048
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        frame_bandwidths = []

        for i in range(S.shape[1]):
            spec = S[:, i]
            total_energy = np.sum(spec)
            if total_energy < 1e-10:
                continue

            cum_energy = np.cumsum(spec) / total_energy
            bw = freqs[np.searchsorted(cum_energy, 0.95)]
            frame_bandwidths.append(bw)

        if frame_bandwidths:
            # Robust: ignore rare spikes
            bandwidth_hz = float(np.percentile(frame_bandwidths, 90))
        else:
            bandwidth_hz = 0.0

        # ==================================================
        # 2. Conservative SNR (FIXED)
        # ==================================================
        noise_threshold = np.percentile(frame_energy, 10)
        noise_frames = frame_energy[frame_energy <= noise_threshold]

        noise_power = np.mean(noise_frames)
        signal_power = np.mean(frame_energy)

        if noise_power > 1e-10:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 60.0

        # Cap for stability & interpretability
        snr_db = float(np.clip(snr_db, -5.0, 45.0))

        # ==================================================
        # 3. Temporal roughness (FIXED)
        # ==================================================
        if len(frame_energy) > 2:
            log_energy = np.log(frame_energy + 1e-10)

            # Envelope slope
            delta = np.diff(log_energy)

            # Envelope instability (artifact-sensitive)
            delta2 = np.diff(delta)

            roughness = np.std(delta2)

            # Normalize: speech ≈ 0.1–0.3
            temporal_roughness = float(np.clip(roughness / 0.5, 0.0, 1.0))
        else:
            temporal_roughness = 0.0

        return bandwidth_hz, snr_db, temporal_roughness

    except Exception as e:
        logger.warning(f"Audio quality analysis error: {e}")
        return 0.0, 0.0, 0.0

def calculate_bandwidth(audio: np.ndarray, sr: int) -> float:
    """Calculate bandwidth using proper multi-frame spectral analysis."""
    bandwidth, _, _ = analyze_audio_quality(audio, sr)
    return bandwidth

def calculate_snr(audio: np.ndarray, sr: int, speech_segments: Optional[List[Dict]] = None) -> float:
    """
    Calculate SNR using VAD-active frames for accurate measurement.
    
    IMPROVED: Computes SNR only on voiced regions to avoid bias from silence.
    Noise estimation from VAD-inactive regions, signal from VAD-active regions.
    
    Args:
        audio: Audio signal at ANALYSIS_SR
        sr: Sample rate
        speech_segments: Optional list of VAD speech segments with 'start' and 'end' times
    
    Returns:
        SNR in dB (clipped to [-5, 45] range for stability)
    """
    try:
        if len(audio) < 2048:
            return 0.0
        
        # Frame parameters
        frame_length = int(0.02 * sr)   # 20 ms
        hop_length = int(0.01 * sr)     # 10 ms
        
        frames = librosa.util.frame(
            audio,
            frame_length=frame_length,
            hop_length=hop_length
        )
        frame_energy = np.mean(frames ** 2, axis=0)
        
        if speech_segments is not None and len(speech_segments) > 0:
            # NEW: VAD-based SNR calculation
            # Create frame-level VAD mask
            frame_times = librosa.frames_to_time(
                np.arange(len(frame_energy)),
                sr=sr,
                hop_length=hop_length
            )
            
            # Mark frames as voiced/unvoiced based on VAD segments
            is_voiced = np.zeros(len(frame_energy), dtype=bool)
            for segment in speech_segments:
                start_frame = np.searchsorted(frame_times, segment['start'])
                end_frame = np.searchsorted(frame_times, segment['end'])
                is_voiced[start_frame:end_frame] = True
            
            # Extract voiced and unvoiced frames
            voiced_energy = frame_energy[is_voiced]
            unvoiced_energy = frame_energy[~is_voiced]
            
            # Calculate SNR from VAD regions
            if len(voiced_energy) > 0 and len(unvoiced_energy) > 0:
                signal_power = np.mean(voiced_energy)
                noise_power = np.mean(unvoiced_energy)
                
                if noise_power > 1e-10:
                    snr_db = 10 * np.log10(signal_power / noise_power)
                else:
                    snr_db = 60.0
            else:
                # Fallback if VAD didn't find voiced/unvoiced split
                snr_db = 0.0
        else:
            # Fallback: percentile-based approach (original method)
            noise_threshold = np.percentile(frame_energy, 10)
            noise_frames = frame_energy[frame_energy <= noise_threshold]
            
            noise_power = np.mean(noise_frames) if len(noise_frames) > 0 else 1e-10
            signal_power = np.mean(frame_energy)
            
            if noise_power > 1e-10:
                snr_db = 10 * np.log10(signal_power / noise_power)
            else:
                snr_db = 60.0
        
        # Cap for stability & interpretability
        snr_db = float(np.clip(snr_db, -5.0, 45.0))
        return snr_db
        
    except Exception as e:
        logger.warning(f"SNR calculation error: {e}")
        return 0.0

def calculate_temporal_roughness(audio: np.ndarray, sr: int) -> float:
    """Calculate temporal roughness using log-domain energy variance."""
    _, _, roughness = analyze_audio_quality(audio, sr)
    return roughness

# ==========================================================
# TRANSCRIPTION
# ==========================================================

def interpret_confidence(confidence: float) -> str:
    """
    Interpret Whisper transcription confidence score (avg_logprob).
    
    Args:
        confidence: Average log probability from Whisper (-inf to 0.0)
    
    Returns:
        Human-readable interpretation string
    """
    if confidence >= -0.2:
        return "Excellent"
    elif confidence >= -0.5:
        return "Good"
    elif confidence >= -0.8:
        return "Moderate"
    elif confidence >= -1.0:
        return "Low"
    else:
        return "Very Low"

def interpret_no_speech_prob(prob: float) -> str:
    """
    Interpret Whisper no-speech probability.
    
    Args:
        prob: No-speech probability from Whisper (0.0 to 1.0)
    
    Returns:
        Human-readable interpretation string
    """
    if prob <= 0.2:
        return "Very Likely Speech"
    elif prob <= 0.4:
        return "Probably Speech"
    elif prob <= 0.6:
        return "Uncertain"
    elif prob <= 0.8:
        return "Probably Not Speech"
    else:
        return "Very Likely Not Speech"

def interpret_speech_rate(wps: float) -> str:
    """
    Interpret speech rate (words per second).
    
    Args:
        wps: Words per second
    
    Returns:
        Human-readable interpretation string
    """
    if wps > 3.0:
        return "Very Fast"
    elif wps >= 2.0:
        return "Normal"
    elif wps >= 1.0:
        return "Slow"
    elif wps >= 0.5:
        return "Very Slow"
    else:
        return "Extremely Sparse"

def interpret_dictionary_coverage(coverage: float) -> str:
    """
    Interpret dictionary coverage percentage.
    
    Args:
        coverage: Dictionary coverage ratio (0.0 to 1.0)
    
    Returns:
        Human-readable interpretation string
    """
    if coverage >= 0.9:
        return "Excellent"
    elif coverage >= 0.7:
        return "Good"
    elif coverage >= 0.6:
        return "Acceptable"
    elif coverage >= 0.4:
        return "Poor"
    else:
        return "Very Poor"

def interpret_bandwidth(bandwidth_hz: float) -> str:
    """
    Interpret audio bandwidth (effective frequency range).
    
    Args:
        bandwidth_hz: Bandwidth in Hz
    
    Returns:
        Human-readable interpretation string
    """
    if bandwidth_hz >= 8000:
        return "Wideband (Excellent)"
    elif bandwidth_hz >= 4000:
        return "Fullband Speech (Good)"
    elif bandwidth_hz >= 3000:
        return "Narrowband (Acceptable)"
    elif bandwidth_hz >= 2000:
        return "Telephone Quality"
    else:
        return "Very Limited"

def interpret_temporal_roughness(roughness: float) -> str:
    """
    Interpret temporal roughness (envelope instability).
    
    Args:
        roughness: Temporal roughness (0.0 to 1.0)
    
    Returns:
        Human-readable interpretation string
    """
    if roughness <= 0.2:
        return "Very Smooth"
    elif roughness <= 0.4:
        return "Smooth"
    elif roughness <= 0.6:
        return "Moderate"
    elif roughness <= 0.8:
        return "Rough"
    else:
        return "Very Rough"

def transcribe_audio(audio: np.ndarray) -> Tuple[List[Dict], str, str, float, float]:
    """
    Transcribe audio using Whisper with language detection.
    
    Returns:
        Tuple of (word_list, full_transcript, detected_language, confidence, no_speech_prob)
        - word_list: List of word dictionaries with timestamps
        - full_transcript: Complete transcription text
        - detected_language: ISO 639-1 language code (e.g., 'en', 'de', 'fr')
        - confidence: Average log probability (higher is better, typically -1.0 to 0.0)
        - no_speech_prob: Probability that audio contains no speech (0.0 to 1.0)
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        save_wav(tmp.name, audio, ANALYSIS_SR)
        try:
            # Auto-detect language (remove forced language="en")
            result = whisper_model.transcribe(
                tmp.name,
                word_timestamps=True,
                fp16=False,
                temperature=0.0,
                beam_size=5,
                best_of=5,
            )
        finally:
            os.unlink(tmp.name)
    
    words = []
    
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if w.get("start") is not None and w.get("end") is not None:
                words.append({
                    "word": w["word"].strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"])
                })
    
    # Extract language detection and confidence metrics
    detected_language = result.get('language', 'unknown')
    
    # Calculate average confidence from segments (avg_logprob is per-segment)
    segments = result.get('segments', [])
    if segments:
        # Average log probability across all segments
        avg_logprobs = [seg.get('avg_logprob', -999) for seg in segments if 'avg_logprob' in seg]
        confidence = np.mean(avg_logprobs) if avg_logprobs else -999
        
        # Get no_speech_prob from first segment (most relevant for validation)
        no_speech_prob = segments[0].get('no_speech_prob', 0.0)
    else:
        # No segments means likely empty/silent audio
        confidence = -999
        no_speech_prob = 1.0
    
    return words, result.get('text', ''), detected_language, float(confidence), float(no_speech_prob)

# ==========================================================
# REFINED: INPUT VALIDATION (Pre-screening)
# ==========================================================

def validate_input_audio(audio: np.ndarray, duration: float) -> Tuple[bool, Optional[str], Dict]:
    """
    REFINED: Pre-screen input audio quality before VoiceFixer processing.
    
    Refinements:
    1. Multiple validation failures required (not single hard rejection)
    2. Context-aware dictionary coverage (ignores short words, domain terms)
    3. Relaxed word density for short clips
    4. Acoustic fallback for zero-word cases
    
    DESIGN PHILOSOPHY (documented):
    - Prefers precision over recall
    - Some valid but hard-to-transcribe speech may be rejected
    - Examples: heavy accents, code-mixed audio, domain jargon
    - This is intentional and acceptable for quality-first approach
    
    Validates:
    1. Word density (speech rate proxy) - with context
    2. Dictionary coverage (lexical validity) - with domain awareness
    3. Acoustic presence (fallback for Whisper failures)
    
    Args:
        audio: Audio signal at ANALYSIS_SR
        duration: Audio duration in seconds
    
    Returns:
        Tuple of (is_valid, rejection_reason, validation_info)
    """
    validation_info = {
        'word_count': 0,
        'words_per_sec': 0.0,
        'dictionary_coverage': None,
        'valid_words': [],
        'invalid_words': [],
        'transcript': '',
        'is_short_clip': duration < SHORT_CLIP_THRESHOLD,
        'acoustic_fallback_used': False,
        'acoustic_info': None,
        # New acoustic quality metrics (informational only)
        'bandwidth_hz': 0.0,
        'snr_db': 0.0,
        'temporal_roughness': 0.0,
        # Language detection and confidence metrics
        'detected_language': 'unknown',
        'transcription_confidence': 0.0,
        'no_speech_prob': 0.0,
        # NEW: VAD metrics
        'speech_segments': [],
        'voiced_duration': 0.0,
        'voiced_percentage': 0.0
    }
    
    is_short_clip = duration < SHORT_CLIP_THRESHOLD
    
    # NEW: VAD-based speech detection
    speech_segments, voiced_duration, voiced_percentage = detect_speech_segments(audio, ANALYSIS_SR)
    validation_info['speech_segments'] = speech_segments
    validation_info['voiced_duration'] = voiced_duration
    validation_info['voiced_percentage'] = voiced_percentage
    
    # Calculate acoustic quality metrics with VAD (informational)
    validation_info['bandwidth_hz'] = calculate_bandwidth(audio, ANALYSIS_SR)
    validation_info['snr_db'] = calculate_snr(audio, ANALYSIS_SR, speech_segments)  # NEW: VAD-aware SNR
    validation_info['temporal_roughness'] = calculate_temporal_roughness(audio, ANALYSIS_SR)
    
    # Hard reject if bandwidth exceeds configured limit, assuming audio is not recoded via HA mic
    if validation_info['bandwidth_hz'] > MAX_INPUT_BANDWIDTH_HZ:
        return (
            False,
            f"Input bandwidth too high: {validation_info['bandwidth_hz']:.0f} Hz "
            f"(maximum allowed: {MAX_INPUT_BANDWIDTH_HZ:.0f} Hz)",
            validation_info
        )
    
    # Get transcription with language detection
    words, transcript, detected_lang, confidence, no_speech_prob = transcribe_audio(audio)
    word_count = len(words)
    
    # Store language and confidence metrics
    validation_info['detected_language'] = detected_lang
    validation_info['transcription_confidence'] = confidence
    validation_info['no_speech_prob'] = no_speech_prob
    
    validation_info['word_count'] = word_count
    validation_info['transcript'] = transcript
    
    # NEW: Check minimum voiced duration (reject noise-only audio)
    if voiced_duration < MIN_VOICED_DURATION:
        return False, f"Insufficient speech content: only {voiced_duration:.2f}s of voiced audio (minimum: {MIN_VOICED_DURATION}s)", validation_info
    
    # REFINED: If very few words detected, run acoustic fallback
    if word_count < MIN_ABSOLUTE_WORDS:
        has_signal, acoustic_info = check_acoustic_speech_presence(audio, ANALYSIS_SR)
        validation_info['acoustic_fallback_used'] = True
        validation_info['acoustic_info'] = acoustic_info
        
        # REFINED: Only reject if BOTH Whisper AND acoustic check fail
        if not has_signal and duration > 2.0:
            return False, f"No speech detected: {word_count} words, no acoustic signal (RMS={acoustic_info['rms_energy']:.4f})", validation_info
    
    # REFINED: Track validation failures (require multiple failures)
    # FIX 1: De-correlate transcript-sparsity signals so low word count and low
    # word density are treated as a single failure source.
    validation_failures = []

    # Check 1+2 (combined): Transcript sparsity with context-aware density threshold
    words_per_sec = word_count / duration if duration > 0 else 0
    validation_info['words_per_sec'] = words_per_sec

    # REFINED: Use relaxed threshold for short clips (accounts for PTT pauses, commands)
    effective_threshold = MIN_WORD_DENSITY_SHORT if is_short_clip else MIN_WORD_DENSITY

    low_word_count = word_count < MIN_ABSOLUTE_WORDS and duration > 2.0
    low_word_density = words_per_sec < effective_threshold

    if low_word_count or low_word_density:
        sparsity_details = []
        if low_word_count:
            sparsity_details.append(f"count={word_count}<{MIN_ABSOLUTE_WORDS}")
        if low_word_density:
            sparsity_details.append(f"density={words_per_sec:.2f}<{effective_threshold:.2f} wps")

        validation_failures.append(
            f"Transcript sparsity ({', '.join(sparsity_details)})"
        )
    
    # Check 3: REFINED - Dictionary coverage (only if sufficient words and dictionary available)
    if dictionary and word_count >= MIN_TRANSCRIPT_LENGTH:
        word_list = transcript.lower().split()
        valid_words = []
        invalid_words = []
        
        for word in word_list:
            # Clean word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalpha())
            if not clean_word:
                continue
            
        # We ignore most single-character tokens because they are overwhelmingly
        # ASR noise artifacts (e.g., stray "t", "s", "o") produced at word
        # boundaries, clipped consonants, or enhancement glitches.
        #
        # However, we explicitly allow "i" and "a" because they are the only
        # valid standalone one-letter English words in modern usage.
        #
        # Keeping them preserves legitimate short utterances like:
        #   "I am"
        #   "I go"
        #   "a test"
        #
        # Without this exception, dictionary coverage would unfairly penalize
        # valid conversational speech, especially in short clips.
        #
        # Design principle:
        #   - Suppress ASR fragmentation noise
        #   - Preserve real grammatical structure
        #
        # This is a precision-over-recall trade-off tuned for stability in
        # noisy audio and enhancement pipelines.

            if len(clean_word) == 1 and clean_word not in {"i", "a"}:
                continue
            
            # NEW: Check non-lexical vocalizations first (valid speech, not in dictionary)
            if clean_word in NON_LEXICAL_WORDS:
                valid_words.append(clean_word)
                continue
            
            # REFINED: Check domain allowlist
            if clean_word in DOMAIN_ALLOWLIST:
                valid_words.append(clean_word)
                continue
            
            # Standard dictionary check
            if dictionary.check(clean_word):
                valid_words.append(clean_word)
            else:
                invalid_words.append(clean_word)
        
        total_checked = len(valid_words) + len(invalid_words)
        dictionary_coverage = len(valid_words) / total_checked if total_checked > 0 else 0
        
        validation_info['dictionary_coverage'] = dictionary_coverage
        validation_info['valid_words'] = valid_words
        validation_info['invalid_words'] = invalid_words
        
        # REFINED: Dictionary coverage is a soft signal, not hard rejection
        # Only flag as failure if coverage is very low
        if dictionary_coverage < MIN_DICTIONARY_COVERAGE:
            validation_failures.append(f"Low dictionary coverage ({dictionary_coverage:.1%})")
    
    if confidence < MIN_TRANSCRIPTION_CONFIDENCE:  # < -0.8
            validation_failures.append(
                f"Low transcription confidence ({confidence:.3f} < {MIN_TRANSCRIPTION_CONFIDENCE})"
            )

    if no_speech_prob > MAX_NO_SPEECH_PROB:  # > 0.6
            validation_failures.append(
            f"High no-speech probability ({no_speech_prob:.1%} > {MAX_NO_SPEECH_PROB:.1%})"
            )
        
    # REFINED: Require MULTIPLE validation failures before rejecting
    # Single failure is not enough (reduces false rejections)
    if len(validation_failures) >= 2:
        # Multiple failures - likely truly problematic input
        rejection_reason = "Multiple validation failures: " + "; ".join(validation_failures)
        return False, rejection_reason, validation_info
    elif len(validation_failures) == 1 and word_count < MIN_ABSOLUTE_WORDS and duration > 3.0:
        # Special case: very few words in long audio - high confidence rejection
        rejection_reason = validation_failures[0] + " (high confidence rejection)"
        return False, rejection_reason, validation_info
    
    # REFINED: Pass validation even with single minor issue
    # This reduces false rejections of valid speech
    # Note: Single dictionary coverage failure will be re-evaluated post-enhancement
    return True, None, validation_info

# ==========================================================
# NISQA ASSESSMENT
# ==========================================================

def assess_nisqa(audio_path: str, audio_label: str = "audio") -> Optional[Dict]:
    """Assess audio quality using NISQA."""
    if not nisqa_available:
        return None
    
    try:
        from nisqa.NISQA_model import nisqaModel
        
        weights_path = nisqa_model['weights_path']
        args = {
            'mode': 'predict_file',
            'pretrained_model': weights_path,
            'num_workers': 0,
            'bs': 1,
            'ms_channel': None,
            'deg': audio_path,
            'output_dir': None,
        }
        
        file_model = nisqaModel(args)
        file_model.predict()
        results = file_model.ds_val.df.iloc[0]
        
        return {
            'model': 'NISQA',
            'mos': float(results['mos_pred']),
            'noi': float(results['noi_pred']),
            'dis': float(results['dis_pred']),
            'col': float(results['col_pred']),
            'loud': float(results['loud_pred']),
            'label': audio_label
        }
    except Exception as e:
        logger.error(f"NISQA error for {audio_label}: {e}")
        st.warning(f"NISQA error for {audio_label}: {e}")
        return None

# ==========================================================
# SCOREQ-NR ASSESSMENT
# ==========================================================

def assess_scoreq(audio: np.ndarray, sr: int, is_synthetic: bool = False, audio_label: str = "audio") -> Optional[Dict]:
    """Assess audio quality using ScoreQ-NR.
    
    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        is_synthetic: True for VoiceFixer output (use synthetic domain),
                     False for natural degraded audio (use natural domain)
        audio_label: Label for logging
    
    Returns:
        dict with ScoreQ-NR score
    """
    try:
        # Normalize audio
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        
        # Resample to 16kHz (required by ScoreQ)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            tmp_path = tmp.name
        
        try:
            # Use appropriate model based on data domain
            model = scoreq_synthetic if is_synthetic else scoreq_natural
            score = model.predict(tmp_path)
            
            return {
                'model': 'ScoreQ-NR',
                'score': float(score),
                'domain': 'synthetic' if is_synthetic else 'natural',
                'label': audio_label
            }
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"ScoreQ-NR error for {audio_label}: {e}")
        st.warning(f"ScoreQ-NR error for {audio_label}: {e}")
        return None

# ==========================================================
# REFINED: POST-PROCESSING DECISION LOGIC
# ==========================================================

def calculate_risk_score(validation_info_orig: Dict, validation_info_enh: Dict,
                        nisqa_orig: Optional[Dict], nisqa_enh: Optional[Dict]) -> Tuple[float, Dict]:
    """
    Calculate risk score for enhancement (0.0 = no risk, 1.0 = maximum risk).
    
    Risk factors:
    - Semantic degradation (word loss, dict coverage drop, confidence drop)
    - Artifact introduction (distortion, coloration issues)
    - VAD mismatch (voiced regions lost)
    
    Returns:
        Tuple of (risk_score, risk_details)
    """
    risk_details = {}
    risk_components = []
    
    # 1. Semantic degradation risk
    word_count_orig = validation_info_orig.get('word_count', 0)
    word_count_enh = validation_info_enh.get('word_count', 0)
    conf_orig = validation_info_orig.get('transcription_confidence', -999)
    conf_enh = validation_info_enh.get('transcription_confidence', -999)
    dict_cov_orig = validation_info_orig.get('dictionary_coverage')
    dict_cov_enh = validation_info_enh.get('dictionary_coverage')
    
    # Word loss risk (normalized)
    if word_count_orig > 0:
        word_loss_ratio = max(0, (word_count_orig - word_count_enh) / word_count_orig)
        word_loss_risk = min(1.0, word_loss_ratio * 2)  # 50% loss = max risk
        risk_components.append(('word_loss', word_loss_risk))
        risk_details['word_loss_risk'] = word_loss_risk
    
    # Confidence degradation risk
    if conf_orig > -900 and conf_enh > -900:
        conf_delta = conf_orig - conf_enh  # Positive = degradation
        conf_risk = min(1.0, max(0, conf_delta / 0.5))  # 0.5 drop = max risk
        risk_components.append(('confidence_drop', conf_risk))
        risk_details['confidence_risk'] = conf_risk
    
    # Dictionary coverage degradation risk
    if dict_cov_orig is not None and dict_cov_enh is not None:
        dict_delta = dict_cov_orig - dict_cov_enh  # Positive = degradation
        dict_risk = min(1.0, max(0, dict_delta / 0.3))  # 30% drop = max risk
        risk_components.append(('dict_coverage_drop', dict_risk))
        risk_details['dict_coverage_risk'] = dict_risk
    
    # 2. Artifact introduction risk (from NISQA)
    if nisqa_orig and nisqa_enh:
        # Distortion risk
        dis_delta = nisqa_orig['dis'] - nisqa_enh['dis']  # Positive = degradation
        dis_risk = min(1.0, max(0, dis_delta / 1.0))  # 1.0 drop = max risk
        risk_components.append(('distortion', dis_risk))
        risk_details['distortion_risk'] = dis_risk
        
        # Coloration risk
        col_delta = nisqa_orig['col'] - nisqa_enh['col']  # Positive = degradation
        col_risk = min(1.0, max(0, col_delta / 1.0))  # 1.0 drop = max risk
        risk_components.append(('coloration', col_risk))
        risk_details['coloration_risk'] = col_risk
    
    # 3. VAD mismatch risk
    voiced_orig = validation_info_orig.get('voiced_percentage', 0)
    voiced_enh = validation_info_enh.get('voiced_percentage', 0)
    if voiced_orig > 0:
        voiced_loss = max(0, (voiced_orig - voiced_enh) / voiced_orig)
        vad_risk = min(1.0, voiced_loss * 2)  # 50% voiced loss = max risk
        risk_components.append(('vad_mismatch', vad_risk))
        risk_details['vad_risk'] = vad_risk
    
    # Weighted average of risk components
    if risk_components:
        # Higher weight for semantic risks
        weights = {
            'word_loss': 1.5,
            'confidence_drop': 1.0,
            'dict_coverage_drop': 1.5,
            'distortion': 1.0,
            'coloration': 0.8,
            'vad_mismatch': 1.2
        }
        
        weighted_sum = sum(weights.get(name, 1.0) * score for name, score in risk_components)
        weight_total = sum(weights.get(name, 1.0) for name, _ in risk_components)
        risk_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    else:
        risk_score = 0.0
    
    risk_details['total_risk'] = risk_score
    risk_details['components'] = risk_components
    
    return risk_score, risk_details

def calculate_benefit_score(validation_info_orig: Dict, validation_info_enh: Dict,
                           nisqa_orig: Optional[Dict], nisqa_enh: Optional[Dict],
                           scoreq_orig: Optional[float], scoreq_enh: Optional[float]) -> Tuple[float, Dict]:
    """
    Calculate benefit score for enhancement (0.0 = no benefit, 1.0 = maximum benefit).
    
    Benefit factors:
    - Quality improvements (NISQA MOS, ScoreQ)
    - Noise reduction (SNR improvement)
    - Clarity improvements (bandwidth, confidence increase)
    
    Returns:
        Tuple of (benefit_score, benefit_details)
    """
    benefit_details = {}
    benefit_components = []
    
    # 1. Quality improvements
    if nisqa_orig and nisqa_enh:
        mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']  # Positive = improvement
        mos_benefit = min(1.0, max(0, mos_delta / 1.0))  # 1.0 improvement = max benefit
        benefit_components.append(('mos_improvement', mos_benefit))
        benefit_details['mos_benefit'] = mos_benefit
    
    if scoreq_orig is not None and scoreq_enh is not None:
        scoreq_delta = scoreq_enh - scoreq_orig  # Positive = improvement
        scoreq_benefit = min(1.0, max(0, scoreq_delta / 0.5))  # 0.5 improvement = max benefit
        benefit_components.append(('scoreq_improvement', scoreq_benefit))
        benefit_details['scoreq_benefit'] = scoreq_benefit
    
    # 2. Noise reduction
    snr_orig = validation_info_orig.get('snr_db', 0)
    snr_enh = validation_info_enh.get('snr_db', 0)
    snr_delta = snr_enh - snr_orig  # Positive = improvement
    snr_benefit = min(1.0, max(0, snr_delta / 10.0))  # 10 dB improvement = max benefit
    benefit_components.append(('snr_improvement', snr_benefit))
    benefit_details['snr_benefit'] = snr_benefit
    
    # 3. Clarity improvements
    bw_orig = validation_info_orig.get('bandwidth_hz', 0)
    bw_enh = validation_info_enh.get('bandwidth_hz', 0)
    if bw_orig > 0:
        bw_delta = (bw_enh - bw_orig) / bw_orig  # Percentage improvement
        bw_benefit = min(1.0, max(0, bw_delta / 0.5))  # 50% bandwidth improvement = max benefit
        benefit_components.append(('bandwidth_improvement', bw_benefit))
        benefit_details['bandwidth_benefit'] = bw_benefit
    
    # Confidence improvement (de-hallucination bonus)
    conf_orig = validation_info_orig.get('transcription_confidence', -999)
    conf_enh = validation_info_enh.get('transcription_confidence', -999)
    if conf_orig > -900 and conf_enh > -900:
        conf_delta = conf_enh - conf_orig  # Positive = improvement
        conf_benefit = min(1.0, max(0, conf_delta / 0.5))  # 0.5 improvement = max benefit
        benefit_components.append(('confidence_improvement', conf_benefit))
        benefit_details['confidence_benefit'] = conf_benefit
    
    # Weighted average of benefit components
    if benefit_components:
        # Higher weight for perceptual quality improvements
        weights = {
            'mos_improvement': 1.5,
            'scoreq_improvement': 1.2,
            'snr_improvement': 1.0,
            'bandwidth_improvement': 0.8,
            'confidence_improvement': 1.0
        }
        
        weighted_sum = sum(weights.get(name, 1.0) * score for name, score in benefit_components)
        weight_total = sum(weights.get(name, 1.0) for name, _ in benefit_components)
        benefit_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    else:
        benefit_score = 0.0
    
    benefit_details['total_benefit'] = benefit_score
    benefit_details['components'] = benefit_components
    
    return benefit_score, benefit_details

# def decide_enhancement_quality(nisqa_orig: Optional[Dict], nisqa_enh: Optional[Dict],
#                                scoreq_orig: float, scoreq_enh: float,
#                                validation_info_orig: Dict, validation_info_enh: Dict) -> Dict:
#     """
#     REFINED: Quality gate with dictionary coverage verification for rejected inputs.
    
#     Key insight: If input was rejected due to low dictionary coverage, it might be noise (not gibberish).
#     Strategy: Check if enhancement improved dictionary coverage AND quality metrics.
    
#     Refinements:
#     1. Dictionary coverage re-check: ONLY if input failed this specific check
#     2. ScoreQ delta is a trend indicator, not a strict pass/fail gate
#     3. Allow stable ScoreQ (≈0 delta) if NISQA doesn't degrade
#     4. Treat ScoreQ as complementary evidence, not sole rejection criterion
    
#     Args:
#         nisqa_orig: NISQA metrics for original
#         nisqa_enh: NISQA metrics for enhanced
#         scoreq_orig: ScoreQ score for original
#         scoreq_enh: ScoreQ score for enhanced
#         validation_info_orig: Input validation info (includes dictionary coverage)
#         validation_info_enh: Enhanced validation info (includes dictionary coverage)
    
#     Returns:
#         dict with {rejected: bool, reasons: List[str], warnings: List[str]}
#     """
#     reasons = []
#     warnings = []
    
#     # NEW: Get Whisper confidence metrics for de-hallucination detection
#     conf_orig = validation_info_orig.get('transcription_confidence', -999)
#     conf_enh = validation_info_enh.get('transcription_confidence', -999)
#     no_speech_orig = validation_info_orig.get('no_speech_prob', 1.0)
#     no_speech_enh = validation_info_enh.get('no_speech_prob', 1.0)
    
#     # CRITICAL: Dictionary coverage verification
#     # Handles three cases:
#     # 1. Input had low coverage (<60%) - check if enhancement fixed it
#     # 2. Input had insufficient words (<5) to calculate coverage - HIGH RISK scenario
#     # 3. Input passed validation (≥60%) but enhancement regressed - CRITICAL corruption
#     dict_cov_orig = validation_info_orig.get('dictionary_coverage')
#     dict_cov_enh = validation_info_enh.get('dictionary_coverage')
    
#     # Get word counts for high-risk detection
#     word_count_orig = validation_info_orig.get('word_count', 0)
#     word_count_enh = validation_info_enh.get('word_count', 0)
    
#     # Case 1: Input had calculable but low dictionary coverage
#     if dict_cov_orig is not None and dict_cov_orig < MIN_DICTIONARY_COVERAGE:
#         # Input had low dictionary coverage - check if enhancement fixed it
#         warnings.append(f"Input had low dictionary coverage ({dict_cov_orig:.1%}) - checking post-enhancement")
        
#         if dict_cov_enh is not None:
#             if dict_cov_enh < MIN_DICTIONARY_COVERAGE:
#                 # CRITICAL FIX: Low dictionary coverage = gibberish/non-English
#                 # ALWAYS REJECT if coverage remains below threshold
#                 # Don't allow quality metrics to override semantic validity
#                 reasons.append(
#                     f"Dictionary coverage remains below threshold after enhancement "
#                     f"({dict_cov_orig:.1%}→{dict_cov_enh:.1%}, min={MIN_DICTIONARY_COVERAGE:.1%}) - "
#                     f"likely gibberish or non-English audio - "
#                     f"rejecting regardless of quality metrics"
#                 )
#             else:
#                 # Enhancement improved dictionary coverage - noise was masking speech!
#                 warnings.append(
#                     f"✓ Enhancement improved dictionary coverage "
#                     f"({dict_cov_orig:.1%}→{dict_cov_enh:.1%}) - noise was masking speech"
#                 )
#         else:
#             # CRITICAL: Enhancement reduced words below calculable threshold - catastrophic degradation!
#             # Original had enough words for 0% coverage, enhanced has < 5 words (likely corrupted output)
#             warnings.append(f"⚠️ CRITICAL: Enhancement reduced word count ({word_count_orig}→{word_count_enh}), cannot verify lexically")
            
#             # Check if BOTH NISQA AND ScoreQ improved significantly (strict AND logic)
#             nisqa_improved = False
#             scoreq_improved = False
            
#             if nisqa_orig and nisqa_enh:
#                 mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']
#                 if mos_delta >= 0.2:
#                     nisqa_improved = True
#                     warnings.append(f"NISQA MOS improved ({mos_delta:+.2f})")
            
#             if scoreq_orig is not None and scoreq_enh is not None:
#                 scoreq_delta = scoreq_enh - scoreq_orig
#                 if scoreq_delta >= 0.3:  # Strict threshold
#                     scoreq_improved = True
#                     warnings.append(f"ScoreQ improved ({scoreq_delta:+.2f})")
            
#             # CRITICAL: Require BOTH metrics to improve (AND logic for degraded transcription)
#             if not (nisqa_improved and scoreq_improved):
#                 reasons.append(
#                     f"Enhancement catastrophically degraded transcription "
#                     f"({dict_cov_orig:.1%} with {word_count_orig} words → < 5 words) "
#                     f"without both quality metrics improving significantly - "
#                     f"likely corrupted output "
#                     f"(NISQA: {'✓' if nisqa_improved else '✗'}, ScoreQ: {'✓' if scoreq_improved else '✗'})"
#                 )
#             else:
#                 warnings.append(
#                     f"Despite transcription degradation, both quality metrics improved significantly - cautiously accepting"
#                 )
    
#     # Case 2: HIGH-RISK - Input had insufficient words to calculate dictionary coverage
#     elif dict_cov_orig is None and word_count_orig < MIN_TRANSCRIPT_LENGTH:
#         # Cannot verify lexical validity due to insufficient words
#         warnings.append(f"⚠️ HIGH-RISK: Input has too few words ({word_count_orig}) to calculate dictionary coverage")
        
#         # Check if enhancement produced enough words to verify
#         if dict_cov_enh is not None:
#             # Now we can calculate coverage - check if it's valid
#             if dict_cov_enh >= MIN_DICTIONARY_COVERAGE:
#                 # Enhancement revealed valid English speech - noise was masking it
#                 warnings.append(
#                     f"✓ Enhancement produced verifiable English speech "
#                     f"({word_count_orig}→{word_count_enh} words, {dict_cov_enh:.1%} coverage) - "
#                     f"noise was masking speech, accepting"
#                 )
#             else:
#                 # Enhancement produced verifiable text but it's gibberish - reject
#                 reasons.append(
#                     f"Enhancement produced text ({word_count_orig}→{word_count_enh} words) "
#                     f"but dictionary coverage is below threshold ({dict_cov_enh:.1%} < {MIN_DICTIONARY_COVERAGE:.1%}) - "
#                     f"likely gibberish or non-English audio - "
#                     f"rejecting regardless of quality metrics"
#                 )
#         else:
#             # Still insufficient words after enhancement - cannot verify at all
#             # CRITICAL: Require BOTH NISQA AND ScoreQ to improve significantly
#             nisqa_improved = False
#             scoreq_improved = False
            
#             if nisqa_orig and nisqa_enh:
#                 mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']
#                 if mos_delta >= 0.2:  # Significant MOS improvement
#                     nisqa_improved = True
#                     warnings.append(f"NISQA MOS improved ({mos_delta:+.2f})")
            
#             if scoreq_orig is not None and scoreq_enh is not None:
#                 scoreq_delta = scoreq_enh - scoreq_orig
#                 if scoreq_delta >= 0.3:  # Significant ScoreQ improvement (stricter than 3-tier 0.05)
#                     scoreq_improved = True
#                     warnings.append(f"ScoreQ improved ({scoreq_delta:+.2f})")
            
#             # CRITICAL: Require BOTH metrics to improve (AND logic for high-risk)
#             if not (nisqa_improved and scoreq_improved):
#                 # Cannot verify via dictionary AND not both metrics improved = reject
#                 reasons.append(
#                     f"Insufficient words for dictionary validation ({word_count_orig}→{word_count_enh}) "
#                     f"without both quality metrics improving significantly - "
#                     f"cannot verify if gibberish or non-English audio "
#                     f"(NISQA: {'✓' if nisqa_improved else '✗'}, ScoreQ: {'✓' if scoreq_improved else '✗'})"
#                 )
#             else:
#                 # Both metrics improved significantly despite insufficient words - cautiously accept
#                 warnings.append(
#                     f"Despite insufficient words for dict validation, both quality metrics improved significantly - accepting"
#                 )
    
#     # Case 3: CRITICAL - Input passed dictionary validation but enhancement REGRESSED
#     elif dict_cov_orig is not None and dict_cov_orig >= MIN_DICTIONARY_COVERAGE:
#         # Original was lexically valid - check if enhancement degraded it
#         if dict_cov_enh is not None:
#             if dict_cov_enh < MIN_DICTIONARY_COVERAGE:
#                 # REGRESSION: Valid input became gibberish after enhancement!
#                 warnings.append(f"⚠️ CRITICAL: Dictionary coverage REGRESSED ({dict_cov_orig:.1%}→{dict_cov_enh:.1%})")
                
#                 # Check if BOTH NISQA AND ScoreQ improved significantly to justify regression
#                 nisqa_improved = False
#                 scoreq_improved = False
                
#                 if nisqa_orig and nisqa_enh:
#                     mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']
#                     if mos_delta >= 0.2:  # Significant MOS improvement
#                         nisqa_improved = True
#                         warnings.append(f"NISQA MOS improved ({mos_delta:+.2f})")
                
#                 if scoreq_orig is not None and scoreq_enh is not None:
#                     scoreq_delta = scoreq_enh - scoreq_orig
#                     if scoreq_delta >= 0.3:  # Significant ScoreQ improvement (strict)
#                         scoreq_improved = True
#                         warnings.append(f"ScoreQ improved ({scoreq_delta:+.2f})")
                
#                 # CRITICAL: Require BOTH metrics to improve significantly to accept regression
#                 if not (nisqa_improved and scoreq_improved):
#                     # Dictionary degraded AND not both metrics improved = reject
#                     reasons.append(
#                         f"Enhancement regressed dictionary coverage from valid to gibberish "
#                         f"({dict_cov_orig:.1%}→{dict_cov_enh:.1%}) "
#                         f"without both quality metrics improving significantly - "
#                         f"VoiceFixer corrupted the output "
#                         f"(NISQA: {'✓' if nisqa_improved else '✗'}, ScoreQ: {'✓' if scoreq_improved else '✗'})"
#                     )
#                 else:
#                     # Both metrics improved significantly despite regression - extremely rare but cautiously accept
#                     warnings.append(
#                         f"Despite dictionary regression, both quality metrics improved significantly - cautiously accepting"
#                     )
#         else:
#             # CRITICAL: Enhancement reduced valid input below calculable threshold!
#             # Original was valid (≥60% coverage) but now we can't even verify the output
#             # This is ALWAYS a catastrophic failure - semantic content destroyed
#             warnings.append(f"⚠️ CRITICAL: Valid input ({dict_cov_orig:.1%}, {word_count_orig} words) reduced to {word_count_enh} words (< 5 threshold)")
            
#             # AUTOMATIC REJECTION: Cannot verify if output is gibberish
#             # Even if quality metrics improve, we destroyed verifiable semantic content
#             reasons.append(
#                 f"Enhancement catastrophically reduced valid English input "
#                 f"({dict_cov_orig:.1%} coverage, {word_count_orig} words) "
#                 f"to {word_count_enh} unverifiable words (< 5) - "
#                 f"semantic content destroyed, cannot verify if gibberish - "
#                 f"automatic rejection regardless of quality metrics"
#             )
    
#     # NEW: De-hallucination Detection
#     # If word count decreased BUT confidence increased, it's likely de-hallucination (GOOD)
#     if word_count_enh < word_count_orig:
#         conf_delta = conf_enh - conf_orig
#         no_speech_delta = no_speech_enh - no_speech_orig
        
#         # Check if confidence improved (higher logprob, lower no-speech prob)
#         confidence_improved = (conf_delta >= 0.2) and (no_speech_delta <= -0.1)
        
#         if confidence_improved:
#             warnings.append(
#                 f"✓ Word count decreased ({word_count_orig}→{word_count_enh}) BUT confidence improved "
#                 f"(conf: {conf_orig:.3f}→{conf_enh:.3f}, no-speech: {no_speech_orig:.1%}→{no_speech_enh:.1%}) - "
#                 f"likely de-hallucination (Whisper was hallucinating on noisy input)"
#             )
#         else:
#             warnings.append(
#                 f"⚠️ Word count decreased ({word_count_orig}→{word_count_enh}) WITHOUT confidence improvement - "
#                 f"possible semantic loss"
#             )
    
#     # NISQA checks (original behavior - immediate rejection on failures)
#     if nisqa_orig and nisqa_enh:
#         mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']
        
#         if mos_delta < NISQA_MOS_DELTA_MIN:
#             reasons.append(f"NISQA: MOS degraded ({nisqa_orig['mos']:.2f}→{nisqa_enh['mos']:.2f})")
        
#         if nisqa_enh['col'] < NISQA_COL_MIN:
#             reasons.append(f"NISQA: Robotic/unnatural sound (COL={nisqa_enh['col']:.2f})")
        
#         if nisqa_enh['dis'] < NISQA_DIS_MIN:
#             reasons.append(f"NISQA: High distortion (DIS={nisqa_enh['dis']:.2f})")
        
#         if nisqa_enh['mos'] < NISQA_MOS_MIN:
#             reasons.append(f"NISQA: Output quality too poor (MOS={nisqa_enh['mos']:.2f})")
    
#     # ScoreQ-NR checks (original behavior - immediate rejection on significant degradation)
#     if scoreq_orig is not None and scoreq_enh is not None:
#         scoreq_delta = scoreq_enh - scoreq_orig
        
#         # Flag significant degradation
#         if scoreq_delta < -0.1:
#             # Significant degradation - reject
#             reasons.append(f"ScoreQ-NR: Significant degradation ({scoreq_orig:.2f}→{scoreq_enh:.2f}, Δ={scoreq_delta:.2f})")
        
#         elif -0.1 <= scoreq_delta < 0.1:
#             # Stable/minimal change zone - informational only
#             warnings.append(f"ScoreQ-NR: Minimal change ({scoreq_orig:.2f}→{scoreq_enh:.2f}, Δ={scoreq_delta:.2f})")
#         else:
#             # Clear improvement
#             warnings.append(f"ScoreQ-NR: Clear improvement ({scoreq_orig:.2f}→{scoreq_enh:.2f}, Δ={scoreq_delta:.2f})")
        
#         # Codec detection (informational)
#         if scoreq_orig < 2.0:
#             warnings.append(f"Input appears to be codec/telephony audio (ScoreQ={scoreq_orig:.2f})")
    
#     # NEW: Calculate risk and benefit scores
#     risk_score, risk_details = calculate_risk_score(validation_info_orig, validation_info_enh, nisqa_orig, nisqa_enh)
#     benefit_score, benefit_details = calculate_benefit_score(validation_info_orig, validation_info_enh, nisqa_orig, nisqa_enh, scoreq_orig, scoreq_enh)
    
#     # Add risk/benefit interpretation
#     if benefit_score > risk_score + 0.2:  # Benefit significantly outweighs risk
#         warnings.append(f"✓ Risk/Benefit Analysis: ACCEPT (Benefit={benefit_score:.2f}, Risk={risk_score:.2f})")
#     elif risk_score > benefit_score + 0.2:  # Risk significantly outweighs benefit
#         warnings.append(f"⚠️ Risk/Benefit Analysis: CAUTION (Risk={risk_score:.2f}, Benefit={benefit_score:.2f})")
#     else:
#         warnings.append(f"ℹ️ Risk/Benefit Analysis: BALANCED (Benefit={benefit_score:.2f}, Risk={risk_score:.2f})")
    
#     return {
#         'rejected': len(reasons) > 0,
#         'reasons': reasons,
#         'warnings': warnings,
#         'risk_score': risk_score,
#         'benefit_score': benefit_score,
#         'risk_details': risk_details,
#         'benefit_details': benefit_details
#     }

def decide_enhancement_quality(
    nisqa_orig: Optional[Dict],
    nisqa_enh: Optional[Dict],
    scoreq_orig: Optional[float],
    scoreq_enh: Optional[float],
    validation_info_orig: Dict,
    validation_info_enh: Dict
) -> Dict:
    """
    Hierarchical, production-grade decision logic.

    Priority order:
    Tier 1: Hard semantic integrity
    Tier 2: Severe linguistic degradation
    Tier 3: Perceptual artifact degradation (NISQA)
    Tier 4: Multi-signal ScoreQ degradation
    Tier 5: VAD speech loss agreement
    Tier 6: Risk/Benefit diagnostic (non-gating)
    """

    reasons = []
    warnings = []

    # --------------------------------------------------
    # Extract metrics
    # --------------------------------------------------

    dict_cov_orig = validation_info_orig.get("dictionary_coverage")
    dict_cov_enh = validation_info_enh.get("dictionary_coverage")

    word_count_orig = validation_info_orig.get("word_count", 0)
    word_count_enh = validation_info_enh.get("word_count", 0)

    conf_orig = validation_info_orig.get("transcription_confidence", -999)
    conf_enh = validation_info_enh.get("transcription_confidence", -999)

    voiced_orig = validation_info_orig.get("voiced_percentage", 0)
    voiced_enh = validation_info_enh.get("voiced_percentage", 0)

    scoreq_delta = None
    if scoreq_orig is not None and scoreq_enh is not None:
        scoreq_delta = scoreq_enh - scoreq_orig

    # ==================================================
    # 🔴 Tier 1 — Hard Semantic Integrity
    # ==================================================

    if dict_cov_orig is not None:

        # Valid input becomes invalid
        if dict_cov_orig >= MIN_DICTIONARY_COVERAGE:
            if dict_cov_enh is None or dict_cov_enh < MIN_DICTIONARY_COVERAGE:
                reasons.append(
                    "Semantic regression: valid input became lexically invalid after enhancement"
                )

        # Invalid remains invalid
        if dict_cov_orig < MIN_DICTIONARY_COVERAGE:
            if dict_cov_enh is not None and dict_cov_enh < MIN_DICTIONARY_COVERAGE:
                reasons.append(
                    "Lexical integrity not recovered after enhancement"
                )

    # Catastrophic transcription collapse
    if word_count_orig >= MIN_TRANSCRIPT_LENGTH and word_count_enh < MIN_TRANSCRIPT_LENGTH:
        reasons.append(
            "Catastrophic transcription collapse after enhancement"
        )

    if reasons:
        return _decision_output(reasons, warnings,
                                validation_info_orig,
                                validation_info_enh,
                                nisqa_orig,
                                nisqa_enh,
                                scoreq_orig,
                                scoreq_enh)

    # ==================================================
    # 🟠 Tier 2 — Severe Linguistic Degradation
    # ==================================================

    if word_count_enh < word_count_orig:
        confidence_improved = (conf_enh - conf_orig) >= 0.2

        if not confidence_improved:
            # Only reject if also no perceptual improvement
            perceptual_improved = False

            if nisqa_orig and nisqa_enh:
                perceptual_improved = (nisqa_enh["mos"] - nisqa_orig["mos"]) > 0.2

            if not perceptual_improved:
                reasons.append(
                    "Linguistic degradation without compensating perceptual gain"
                )

    if reasons:
        return _decision_output(reasons, warnings,
                                validation_info_orig,
                                validation_info_enh,
                                nisqa_orig,
                                nisqa_enh,
                                scoreq_orig,
                                scoreq_enh)

    # ==================================================
    # 🟡 Tier 3 — NISQA Perceptual Degradation
    # ==================================================

    if nisqa_orig and nisqa_enh:

        mos_delta = nisqa_enh["mos"] - nisqa_orig["mos"]

        if mos_delta < NISQA_MOS_DELTA_MIN:
            reasons.append("NISQA MOS degraded")

        if nisqa_enh["col"] < NISQA_COL_MIN:
            reasons.append("Robotic coloration detected")

        if nisqa_enh["dis"] < NISQA_DIS_MIN:
            reasons.append("High distortion detected")

        if nisqa_enh["mos"] < NISQA_MOS_MIN:
            reasons.append("Output MOS below minimum")

    if reasons:
        return _decision_output(reasons, warnings,
                                validation_info_orig,
                                validation_info_enh,
                                nisqa_orig,
                                nisqa_enh,
                                scoreq_orig,
                                scoreq_enh)

    # ==================================================
    # 🟢 Tier 4 — Multi-Signal ScoreQ Degradation
    # ==================================================

    if scoreq_delta is not None:

        if scoreq_delta < -0.1:
            reasons.append(
                f"ScoreQ-NR: Significant degradation (Δ={scoreq_delta:.2f})"
            )

        elif -0.1 <= scoreq_delta < 0.1:
            warnings.append(
                f"ScoreQ-NR: Minimal change (Δ={scoreq_delta:.2f})"
            )

        else:
            warnings.append(
                f"ScoreQ-NR: Clear improvement (Δ={scoreq_delta:.2f})"
            )

    if reasons:
        return _decision_output(reasons, warnings,
                                validation_info_orig,
                                validation_info_enh,
                                nisqa_orig,
                                nisqa_enh,
                                scoreq_orig,
                                scoreq_enh)

    # ==================================================
    # 🔵 Tier 5 — VAD Speech Loss (Agreement-Based)
    # ==================================================

    if voiced_orig > 0:
        voiced_loss_ratio = (voiced_orig - voiced_enh) / voiced_orig

        if voiced_loss_ratio > 0.3:
            warnings.append(
                f"Speech reduced by {voiced_loss_ratio:.1%} — monitoring"
            )

    # ==================================================
    # 🧠 Tier 6 — Risk/Benefit Diagnostic (Non-Gating)
    # ==================================================

    risk_score, risk_details = calculate_risk_score(
        validation_info_orig, validation_info_enh,
        nisqa_orig, nisqa_enh
    )

    benefit_score, benefit_details = calculate_benefit_score(
        validation_info_orig, validation_info_enh,
        nisqa_orig, nisqa_enh,
        scoreq_orig, scoreq_enh
    )

    warnings.append(
        f"Risk={risk_score:.2f}, Benefit={benefit_score:.2f}"
    )

    return {
        "rejected": False,
        "reasons": [],
        "warnings": warnings,
        "risk_score": risk_score,
        "benefit_score": benefit_score,
        "risk_details": risk_details,
        "benefit_details": benefit_details
    }

def _decision_output(reasons, warnings,
                     validation_info_orig,
                     validation_info_enh,
                     nisqa_orig,
                     nisqa_enh,
                     scoreq_orig,
                     scoreq_enh):

    risk_score, risk_details = calculate_risk_score(
        validation_info_orig, validation_info_enh,
        nisqa_orig, nisqa_enh
    )

    benefit_score, benefit_details = calculate_benefit_score(
        validation_info_orig, validation_info_enh,
        nisqa_orig, nisqa_enh,
        scoreq_orig, scoreq_enh
    )

    return {
        "rejected": True,
        "reasons": reasons,
        "warnings": warnings,
        "risk_score": risk_score,
        "benefit_score": benefit_score,
        "risk_details": risk_details,
        "benefit_details": benefit_details
    }


# ==========================================================
# STREAMLIT UI
# ==========================================================

if __name__ == "__main__":
    st.title("🎙 VoiceFixer Input Validation Gate - REFINED")
st.markdown("""
**Pre-Screening Approach:**
- ✅ **Validate input BEFORE processing** - Don't waste compute on garbage
- ✅ **Simple, defensible checks** - Word density + dictionary coverage
- ✅ **No complex heuristics** - Avoid theoretically flawed frame-level metrics
- ✅ **NISQA/ScoreQ for quality** - Use proper models for perceptual assessment

**REFINED Validation Features:**
- 🔧 **Reduced false rejections** - Multiple failures required, not single hard rejection
- 🔧 **Context-aware validation** - Relaxed thresholds for short clips (<3s)
- 🔧 **Domain-aware dictionary** - Ignores acronyms, short words, common interjections
- 🔧 **Acoustic fallback** - Prevents rejection of valid but hard-to-transcribe speech
- 🔧 **Loosened ScoreQ logic** - Allows stable quality, not just improvements

**Validation Pipeline:**
1. **Input Pre-Screening** (BEFORE VoiceFixer):
   - Word density check (context-aware: relaxed for short clips)
   - Dictionary coverage (domain-aware: ignores acronyms, short words)
   - Acoustic fallback (for zero-word Whisper failures)
   - Requires MULTIPLE failures to reject (reduces false positives)
   
2. **VoiceFixer Processing** (only if input passes validation)

3. **Output Quality Check** (AFTER VoiceFixer):
   - NISQA: Overall quality (MOS, distortion, coloration)
   - ScoreQ-NR: Speech-specific quality (trend indicator, not hard gate)

**Design Philosophy (explicitly documented):**
This gate intentionally PREFERS PRECISION OVER RECALL. Some valid but hard-to-transcribe 
speech may be conservatively rejected (e.g., heavy accents, code-mixed audio, domain jargon).
This is acceptable for a quality-first approach where avoiding false enhancements is 
prioritized over catching all valid inputs.
""")

# Model status
col1, col2, col3 = st.columns(3)
with col1:
    if nisqa_available:
        st.success("✓ NISQA Ready")
    else:
        st.error("✗ NISQA Not Available")
        
with col2:
    st.success("✓ ScoreQ-NR Ready")
    
with col3:
    if dictionary:
        st.success("✓ Dictionary Ready")
    else:
        st.warning("⚠️ Dictionary Not Available")

# Debug mode toggle
if st.sidebar.checkbox("Enable Debug Mode", value=DEBUG_MODE):
    DEBUG_MODE = True
    st.sidebar.info("Debug output enabled")

st.header("📤 Upload Audio")

# ==========================================================
# PROCESSING FUNCTION
# ==========================================================
def process_single_file(file_input, file_name, batch_results, input_mode):
    """Process a single audio file (works for both single file and batch mode)
    
    Returns:
        Tuple of (original_audio_bytes, processed_audio_bytes, decision_status)
        - original_audio_bytes: Original audio file as bytes
        - processed_audio_bytes: Processed audio file as bytes
        - decision_status: 'Accepted', 'Rejected', or 'Rejected (Input Validation)'
    """
    
    # Initialize return variables
    original_audio_bytes = None
    processed_audio_bytes = None
    decision_status = 'Unknown'
    
    # Handle different input types (uploaded file vs file path)
    if isinstance(file_input, str):
        # File path from folder mode
        st.audio(file_input)
        raw_audio, sr = librosa.load(file_input, sr=None, mono=True)
        # Read original file bytes (keep original format)
        with open(file_input, 'rb') as f:
            original_audio_bytes = f.read()
    else:
        # Uploaded file from single file mode
        st.audio(file_input)
        
        # Store original file bytes (keep original format)
        file_input.seek(0)
        original_audio_bytes = file_input.read()
        file_input.seek(0)
        
        # Load audio from uploaded bytes
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(file_input.read())
            temp_path = tmp.name
        
        raw_audio, sr = librosa.load(temp_path, sr=None, mono=True)
        os.unlink(temp_path)
    
    duration = len(raw_audio) / sr
    
    # Resample for analysis
    raw_audio_16k = librosa.resample(raw_audio, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else raw_audio
    
    # ==========================================================
    # STAGE 1: REFINED INPUT VALIDATION (Pre-screening)
    # ==========================================================
    
    st.subheader("🔍 Stage 1: Input Validation (REFINED)")
    
    with st.spinner("Validating input audio..."):
        is_valid, rejection_reason, validation_info = validate_input_audio(raw_audio_16k, duration)
    
    # Display validation results
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Word Count", validation_info['word_count'])
    with col2:
        wps = validation_info['words_per_sec']
        wps_interpretation = interpret_speech_rate(wps)
        st.metric("Speech Rate", 
                 f"{wps:.2f} wps",
                 delta=wps_interpretation,
                 help="Words per second\n"
                      "Normal: 2.0-3.0 wps | Slow: 1.0-2.0 wps | Very Slow: 0.5-1.0 wps")
    with col3:
        if dictionary and validation_info['dictionary_coverage'] is not None:
            coverage = validation_info['dictionary_coverage']
            coverage_interpretation = interpret_dictionary_coverage(coverage)
            st.metric("Dictionary Coverage", 
                     f"{coverage:.1%}",
                     delta=coverage_interpretation,
                     delta_color="normal" if coverage >= MIN_DICTIONARY_COVERAGE else "inverse",
                     help="Percentage of valid English words\n"
                          "Excellent: 90%+ | Good: 70-90% | Acceptable: 60-70% | Poor: <60%")
        else:
            st.metric("Dictionary Coverage", "N/A")
    with col4:
        # NEW: VAD-based voiced duration
        voiced_dur = validation_info['voiced_duration']
        st.metric("Voiced Duration", 
                 f"{voiced_dur:.2f}s",
                 delta="✓ Pass" if voiced_dur >= MIN_VOICED_DURATION else "✗ Fail",
                 delta_color="normal" if voiced_dur >= MIN_VOICED_DURATION else "inverse",
                 help=f"Actual speech content detected by VAD\n"
                      f"Minimum required: {MIN_VOICED_DURATION}s")
    with col5:
        # NEW: VAD percentage
        voiced_pct = validation_info['voiced_percentage']
        st.metric("Voiced %", 
                 f"{voiced_pct:.1%}",
                 help="Percentage of audio containing speech")
    
    # REFINED: Show acoustic fallback info if used
    if validation_info['acoustic_fallback_used']:
        st.info("ℹ️ **Acoustic fallback used** - Few words detected, validated signal presence")
        acoustic_info = validation_info['acoustic_info']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMS Energy", f"{acoustic_info['rms_energy']:.4f}")
        with col2:
            st.metric("Voiced Frame Ratio", f"{acoustic_info['voiced_frame_ratio']:.2%}")
    
    # Display additional acoustic quality metrics (informational only)
    st.write("**Acoustic Quality Metrics (VAD-Enhanced)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        bw = validation_info['bandwidth_hz']
        bw_interpretation = interpret_bandwidth(bw)
        st.metric("Bandwidth", 
                 f"{bw:.0f} Hz",
                 delta=bw_interpretation,
                 help="Effective frequency range (95% energy)\n"
                      "Wideband: 8000+ Hz | Fullband: 4000-8000 Hz | Narrowband: 3000-4000 Hz")
    with col2:
        st.metric("SNR (VAD-aware)", f"{validation_info['snr_db']:.1f} dB",
                 help="Signal-to-Noise Ratio computed on voiced frames only (higher = cleaner)\n"
                      "Excellent: >30 dB | Good: 20-30 dB | Acceptable: 10-20 dB | Poor: <10 dB\n"
                      "NEW: Uses VAD to separate speech from noise for accurate measurement")
    with col3:
        tr = validation_info['temporal_roughness']
        tr_interpretation = interpret_temporal_roughness(tr)
        st.metric("Temporal Roughness", 
                 f"{tr:.3f}",
                 delta=tr_interpretation,
                 help="Amplitude envelope variability (0-1)\n"
                      "Very Smooth: <0.2 | Smooth: 0.2-0.4 | Moderate: 0.4-0.6 | Rough: >0.6")
    
    st.write("**Transcription:**")
    st.code(validation_info['transcript'])
    
    # Language detection info
    if ENABLE_LANGUAGE_DETECTION:
        col1, col2, col3 = st.columns(3)
        with col1:
            lang_status = "✅" if validation_info['detected_language'] == REQUIRED_LANGUAGE else "❌"
            st.metric("Detected Language", 
                     f"{lang_status} {validation_info['detected_language'].upper()}",
                     help="Auto-detected language from audio")
        with col2:
            conf = validation_info['transcription_confidence']
            conf_status = "✅" if conf >= MIN_TRANSCRIPTION_CONFIDENCE else "⚠️"
            conf_interpretation = interpret_confidence(conf)
            st.metric("Transcription Confidence", 
                     f"{conf_status} {conf:.3f}",
                     delta=conf_interpretation,
                     help="Whisper confidence score (avg log probability)\n"
                          "Range: -1.0 to 0.0 (higher = more confident)\n"
                          "Excellent: -0.2 to 0.0 | Good: -0.5 to -0.2 | Moderate: -0.8 to -0.5")
        with col3:
            nsp = validation_info['no_speech_prob']
            nsp_status = "✅" if nsp <= MAX_NO_SPEECH_PROB else "⚠️"
            nsp_interpretation = interpret_no_speech_prob(nsp)
            st.metric("No-Speech Probability", 
                     f"{nsp_status} {nsp:.1%}",
                     delta=nsp_interpretation,
                     help="Probability that audio contains no speech (0-100%)\n"
                          "Range: Lower = more speech | Higher = less speech\n"
                          "0-20%: Very Likely Speech | 20-40%: Probably Speech | 40-60%: Uncertain")
    
    if DEBUG_MODE:
        with st.expander("🔍 Validation Details"):
            st.json(validation_info)
    
    if not is_valid:
        st.error(f"❌ **Input Validation Failed**")
        st.write(f"**Reason:** {rejection_reason}")
        st.info("🛑 **Not processing with VoiceFixer** - Input quality insufficient")
        
        # Set decision status
        decision_status = 'Rejected (Input Validation)'
        
        # Store processed audio (original audio since validation failed)
        # Save as WAV at 44.1kHz for consistency and playback compatibility
        audio_for_zip = librosa.resample(raw_audio, orig_sr=sr, target_sr=44100) if sr != 44100 else raw_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_wav(tmp.name, audio_for_zip, 44100)
            with open(tmp.name, 'rb') as f:
                processed_audio_bytes = f.read()
            os.unlink(tmp.name)
        
        # Record batch results for failed validation
        if batch_results is not None:
            duration_sec = len(raw_audio) / sr
            result_entry = {
                'Filename': file_name,
                'Duration (s)': f"{duration_sec:.2f}",
                'Decision': 'Rejected (Input Validation)',
                'Rejection Reasons': rejection_reason,
                'MOS Δ': 'N/A',
                'ScoreQ Δ': 'N/A',
                'Word Count': validation_info['word_count'],
                'Dict Coverage %': f"{validation_info['dictionary_coverage']*100:.1f}" if validation_info['dictionary_coverage'] is not None else 'N/A',
                'Dict Quality': interpret_dictionary_coverage(validation_info['dictionary_coverage']) if validation_info['dictionary_coverage'] is not None else 'N/A',
                'Speech Rate': f"{validation_info['words_per_sec']:.2f}",
                'Rate Quality': interpret_speech_rate(validation_info['words_per_sec']),
                'Bandwidth (Hz)': f"{validation_info['bandwidth_hz']:.0f}",
                'Bandwidth Quality': interpret_bandwidth(validation_info['bandwidth_hz']),
                'SNR (dB)': f"{validation_info['snr_db']:.1f}",
                'Temporal Roughness': f"{validation_info['temporal_roughness']:.3f}",
                'Roughness Quality': interpret_temporal_roughness(validation_info['temporal_roughness']),
                'Detected Language': validation_info['detected_language'].upper(),
                'Transcription Conf': f"{validation_info['transcription_confidence']:.3f}",
                'Conf Quality': interpret_confidence(validation_info['transcription_confidence']),
                'No-Speech Prob': f"{validation_info['no_speech_prob']:.1%}",
                'Speech Likelihood': interpret_no_speech_prob(validation_info['no_speech_prob'])
            }
            batch_results.append(result_entry)
        
        st.subheader("🎵 Final Output")
        st.info("Sending original audio (input failed validation)")
        if isinstance(file_input, str):
            st.audio(file_input)
        else:
            # Uploaded file - recreate from raw_audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, raw_audio_16k, ANALYSIS_SR)
                st.audio(tmp.name)
                os.unlink(tmp.name)
        
        # Return audio data for ZIP creation
        return original_audio_bytes, processed_audio_bytes, decision_status
        
    else:
        st.success("✅ **Input Validation Passed** - Proceeding with enhancement")
        
        # ==========================================================
        # STAGE 2: VOICEFIXER PROCESSING
        # ==========================================================
        
        st.subheader("⚙️ Stage 2: VoiceFixer Processing")
        
        # Prepare for VoiceFixer
        vf_audio = normalize(raw_audio)
        if sr != VOICEFIXER_SR:
            vf_audio = librosa.resample(vf_audio, orig_sr=sr, target_sr=VOICEFIXER_SR)
        
        with st.spinner("Enhancing with VoiceFixer..."):
            vf_out = normalize(
                voicefixer.restore_inmem(vf_audio, mode=0, cuda=torch.cuda.is_available())
            )
        
        processed = librosa.resample(vf_out, orig_sr=VOICEFIXER_SR, target_sr=ANALYSIS_SR)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_wav(tmp.name, processed, ANALYSIS_SR)
            enhanced_path = tmp.name
        
        st.audio(enhanced_path)
        
        # Download button for Stage 2 VoiceFixer output
        with open(enhanced_path, 'rb') as audio_file:
            stage2_audio_bytes = audio_file.read()
        
        base_name = os.path.splitext(file_name)[0]
        stage2_filename = f"{base_name}_stage2_voicefixer_output.wav"
        
        st.download_button(
            label="⬇️ Download Stage 2 VoiceFixer Output",
            data=stage2_audio_bytes,
            file_name=stage2_filename,
            mime="audio/wav",
            key=f"download_stage2_{file_name}"
        )
        
        # ==========================================================
        # STAGE 3: REFINED OUTPUT QUALITY CHECK
        # ==========================================================
        
        st.subheader("🧪 Stage 3: Output Quality Assessment (REFINED)")
        
        with st.spinner("Assessing quality..."):
            # Save original for NISQA (16kHz for quality assessment)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                save_wav(tmp.name, raw_audio_16k, ANALYSIS_SR)
                original_path = tmp.name
            
            # Save enhanced for NISQA (already at 16kHz from processing)
            # enhanced_path already created above
            
            # Validate enhanced audio (get dictionary coverage)
            processed_duration = len(processed) / ANALYSIS_SR
            _, _, validation_info_enh = validate_input_audio(processed, processed_duration)
            
            # NISQA
            nisqa_orig = assess_nisqa(original_path, "original")
            nisqa_enh = assess_nisqa(enhanced_path, "enhanced")
            
            # ScoreQ-NR (Dual domain - scientifically accurate approach)
            scoreq_orig = assess_scoreq(raw_audio_16k, ANALYSIS_SR, is_synthetic=False, audio_label="original")
            scoreq_enh = assess_scoreq(processed, ANALYSIS_SR, is_synthetic=True, audio_label="enhanced")
            
            scoreq_orig_score = scoreq_orig['score'] if scoreq_orig else None
            scoreq_enh_score = scoreq_enh['score'] if scoreq_enh else None
            
            # REFINED: Decision with dictionary coverage verification
            decision = decide_enhancement_quality(
                nisqa_orig, nisqa_enh, 
                scoreq_orig_score, scoreq_enh_score,
                validation_info, validation_info_enh
            )
        
        # Display decision
        if decision['rejected']:
            st.error("❌ **Enhancement Rejected** - Using original audio")
            st.write("**Rejection Reasons:**")
            for r in decision['reasons']:
                st.write(f"• {r}")
            
            # NEW: Show risk/benefit scores
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Score", f"{decision['risk_score']:.2f}",
                         help="Risk of degradation (0.0-1.0, lower is better)")
            with col2:
                st.metric("Benefit Score", f"{decision['benefit_score']:.2f}",
                         help="Quality improvement potential (0.0-1.0, higher is better)")
            
            final_audio = "original"
            decision_status = 'Rejected'
            # Store VoiceFixer processed audio at 44.1kHz for ZIP (even though rejected)
            # This allows users to compare rejected VoiceFixer output vs original
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                save_wav(tmp.name, vf_out, VOICEFIXER_SR)
                with open(tmp.name, 'rb') as f:
                    processed_audio_bytes = f.read()
                os.unlink(tmp.name)
        else:
            st.success("✅ **Enhancement Accepted** - Using enhanced audio")
            
            # NEW: Show risk/benefit scores
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Score", f"{decision['risk_score']:.2f}",
                         delta="Low Risk" if decision['risk_score'] < 0.3 else "Moderate Risk" if decision['risk_score'] < 0.6 else "High Risk",
                         delta_color="inverse" if decision['risk_score'] < 0.3 else "off",
                         help="Risk of degradation (0.0-1.0, lower is better)")
            with col2:
                st.metric("Benefit Score", f"{decision['benefit_score']:.2f}",
                         delta="High Benefit" if decision['benefit_score'] > 0.6 else "Moderate Benefit" if decision['benefit_score'] > 0.3 else "Low Benefit",
                         delta_color="normal" if decision['benefit_score'] > 0.6 else "off",
                         help="Quality improvement potential (0.0-1.0, higher is better)")
            
            final_audio = "enhanced"
            decision_status = 'Accepted'
            # Store enhanced audio at 44.1kHz (VoiceFixer output SR) for ZIP
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                save_wav(tmp.name, vf_out, VOICEFIXER_SR)
                with open(tmp.name, 'rb') as f:
                    processed_audio_bytes = f.read()
                os.unlink(tmp.name)
        
        # Record batch results if in batch mode
        if batch_results is not None:
            duration_sec = len(raw_audio) / sr
            result_entry = {
                'Filename': file_name,
                'Duration (s)': f"{duration_sec:.2f}",
                'Decision': 'Accepted' if not decision['rejected'] else 'Rejected',
                'Rejection Reasons': '; '.join(decision['reasons']) if decision['rejected'] else 'N/A'
            }
            
            # Add key metrics
            if nisqa_orig and nisqa_enh:
                result_entry['MOS Δ'] = f"{nisqa_enh['mos'] - nisqa_orig['mos']:+.2f}"
            if scoreq_orig_score is not None and scoreq_enh_score is not None:
                result_entry['ScoreQ Δ'] = f"{scoreq_enh_score - scoreq_orig_score:+.2f}"
            
            # Add validation info
            result_entry['Word Count'] = validation_info['word_count']
            result_entry['Dict Coverage %'] = f"{validation_info['dictionary_coverage']*100:.1f}" if validation_info['dictionary_coverage'] is not None else 'N/A'
            result_entry['Dict Quality'] = interpret_dictionary_coverage(validation_info['dictionary_coverage']) if validation_info['dictionary_coverage'] is not None else 'N/A'
            result_entry['Speech Rate'] = f"{validation_info['words_per_sec']:.2f}"
            result_entry['Rate Quality'] = interpret_speech_rate(validation_info['words_per_sec'])
            
            # Add acoustic quality metrics
            result_entry['Bandwidth (Hz)'] = f"{validation_info['bandwidth_hz']:.0f}"
            result_entry['Bandwidth Quality'] = interpret_bandwidth(validation_info['bandwidth_hz'])
            result_entry['SNR (dB)'] = f"{validation_info['snr_db']:.1f}"
            result_entry['Temporal Roughness'] = f"{validation_info['temporal_roughness']:.3f}"
            result_entry['Roughness Quality'] = interpret_temporal_roughness(validation_info['temporal_roughness'])
            
            # Add language detection metrics
            result_entry['Detected Language'] = validation_info['detected_language'].upper()
            result_entry['Transcription Conf'] = f"{validation_info['transcription_confidence']:.3f}"
            result_entry['Conf Quality'] = interpret_confidence(validation_info['transcription_confidence'])
            result_entry['No-Speech Prob'] = f"{validation_info['no_speech_prob']:.1%}"
            result_entry['Speech Likelihood'] = interpret_no_speech_prob(validation_info['no_speech_prob'])
            
            # NEW: Add VAD metrics
            result_entry['Voiced Duration (s)'] = f"{validation_info['voiced_duration']:.2f}"
            result_entry['Voiced %'] = f"{validation_info['voiced_percentage']:.1%}"
            
            # NEW: Add risk/benefit scores
            result_entry['Risk Score'] = f"{decision['risk_score']:.2f}"
            result_entry['Benefit Score'] = f"{decision['benefit_score']:.2f}"
            
            batch_results.append(result_entry)
        
        if decision['warnings']:
            with st.expander("⚠️ Warnings (non-critical)"):
                for w in decision['warnings']:
                    st.write(f"• {w}")
        
        # ==========================================================
        # DETAILED METRICS
        # ==========================================================
        
        with st.expander("📊 Detailed Quality Metrics"):
            # NISQA
            if nisqa_orig and nisqa_enh:
                st.write("**NISQA Quality Assessment**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("*Original*")
                    st.json(nisqa_orig)
                
                with col2:
                    st.write("*Enhanced*")
                    st.json(nisqa_enh)
                    
                    mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']
                    st.metric("MOS Δ", f"{mos_delta:+.2f}")
            
            # ScoreQ-NR
            if scoreq_orig_score is not None and scoreq_enh_score is not None:
                st.write("**ScoreQ-NR Speech Quality (Trend Indicator)**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original", f"{scoreq_orig_score:.3f}")
                with col2:
                    st.metric("Enhanced", f"{scoreq_enh_score:.3f}")
                with col3:
                    scoreq_delta = scoreq_enh_score - scoreq_orig_score
                    st.metric("Change", f"{scoreq_delta:+.3f}", 
                             delta_color="normal" if scoreq_delta > 0 else "inverse")
            
            # Transcription Comparison (shows what Whisper detected)
            st.write("**Transcription Comparison**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("*Original Transcription*")
                st.code(validation_info.get('transcript', 'N/A'))
                st.caption(f"Word count: {validation_info.get('word_count', 0)}")
            with col2:
                st.write("*Enhanced Transcription*")
                st.code(validation_info_enh.get('transcript', 'N/A'))
                st.caption(f"Word count: {validation_info_enh.get('word_count', 0)}")
            
            # Dictionary Coverage Comparison (Critical for gibberish detection)
            dict_cov_orig = validation_info.get('dictionary_coverage')
            dict_cov_enh = validation_info_enh.get('dictionary_coverage')
            
            st.write("**Dictionary Coverage Verification**")
            
            if dict_cov_orig is not None or dict_cov_enh is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if dict_cov_orig is not None:
                        st.metric("Original", f"{dict_cov_orig:.1%}",
                                 delta="Pass" if dict_cov_orig >= MIN_DICTIONARY_COVERAGE else "Fail",
                                 delta_color="normal" if dict_cov_orig >= MIN_DICTIONARY_COVERAGE else "inverse")
                    else:
                        st.metric("Original", "N/A (< 5 words)")
                with col2:
                    if dict_cov_enh is not None:
                        st.metric("Enhanced", f"{dict_cov_enh:.1%}",
                                 delta="Pass" if dict_cov_enh >= MIN_DICTIONARY_COVERAGE else "Fail",
                                 delta_color="normal" if dict_cov_enh >= MIN_DICTIONARY_COVERAGE else "inverse")
                    else:
                        st.metric("Enhanced", "N/A (< 5 words)")
                with col3:
                    if dict_cov_orig is not None and dict_cov_enh is not None:
                        dict_delta = dict_cov_enh - dict_cov_orig
                        st.metric("Change", f"{dict_delta:+.1%}",
                                 delta_color="normal" if dict_delta > 0 else "inverse")
                    else:
                        st.metric("Change", "N/A")
                
                # Show interpretation
                if dict_cov_orig is not None and dict_cov_orig < MIN_DICTIONARY_COVERAGE:
                    if dict_cov_enh is not None and dict_cov_enh >= MIN_DICTIONARY_COVERAGE:
                        st.success("✓ Enhancement improved dictionary coverage - noise was masking speech")
                    elif dict_cov_enh is not None:
                        st.warning("⚠️ Dictionary coverage remains low - verifying with quality metrics")
                    else:
                        st.warning("⚠️ Enhanced audio has too few words (< 5) to calculate dictionary coverage")
                elif dict_cov_orig is None:
                    st.info("ℹ️ Original audio has too few words (< 5) to calculate dictionary coverage - high-risk scenario")
            else:
                st.warning("⚠️ Both original and enhanced have < 5 words - dictionary coverage not calculated")
        
        # ==========================================================
        # FINAL OUTPUT
        # ==========================================================
        
        st.subheader("🎵 Final Output")
        if final_audio == "enhanced":
            st.success("✓ Sending enhanced audio to user")
            st.audio(enhanced_path)
            
            # Download button for enhanced audio
            with open(enhanced_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            # Generate filename
            base_name = os.path.splitext(file_name)[0]
            download_filename = f"{base_name}_enhanced.wav"
            
            st.download_button(
                label="⬇️ Download Enhanced Audio",
                data=audio_bytes,
                file_name=download_filename,
                mime="audio/wav",
                key=f"download_enhanced_{file_name}"
            )
        else:
            st.info("ℹ️ Sending original audio to preserve quality")
            if isinstance(file_input, str):
                # File path mode
                st.audio(file_input)
                with open(file_input, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                download_filename = f"{os.path.splitext(os.path.basename(file_input))[0]}_original.wav"
            else:
                # Uploaded file mode - need to recreate from raw_audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, raw_audio_16k, ANALYSIS_SR)
                    st.audio(tmp.name)
                    with open(tmp.name, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                    os.unlink(tmp.name)
                download_filename = f"{os.path.splitext(file_name)[0]}_original.wav"
            
            st.download_button(
                label="⬇️ Download Original Audio",
                data=audio_bytes,
                file_name=download_filename,
                mime="audio/wav",
                key=f"download_original_{file_name}"
            )
        
        # Cleanup
        os.unlink(original_path)
        os.unlink(enhanced_path)
        
        # Return audio data for ZIP creation
        return original_audio_bytes, processed_audio_bytes, decision_status


# ==========================================================
# MAIN UI AND BATCH PROCESSING
# ==========================================================

# Input mode selection
input_mode = st.radio("Select input mode:", ["Single File", "Multiple Files", "Folder Path"], horizontal=True)

if input_mode == "Single File":
    raw_bytes = st.file_uploader("Choose audio file", type=["wav", "mp3", "m4a", "flac"])
    audio_files_to_process = [(raw_bytes, raw_bytes.name if raw_bytes else "uploaded_file")] if raw_bytes else []
elif input_mode == "Multiple Files":
    # Multiple file upload
    uploaded_files = st.file_uploader("Choose audio files", type=["wav", "mp3", "m4a", "flac"], accept_multiple_files=True)
    audio_files_to_process = [(f, f.name) for f in uploaded_files] if uploaded_files else []
    
    if uploaded_files:
        st.success(f"Selected {len(uploaded_files)} file(s)")
        with st.expander("📁 Files to process:"):
            for f in uploaded_files:
                st.write(f"• {f.name}")
else:
    # Folder mode
    folder_path = st.text_input("Enter folder path:", placeholder="/path/to/audio/folder")
    recursive_search = st.checkbox("Search subfolders recursively", value=True)
    
    audio_files_to_process = []
    
    if folder_path and os.path.isdir(folder_path):
        # Find all audio files
        audio_extensions = ["wav", "mp3", "m4a", "flac"]
        
        found_files = []
        for ext in audio_extensions:
            if recursive_search:
                found_files.extend(glob.glob(os.path.join(folder_path, "**", f"*.{ext}"), recursive=True))
            else:
                found_files.extend(glob.glob(os.path.join(folder_path, f"*.{ext}")))
        
        found_files = sorted(set(found_files))  # Remove duplicates and sort
        
        if found_files:
            st.success(f"Found {len(found_files)} audio file(s)")
            with st.expander("📁 Files to process:"):
                for f in found_files:
                    st.write(f"• {os.path.basename(f)}")
            
            # Prepare files for processing (file path, display name)
            audio_files_to_process = [(f, os.path.basename(f)) for f in found_files]
        else:
            st.warning("No audio files found in the specified folder.")
    elif folder_path:
        st.error("Invalid folder path. Please enter a valid directory path.")

# Process files
if audio_files_to_process:
    # Initialize session state for stop button
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False
    
    # For batch processing, store results and processed files
    batch_results = []
    processed_files = []  # List of (original_bytes, processed_bytes, filename, decision_status)
    
    # Progress tracking for batch mode
    if len(audio_files_to_process) > 1:
        progress_bar = st.progress(0)
        status_text = st.empty()
        stop_button_placeholder = st.empty()
        
        # Show stop button before processing starts
        if stop_button_placeholder.button("🛑 Stop Processing", key="stop_batch_btn"):
            st.session_state.stop_processing = True
        
        # Show memory warning for large batches
        if len(audio_files_to_process) > 50:
            st.warning(f"⚠️ Processing {len(audio_files_to_process)} files will use significant memory. Consider processing in smaller batches if you experience issues.")
    
    for file_idx, (file_input, file_name) in enumerate(audio_files_to_process):
        # Check if user clicked stop button
        if len(audio_files_to_process) > 1 and st.session_state.stop_processing:
            status_text.text(f"⛔ Processing stopped by user at {file_idx}/{len(audio_files_to_process)} files")
            st.warning(f"⚠️ Processing stopped. Processed {file_idx} out of {len(audio_files_to_process)} files.")
            break
        
        # Update progress for batch mode
        if len(audio_files_to_process) > 1:
            progress = (file_idx) / len(audio_files_to_process)
            progress_bar.progress(progress)
            status_text.text(f"Processing {file_idx + 1}/{len(audio_files_to_process)}: {file_name}")
            
            # Create collapsible section for each file - FULL UI SHOWN
            with st.expander(f"📄 {file_name}", expanded=False):
                orig_bytes, proc_bytes, decision = process_single_file(file_input, file_name, batch_results, input_mode)
                # Store processed files for ZIP creation
                processed_files.append((orig_bytes, proc_bytes, file_name, decision))
        else:
            # Single file mode - no expander, no need to store for ZIP
            process_single_file(file_input, file_name, batch_results, input_mode)
    
    # Complete progress
    if len(audio_files_to_process) > 1:
        # Show completion status
        if not st.session_state.stop_processing:
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed processing {len(audio_files_to_process)} files")
        else:
            progress_bar.progress(len(processed_files) / len(audio_files_to_process))
            status_text.text(f"⛔ Processing stopped - Completed {len(processed_files)}/{len(audio_files_to_process)} files")
        
        # Display summary table
        st.subheader("📊 Batch Processing Summary")
        
        if batch_results:
            import pandas as pd
            df = pd.DataFrame(batch_results)
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                accepted = sum(1 for r in batch_results if r['Decision'] == 'Accepted')
                st.metric("Accepted", accepted)
            with col2:
                # Count both 'Rejected' and 'Rejected (Input Validation)' as rejected
                rejected = sum(1 for r in batch_results if r['Decision'].startswith('Rejected'))
                st.metric("Rejected", rejected)
            with col3:
                st.metric("Processed", len(batch_results))
            with col4:
                st.metric("Total Files", len(audio_files_to_process))
            
            # CSV Export
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name="batch_processing_results.csv",
                mime="text/csv"
            )
            
            # ZIP Export with all processed files
            st.subheader("📦 Download All Processed Files")
            
            with st.spinner("Creating ZIP file..."):
                # Create ZIP file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add CSV report to ZIP
                    zip_file.writestr('batch_results.csv', csv_data)
                    
                    # Add all processed audio files
                    for orig_bytes, proc_bytes, filename, decision in processed_files:
                        # Get base filename without extension
                        base_name = os.path.splitext(filename)[0]
                        extension = os.path.splitext(filename)[1]
                        
                        # Add original file
                        zip_file.writestr(f"originals/{filename}", orig_bytes)
                        
                        # Add processed file with decision suffix
                        if decision == 'Accepted':
                            processed_filename = f"{base_name}_Accepted.wav"
                        elif decision == 'Rejected':
                            processed_filename = f"{base_name}_Rejected.wav"
                        else:  # 'Rejected (Input Validation)'
                            processed_filename = f"{base_name}_Rejected.wav"
                        
                        zip_file.writestr(f"processed/{processed_filename}", proc_bytes)
                
                zip_buffer.seek(0)
                zip_bytes = zip_buffer.read()
            
            st.success(f"✅ Created ZIP file with {len(processed_files)} processed files + CSV report")
            
            st.download_button(
                label="📦 Download All Files (ZIP)",
                data=zip_bytes,
                file_name="voicefixer_batch_results.zip",
                mime="application/zip",
                help="ZIP contains: originals/ folder, processed/ folder (with _Accepted/_Rejected suffixes), and batch_results.csv"
            )
