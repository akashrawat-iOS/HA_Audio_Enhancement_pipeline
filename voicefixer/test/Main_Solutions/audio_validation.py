"""
Audio Validation Module
======================

Input validation logic for VoiceFixer pre-screening.

This module provides functions to validate audio quality BEFORE VoiceFixer processing,
including:
- Word density (speech rate) validation
- Dictionary coverage (lexical validity) checks
- Acoustic fallback for Whisper failures
- Language detection (English-only mode)
- Context-aware thresholds for short clips

Author: Akash Rawat
Date: February 2026
"""

import numpy as np
import librosa
import whisper
import tempfile
import soundfile as sf
import enchant
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# ==========================================================
# CONFIGURATION
# ==========================================================

ANALYSIS_SR = 16000

# Input Validation Thresholds
MIN_WORD_DENSITY = 0.8
MIN_WORD_DENSITY_SHORT = 0.5
MIN_DICTIONARY_COVERAGE = 0.6
MIN_ABSOLUTE_WORDS = 3
MIN_TRANSCRIPT_LENGTH = 5
SHORT_CLIP_THRESHOLD = 3.0

# Domain allowlist
DOMAIN_ALLOWLIST = {
    'ptt', 'dsp', 'ai', 'ml', 'api', 'url', 'http', 'https', 'ui', 'ux',
    'ok', 'okay', 'um', 'uh', 'hmm', 'yeah', 'yep', 'nope', 'gonna', 'wanna'
}

# Acoustic fallback
RMS_ENERGY_MIN = 0.01
VOICED_FRAME_RATIO_MIN = 0.15

# Language detection
ENABLE_LANGUAGE_DETECTION = True
REQUIRED_LANGUAGE = "en"
MIN_TRANSCRIPTION_CONFIDENCE = -0.8
MAX_NO_SPEECH_PROB = 0.6

# Device
DEVICE = "cuda" if np.cuda.is_available() else "cpu"

# ==========================================================
# MODEL LOADING
# ==========================================================

def load_whisper_model(model_name: str = "base"):
    """Load Whisper model for transcription."""
    return whisper.load_model(model_name, device=DEVICE)

def load_dictionary(lang: str = "en_US"):
    """Load dictionary for word validation."""
    try:
        return enchant.Dict(lang)
    except enchant.errors.DictNotFoundError:
        logger.warning(f"⚠️ {lang} dictionary not found, trying en_GB")
        try:
            return enchant.Dict("en_GB")
        except:
            logger.error("❌ No English dictionary available")
            return None

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
# ACOUSTIC CHECKS
# ==========================================================

def check_acoustic_speech_presence(audio: np.ndarray, sr: int) -> Tuple[bool, Dict]:
    """
    Lightweight acoustic check for speech presence (fallback for Whisper failures).
    
    Args:
        audio: Audio signal
        sr: Sample rate
    
    Returns:
        Tuple of (has_speech_signal, acoustic_info)
    """
    acoustic_info = {
        'rms_energy': 0.0,
        'voiced_frame_ratio': 0.0,
        'has_signal': False
    }
    
    rms = np.sqrt(np.mean(audio ** 2))
    acoustic_info['rms_energy'] = float(rms)
    
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    frame_energies = np.sqrt(np.mean(frames ** 2, axis=0))
    
    if len(frame_energies) > 0:
        noise_floor = np.median(frame_energies)
        voiced_frames = np.sum(frame_energies > noise_floor * 2)
        voiced_ratio = voiced_frames / len(frame_energies)
        acoustic_info['voiced_frame_ratio'] = float(voiced_ratio)
    
    has_signal = (rms >= RMS_ENERGY_MIN) or (acoustic_info['voiced_frame_ratio'] >= VOICED_FRAME_RATIO_MIN)
    acoustic_info['has_signal'] = has_signal
    
    return has_signal, acoustic_info

def analyze_audio_quality(audio: np.ndarray, sr: int) -> Tuple[float, float, float]:
    """
    Calculate acoustic quality metrics (informational only).
    
    Returns:
        Tuple of (bandwidth_hz, snr_db, temporal_roughness)
    """
    try:
        if len(audio) < 2048:
            return 0.0, 0.0, 0.0

        audio = audio.astype(np.float32)
        audio -= np.mean(audio)

        frame_length = int(0.02 * sr)
        hop_length = int(0.01 * sr)

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.mean(frames ** 2, axis=0)

        # Bandwidth
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

        bandwidth_hz = float(np.percentile(frame_bandwidths, 90)) if frame_bandwidths else 0.0

        # SNR
        noise_threshold = np.percentile(frame_energy, 10)
        noise_frames = frame_energy[frame_energy <= noise_threshold]
        noise_power = np.mean(noise_frames)
        signal_power = np.mean(frame_energy)

        if noise_power > 1e-10:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 60.0
        snr_db = float(np.clip(snr_db, -5.0, 45.0))

        # Temporal roughness
        if len(frame_energy) > 2:
            log_energy = np.log(frame_energy + 1e-10)
            delta = np.diff(log_energy)
            delta2 = np.diff(delta)
            roughness = np.std(delta2)
            temporal_roughness = float(np.clip(roughness / 0.5, 0.0, 1.0))
        else:
            temporal_roughness = 0.0

        return bandwidth_hz, snr_db, temporal_roughness

    except Exception as e:
        logger.warning(f"Audio quality analysis error: {e}")
        return 0.0, 0.0, 0.0

# ==========================================================
# TRANSCRIPTION
# ==========================================================

def transcribe_audio(audio: np.ndarray, whisper_model, sr: int = ANALYSIS_SR) -> Tuple[List[Dict], str, str, float, float]:
    """
    Transcribe audio using Whisper with language detection.
    
    Args:
        audio: Audio signal
        whisper_model: Loaded Whisper model
        sr: Sample rate
    
    Returns:
        Tuple of (word_list, full_transcript, detected_language, confidence, no_speech_prob)
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        save_wav(tmp.name, audio, sr)
        try:
            result = whisper_model.transcribe(
                tmp.name,
                word_timestamps=True,
                fp16=False,
                temperature=0.0,
                beam_size=5,
                best_of=5
            )
        finally:
            import os
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
    
    detected_language = result.get('language', 'unknown')
    
    segments = result.get('segments', [])
    if segments:
        avg_logprobs = [seg.get('avg_logprob', -999) for seg in segments if 'avg_logprob' in seg]
        confidence = np.mean(avg_logprobs) if avg_logprobs else -999
        no_speech_prob = segments[0].get('no_speech_prob', 0.0)
    else:
        confidence = -999
        no_speech_prob = 1.0
    
    return words, result.get('text', ''), detected_language, float(confidence), float(no_speech_prob)

# ==========================================================
# INPUT VALIDATION
# ==========================================================

def validate_input_audio(
    audio: np.ndarray, 
    duration: float,
    whisper_model,
    dictionary = None
) -> Tuple[bool, Optional[str], Dict]:
    """
    Pre-screen input audio quality before VoiceFixer processing.
    
    Args:
        audio: Audio signal at ANALYSIS_SR
        duration: Audio duration in seconds
        whisper_model: Loaded Whisper model
        dictionary: Loaded enchant dictionary (optional)
    
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
        'bandwidth_hz': 0.0,
        'snr_db': 0.0,
        'temporal_roughness': 0.0,
        'detected_language': 'unknown',
        'transcription_confidence': 0.0,
        'no_speech_prob': 0.0
    }
    
    is_short_clip = duration < SHORT_CLIP_THRESHOLD
    
    # Calculate acoustic quality metrics
    validation_info['bandwidth_hz'], validation_info['snr_db'], validation_info['temporal_roughness'] = \
        analyze_audio_quality(audio, ANALYSIS_SR)
    
    # Get transcription
    words, transcript, detected_lang, confidence, no_speech_prob = \
        transcribe_audio(audio, whisper_model, ANALYSIS_SR)
    word_count = len(words)
    
    validation_info['detected_language'] = detected_lang
    validation_info['transcription_confidence'] = confidence
    validation_info['no_speech_prob'] = no_speech_prob
    validation_info['word_count'] = word_count
    validation_info['transcript'] = transcript
    
    # Acoustic fallback for low word count
    if word_count < MIN_ABSOLUTE_WORDS:
        has_signal, acoustic_info = check_acoustic_speech_presence(audio, ANALYSIS_SR)
        validation_info['acoustic_fallback_used'] = True
        validation_info['acoustic_info'] = acoustic_info
        
        if not has_signal and duration > 2.0:
            return False, f"No speech detected: {word_count} words, no acoustic signal (RMS={acoustic_info['rms_energy']:.4f})", validation_info
    
    validation_failures = []
    
    # Check 1: Minimum absolute word count
    if word_count < MIN_ABSOLUTE_WORDS and duration > 2.0:
        validation_failures.append(f"Low word count ({word_count} in {duration:.1f}s)")
    
    # Check 2: Word density
    words_per_sec = word_count / duration if duration > 0 else 0
    validation_info['words_per_sec'] = words_per_sec
    
    effective_threshold = MIN_WORD_DENSITY_SHORT if is_short_clip else MIN_WORD_DENSITY
    
    if words_per_sec < effective_threshold:
        validation_failures.append(f"Low word density ({words_per_sec:.2f} wps, threshold={effective_threshold:.2f} wps)")
    
    # Check 3: Dictionary coverage
    if dictionary and word_count >= MIN_TRANSCRIPT_LENGTH:
        word_list = transcript.lower().split()
        valid_words = []
        invalid_words = []
        
        for word in word_list:
            clean_word = ''.join(c for c in word if c.isalpha())
            if not clean_word or len(clean_word) < 3:
                continue
            
            if clean_word in DOMAIN_ALLOWLIST:
                valid_words.append(clean_word)
                continue
            
            if dictionary.check(clean_word):
                valid_words.append(clean_word)
            else:
                invalid_words.append(clean_word)
        
        total_checked = len(valid_words) + len(invalid_words)
        dictionary_coverage = len(valid_words) / total_checked if total_checked > 0 else 0
        
        validation_info['dictionary_coverage'] = dictionary_coverage
        validation_info['valid_words'] = valid_words
        validation_info['invalid_words'] = invalid_words
        
        if dictionary_coverage < MIN_DICTIONARY_COVERAGE:
            validation_failures.append(f"Low dictionary coverage ({dictionary_coverage:.1%})")
    
    # Require multiple validation failures
    if len(validation_failures) >= 2:
        rejection_reason = "Multiple validation failures: " + "; ".join(validation_failures)
        return False, rejection_reason, validation_info
    elif len(validation_failures) == 1 and word_count < MIN_ABSOLUTE_WORDS and duration > 3.0:
        rejection_reason = validation_failures[0] + " (high confidence rejection)"
        return False, rejection_reason, validation_info
    
    return True, None, validation_info
