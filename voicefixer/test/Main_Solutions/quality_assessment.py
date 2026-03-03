"""
Quality Assessment Module
========================

NISQA and ScoreQ-NR quality assessment for VoiceFixer validation.

This module provides functions to assess audio quality using:
- NISQA: Overall perceptual quality (MOS, noise, distortion, coloration)
- ScoreQ-NR: Speech-specific quality (dual domain: natural/synthetic)

Author: Akash Rawat
Date: February 2026
"""

import numpy as np
import librosa
import tempfile
import soundfile as sf
from typing import Optional, Dict
import logging
import os

logger = logging.getLogger(__name__)

# ==========================================================
# CONFIGURATION
# ==========================================================

ANALYSIS_SR = 16000

# NISQA Thresholds
NISQA_MOS_DELTA_MIN = -0.1
NISQA_COL_MIN = 3.0
NISQA_DIS_MIN = 3.0
NISQA_MOS_MIN = 2.5

# ScoreQ Thresholds
SCOREQ_DEGRADATION_THRESHOLD = -0.1
SCOREQ_IMPROVEMENT_THRESHOLD = 0.1

# ==========================================================
# MODEL LOADING
# ==========================================================

def load_nisqa_model(weights_path: str):
    """Load NISQA model with weights."""
    from nisqa.NISQA_model import nisqaModel
    
    return {
        'loaded': True,
        'weights_path': weights_path
    }

def load_scoreq_models():
    """Load ScoreQ-NR models (dual domain)."""
    from scoreq import Scoreq
    
    scoreq_natural = Scoreq(mode="nr", data_domain="natural")
    scoreq_synthetic = Scoreq(mode="nr", data_domain="synthetic")
    
    return scoreq_natural, scoreq_synthetic

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
# NISQA ASSESSMENT
# ==========================================================

def assess_nisqa(
    audio_path: str,
    nisqa_model: Dict,
    audio_label: str = "audio"
) -> Optional[Dict]:
    """
    Assess audio quality using NISQA.
    
    Args:
        audio_path: Path to audio file
        nisqa_model: Loaded NISQA model dict
        audio_label: Label for logging
    
    Returns:
        Dict with NISQA metrics or None if unavailable
    """
    if not nisqa_model or not nisqa_model.get('loaded'):
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
        return None

# ==========================================================
# SCOREQ-NR ASSESSMENT
# ==========================================================

def assess_scoreq(
    audio: np.ndarray,
    sr: int,
    scoreq_model,
    is_synthetic: bool = False,
    audio_label: str = "audio"
) -> Optional[Dict]:
    """
    Assess audio quality using ScoreQ-NR.
    
    Args:
        audio: Audio signal (numpy array)
        sr: Sample rate
        scoreq_model: Loaded ScoreQ model (natural or synthetic)
        is_synthetic: True for VoiceFixer output, False for natural audio
        audio_label: Label for logging
    
    Returns:
        Dict with ScoreQ-NR score or None if error
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
            score = scoreq_model.predict(tmp_path)
            
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
        return None

# ==========================================================
# POST-PROCESSING DECISION LOGIC
# ==========================================================

def decide_enhancement_quality(
    nisqa_orig: Optional[Dict],
    nisqa_enh: Optional[Dict],
    scoreq_orig: float,
    scoreq_enh: float,
    validation_info_orig: Dict,
    validation_info_enh: Dict
) -> Dict:
    """
    Quality gate with dictionary coverage verification.
    
    Args:
        nisqa_orig: NISQA metrics for original
        nisqa_enh: NISQA metrics for enhanced
        scoreq_orig: ScoreQ score for original
        scoreq_enh: ScoreQ score for enhanced
        validation_info_orig: Input validation info
        validation_info_enh: Enhanced validation info
    
    Returns:
        Dict with {rejected: bool, reasons: List[str], warnings: List[str]}
    """
    reasons = []
    warnings = []
    
    # Dictionary coverage verification
    dict_cov_orig = validation_info_orig.get('dictionary_coverage')
    dict_cov_enh = validation_info_enh.get('dictionary_coverage')
    word_count_orig = validation_info_orig.get('word_count', 0)
    word_count_enh = validation_info_enh.get('word_count', 0)
    
    # Case 1: Input had low coverage
    if dict_cov_orig is not None and dict_cov_orig < 0.6:
        warnings.append(f"Input had low dictionary coverage ({dict_cov_orig:.1%})")
        
        if dict_cov_enh is not None and dict_cov_enh < 0.6:
            nisqa_improved = False
            scoreq_improved = False
            
            if nisqa_orig and nisqa_enh:
                mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']
                if mos_delta >= 0.2:
                    nisqa_improved = True
            
            if scoreq_orig is not None and scoreq_enh is not None:
                scoreq_delta = scoreq_enh - scoreq_orig
                if scoreq_delta >= 0.3:
                    scoreq_improved = True
            
            if not (nisqa_improved and scoreq_improved):
                reasons.append(
                    f"Dictionary coverage remains low ({dict_cov_orig:.1%}→{dict_cov_enh:.1%}) "
                    f"without both quality metrics improving"
                )
    
    # NISQA checks
    if nisqa_orig and nisqa_enh:
        mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']
        
        if mos_delta < NISQA_MOS_DELTA_MIN:
            reasons.append(f"NISQA: MOS degraded ({nisqa_orig['mos']:.2f}→{nisqa_enh['mos']:.2f})")
        
        if nisqa_enh['col'] < NISQA_COL_MIN:
            reasons.append(f"NISQA: Robotic/unnatural sound (COL={nisqa_enh['col']:.2f})")
        
        if nisqa_enh['dis'] < NISQA_DIS_MIN:
            reasons.append(f"NISQA: High distortion (DIS={nisqa_enh['dis']:.2f})")
        
        if nisqa_enh['mos'] < NISQA_MOS_MIN:
            reasons.append(f"NISQA: Output quality too poor (MOS={nisqa_enh['mos']:.2f})")
    
    # ScoreQ-NR checks (three-tier approach)
    if scoreq_orig is not None and scoreq_enh is not None:
        scoreq_delta = scoreq_enh - scoreq_orig
        
        if scoreq_delta < SCOREQ_DEGRADATION_THRESHOLD:
            reasons.append(f"ScoreQ-NR: Significant degradation (Δ={scoreq_delta:.2f})")
        
        elif SCOREQ_DEGRADATION_THRESHOLD <= scoreq_delta < SCOREQ_IMPROVEMENT_THRESHOLD:
            nisqa_stable = False
            if nisqa_orig and nisqa_enh:
                mos_delta = nisqa_enh['mos'] - nisqa_orig['mos']
                nisqa_stable = mos_delta >= NISQA_MOS_DELTA_MIN
            
            if not nisqa_stable:
                reasons.append(f"ScoreQ-NR: Minimal improvement with NISQA degradation (Δ={scoreq_delta:.2f})")
        
        if scoreq_orig < 2.0:
            warnings.append(f"Input appears to be codec/telephony audio (ScoreQ={scoreq_orig:.2f})")
    
    return {
        'rejected': len(reasons) > 0,
        'reasons': reasons,
        'warnings': warnings
    }
