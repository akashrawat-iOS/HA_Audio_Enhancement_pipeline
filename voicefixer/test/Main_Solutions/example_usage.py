"""
Example: Using Modularized VoiceFixer Validation
================================================

This example demonstrates how to use the modularized components
for audio validation and quality assessment.

Author: Akash Rawat
Date: February 2026
"""

import librosa
import numpy as np
from audio_validation import (
    load_whisper_model,
    load_dictionary,
    validate_input_audio
)
from quality_assessment import (
    load_nisqa_model,
    load_scoreq_models,
    assess_nisqa,
    assess_scoreq,
    decide_enhancement_quality
)
from voicefixer import VoiceFixer

def process_audio_file(input_path: str, output_path: str):
    """
    Complete audio enhancement pipeline with validation.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save enhanced audio
    """
    
    # Step 1: Load models
    print("Loading models...")
    whisper_model = load_whisper_model("base")
    dictionary = load_dictionary("en_US")
    voicefixer = VoiceFixer()
    
    # Try to load NISQA (optional)
    try:
        nisqa_model = load_nisqa_model("nisqa.tar")
    except:
        print("⚠️ NISQA not available, continuing without it")
        nisqa_model = None
    
    # Load ScoreQ models
    scoreq_natural, scoreq_synthetic = load_scoreq_models()
    
    # Step 2: Load and validate input
    print(f"\nProcessing: {input_path}")
    audio, sr = librosa.load(input_path, sr=16000, mono=True)
    duration = len(audio) / sr
    
    print(f"Duration: {duration:.2f}s")
    
    # Step 3: Validate input audio
    print("\n🔍 Stage 1: Input Validation")
    is_valid, rejection_reason, validation_info = validate_input_audio(
        audio, duration, whisper_model, dictionary
    )
    
    print(f"  Word count: {validation_info['word_count']}")
    print(f"  Speech rate: {validation_info['words_per_sec']:.2f} wps")
    print(f"  Dictionary coverage: {validation_info['dictionary_coverage']:.1%}" 
          if validation_info['dictionary_coverage'] else "  Dictionary coverage: N/A")
    print(f"  Language: {validation_info['detected_language'].upper()}")
    
    if not is_valid:
        print(f"\n❌ Input validation failed: {rejection_reason}")
        print("Saving original audio without enhancement")
        import soundfile as sf
        sf.write(output_path, audio, sr)
        return
    
    print("✅ Input validation passed")
    
    # Step 4: Enhance with VoiceFixer
    print("\n⚙️ Stage 2: VoiceFixer Enhancement")
    
    # Resample for VoiceFixer
    audio_44k = librosa.resample(audio, orig_sr=16000, target_sr=44100)
    
    # Enhance
    enhanced = voicefixer.restore_inmem(audio_44k, mode=0, cuda=False)
    
    # Resample back to 16kHz for assessment
    enhanced_16k = librosa.resample(enhanced, orig_sr=44100, target_sr=16000)
    
    print("Enhancement complete")
    
    # Step 5: Quality assessment
    print("\n🧪 Stage 3: Quality Assessment")
    
    # Save temp files for NISQA
    import tempfile
    import soundfile as sf
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_orig:
        sf.write(tmp_orig.name, audio, 16000)
        orig_path = tmp_orig.name
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_enh:
        sf.write(tmp_enh.name, enhanced_16k, 16000)
        enh_path = tmp_enh.name
    
    # NISQA assessment
    nisqa_orig = assess_nisqa(orig_path, nisqa_model, "original") if nisqa_model else None
    nisqa_enh = assess_nisqa(enh_path, nisqa_model, "enhanced") if nisqa_model else None
    
    if nisqa_orig and nisqa_enh:
        print(f"  NISQA MOS: {nisqa_orig['mos']:.2f} → {nisqa_enh['mos']:.2f} (Δ{nisqa_enh['mos']-nisqa_orig['mos']:+.2f})")
    
    # ScoreQ-NR assessment
    scoreq_orig = assess_scoreq(audio, 16000, scoreq_natural, is_synthetic=False, audio_label="original")
    scoreq_enh = assess_scoreq(enhanced_16k, 16000, scoreq_synthetic, is_synthetic=True, audio_label="enhanced")
    
    scoreq_orig_score = scoreq_orig['score'] if scoreq_orig else None
    scoreq_enh_score = scoreq_enh['score'] if scoreq_enh else None
    
    if scoreq_orig_score and scoreq_enh_score:
        print(f"  ScoreQ-NR: {scoreq_orig_score:.2f} → {scoreq_enh_score:.2f} (Δ{scoreq_enh_score-scoreq_orig_score:+.2f})")
    
    # Validate enhanced audio
    _, _, validation_info_enh = validate_input_audio(
        enhanced_16k, len(enhanced_16k)/16000, whisper_model, dictionary
    )
    
    # Decision
    decision = decide_enhancement_quality(
        nisqa_orig, nisqa_enh,
        scoreq_orig_score, scoreq_enh_score,
        validation_info, validation_info_enh
    )
    
    # Cleanup temp files
    import os
    os.unlink(orig_path)
    os.unlink(enh_path)
    
    # Step 6: Save output
    print("\n🎵 Final Decision")
    if decision['rejected']:
        print("❌ Enhancement rejected - using original")
        print("Reasons:")
        for reason in decision['reasons']:
            print(f"  • {reason}")
        sf.write(output_path, audio, 16000)
    else:
        print("✅ Enhancement accepted - using enhanced")
        if decision['warnings']:
            print("Warnings:")
            for warning in decision['warnings']:
                print(f"  • {warning}")
        sf.write(output_path, enhanced_16k, 16000)
    
    print(f"\nOutput saved: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python example_usage.py <input.wav> <output.wav>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_audio_file(input_file, output_file)
