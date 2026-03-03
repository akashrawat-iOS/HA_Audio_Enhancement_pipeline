# Changelog

All notable changes to the VoiceFixer Input Validation Gate project.

## [2.0.0-REFINED] - 2026-02-09

### Added
- **Language Detection (English-Only Mode)**
  - Auto-detects audio language and rejects non-English inputs
  - Prevents Whisper hallucinations (e.g., German audio transcribed as English gibberish)
  - Configurable via `ENABLE_LANGUAGE_DETECTION` flag
  - Displays detected language, confidence, and no-speech probability

- **Multi-Tier Validation System**
  - Requires MULTIPLE failures before rejection (reduces false positives)
  - Single minor issue no longer causes hard rejection
  - Context-aware: different thresholds for short clips (<3s)

- **Domain-Aware Dictionary Validation**
  - Allowlist for common acronyms and interjections
  - Ignores very short words (<3 characters)
  - Customizable via `DOMAIN_ALLOWLIST` configuration

- **Acoustic Fallback Mechanism**
  - RMS energy and voiced frame ratio checks
  - Prevents rejection of valid speech that Whisper fails to transcribe
  - Used only when word count is very low

- **Three-Tier ScoreQ Decision Logic**
  - Significant degradation (< -0.1): Always reject
  - Stable/minimal change (-0.1 to 0.1): Check NISQA as tiebreaker
  - Clear improvement (>= 0.1): Always accept
  - Previously: Rejected anything with delta <= 0.05 (too strict)

- **Comprehensive Interpretation Functions**
  - Human-readable interpretations for all metrics
  - `interpret_confidence()` - Whisper confidence levels
  - `interpret_no_speech_prob()` - Speech likelihood
  - `interpret_speech_rate()` - Words per second categories
  - `interpret_dictionary_coverage()` - Lexical quality
  - `interpret_bandwidth()` - Frequency range quality
  - `interpret_temporal_roughness()` - Envelope stability

- **Enhanced CSV Export**
  - All metrics with human-readable interpretations
  - Bandwidth, SNR, temporal roughness columns
  - Language detection results
  - Confidence and no-speech probability metrics

- **Dictionary Coverage Re-Check**
  - Post-enhancement validation of transcription quality
  - Detects VoiceFixer corruption (valid → gibberish regression)
  - Three cases handled: low coverage, insufficient words, regression

- **Batch Processing Download Buttons**
  - Stage 2 VoiceFixer output downloadable before quality gate
  - Allows users to compare original, Stage 2, and final output

### Fixed
- **NumPy Compatibility (CRITICAL)**
  - Fixed "Numpy is not available" RuntimeError
  - Downgraded from NumPy 2.0.2 to 1.26.4
  - PyTorch and Whisper incompatible with NumPy 2.x
  - Updated requirements.txt to enforce `numpy<2.0`

- **Whisper Confidence Extraction**
  - Previously showed -999 (top-level result)
  - Now correctly extracts segment-level `avg_logprob`
  - Uses `np.mean()` across all segments for accurate confidence

- **ScoreQ Threshold Documentation**
  - Updated comment to reflect actual implementation (0.1, not 0.05)
  - Three-tier logic uses -0.1 to 0.1 range for stable zone

### Changed
- **Relaxed Input Validation**
  - `MIN_WORD_DENSITY_SHORT = 0.5` (was implicit 0.8)
  - `SHORT_CLIP_THRESHOLD = 3.0s` for context-aware thresholds
  - Multiple failures required (was single failure)

- **Improved UI Feedback**
  - All metrics show delta interpretations
  - Tooltips explain interpretation ranges
  - Color-coded status indicators (✅/⚠️/❌)

- **ScoreQ as Trend Indicator**
  - No longer sole rejection criterion
  - Used complementarily with NISQA
  - Allows stable quality (not just improvements)

### Deprecated
- `SCOREQ_MIN_DELTA` constant no longer used in logic
  - Kept for backward compatibility
  - Replaced by three-tier thresholds

### Modularization
- **audio_validation.py** - Input validation logic (standalone)
- **quality_assessment.py** - NISQA/ScoreQ assessments (standalone)
- **example_usage.py** - Command-line usage example
- Streamlit UI remains in main file for now

### Documentation
- **README.md** - Comprehensive installation, usage, troubleshooting
- **requirements.txt** - Exact dependencies with version constraints
- **CHANGELOG.md** - This file (version history)
- Inline comments improved throughout codebase

### Known Issues
- ScoreQ dependency warning (`numpy>=2.0`) can be ignored
  - ScoreQ works correctly with NumPy 1.26.4
  - Waiting for official ScoreQ update to relax constraint

## [1.0.0-BASIC] - 2026-01-15

### Added
- Initial pre-screening approach
- NISQA integration for overall quality
- ScoreQ-NR integration for speech quality
- Dictionary-based gibberish detection
- Basic word density validation
- Single file processing via Streamlit
- Batch folder processing
- CSV export of results

### Fixed
- N/A (initial release)

### Known Limitations
- Forced English transcription (`language="en"`) causes hallucinations
- Single failure causes hard rejection (too strict)
- No context awareness for short clips
- No acoustic fallback for Whisper failures
- ScoreQ rejection too strict (requires improvement, not stability)
- No human-readable interpretations for metrics
