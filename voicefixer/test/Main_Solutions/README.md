# VoiceFixer Input Validation Gate - REFINED

Production-ready audio enhancement with intelligent pre-screening and quality validation.

## 🎯 Features

- ✅ **Language Detection** - English-only mode prevents hallucinations (e.g., German audio transcribed as gibberish)
- ✅ **Dictionary-Based Gibberish Detection** - Validates lexical validity with domain-aware allowlist
- ✅ **Multi-Tier Validation** - Reduces false rejections (requires multiple failures, not single hard rejection)
- ✅ **Comprehensive Quality Metrics** - NISQA (MOS, distortion, coloration) + ScoreQ-NR (speech-specific)
- ✅ **Batch Processing** - Process folders with CSV export and detailed metrics
- ✅ **Acoustic Fallback** - Prevents rejection of valid but hard-to-transcribe speech
- ✅ **Context-Aware Thresholds** - Relaxed validation for short clips (<3s)

## 📋 Prerequisites

- Python 3.8+
- **NumPy < 2.0** (CRITICAL: NumPy 2.x breaks PyTorch/Whisper compatibility)
- CUDA-capable GPU (optional, recommended for faster processing)

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NISQA Weights

Download `nisqa.tar` from the [NISQA repository](https://github.com/gabrielmittag/NISQA) and place it in one of these locations:
- Same directory as the script
- Parent directory (`../nisqa.tar`)

### 3. Install Dictionary (if not already installed)

```bash
# macOS
brew install enchant

# Ubuntu/Debian
sudo apt-get install python3-enchant

# Windows
# PyEnchant is automatically installed via pip
```

## 💻 Usage

### Single File Processing

```bash
streamlit run voicefixer_input_validation_gate_refined.py
```

Then:
1. Select **"Single File"** mode
2. Upload your audio file (WAV, MP3, M4A, FLAC)
3. View validation results and quality metrics
4. Download enhanced audio if accepted

### Batch Processing

```bash
streamlit run voicefixer_input_validation_gate_refined.py
```

Then:
1. Select **"Folder Path"** mode
2. Enter directory path (e.g., `/path/to/audio/files`)
3. Enable **"Search subfolders recursively"** if needed
4. View batch summary table with all metrics
5. Download CSV export for analysis

### Command-Line Usage (Modularized Version)

```python
from audio_validation import validate_input_audio
from quality_assessment import assess_nisqa, assess_scoreq
import librosa

# Load audio
audio, sr = librosa.load("input.wav", sr=16000, mono=True)
duration = len(audio) / sr

# Validate input
is_valid, reason, info = validate_input_audio(audio, duration)

if is_valid:
    print(f"✅ Valid input: {info['word_count']} words, {info['dictionary_coverage']:.1%} coverage")
else:
    print(f"❌ Rejected: {reason}")
```

## ⚙️ Configuration

Edit thresholds in the configuration section (lines 72-142):

### Input Validation Thresholds

```python
MIN_WORD_DENSITY = 0.8           # Minimum words/sec (normal speech: 2-3 wps)
MIN_WORD_DENSITY_SHORT = 0.5     # Relaxed for short clips (<3s)
MIN_DICTIONARY_COVERAGE = 0.6    # Valid English words ratio (60%)
SHORT_CLIP_THRESHOLD = 3.0       # Duration for relaxed validation (seconds)
```

### Quality Assessment Thresholds

```python
NISQA_MOS_MIN = 2.5              # Minimum output quality
NISQA_COL_MIN = 3.0              # Min coloration (detects robotic sound)
NISQA_DIS_MIN = 3.0              # Min distortion score
NISQA_MOS_DELTA_MIN = -0.1       # Max acceptable MOS degradation
```

### Language Detection

```python
ENABLE_LANGUAGE_DETECTION = True  # Auto-detect and reject non-English
REQUIRED_LANGUAGE = "en"          # ISO 639-1 language code
MIN_TRANSCRIPTION_CONFIDENCE = -0.8  # Whisper confidence threshold
MAX_NO_SPEECH_PROB = 0.6          # Maximum no-speech probability
```

### Domain-Specific Allowlist

Add custom acronyms/terms that shouldn't count against dictionary coverage:

```python
DOMAIN_ALLOWLIST = {
    'ptt', 'dsp', 'ai', 'ml', 'api', 'url', 'http', 'https', 'ui', 'ux',
    'ok', 'okay', 'um', 'uh', 'hmm', 'yeah', 'yep', 'nope', 'gonna', 'wanna',
    # Add your custom terms here
    'yourterm1', 'yourterm2'
}
```

## 📊 Output Metrics

### Input Validation Stage
- **Word Count** - Total words detected by Whisper
- **Speech Rate** - Words per second (Normal: 2-3 wps)
- **Dictionary Coverage** - Percentage of valid English words
- **Detected Language** - ISO 639-1 language code
- **Transcription Confidence** - Whisper avg_logprob (-1.0 to 0.0)
- **No-Speech Probability** - Likelihood audio contains no speech

### Acoustic Quality Metrics (Informational)
- **Bandwidth** - Effective frequency range (Hz)
- **SNR** - Signal-to-Noise Ratio (dB)
- **Temporal Roughness** - Envelope instability (0-1)

### Output Quality Assessment
- **NISQA MOS** - Mean Opinion Score (1-5 scale)
- **NISQA Distortion** - Distortion score
- **NISQA Coloration** - Naturalness score (detects robotic sound)
- **ScoreQ-NR** - Speech-specific quality score

### CSV Export Columns
All metrics above plus:
- Filename, Duration, Decision (Accepted/Rejected)
- Rejection Reasons
- MOS Δ, ScoreQ Δ (improvement deltas)
- Human-readable interpretations for all metrics

## 🔧 Troubleshooting

### "Numpy is not available" Error

This indicates NumPy 2.x incompatibility:

```bash
pip uninstall numpy
pip install "numpy<2.0"
```

### Low Dictionary Coverage on Valid Speech

**Option 1:** Add domain-specific terms to allowlist (line 100)

```python
DOMAIN_ALLOWLIST = {
    'ptt', 'dsp', 'ai', 'ml',
    'yourterm1', 'yourterm2'  # Add custom terms
}
```

**Option 2:** Lower threshold (not recommended)

```python
MIN_DICTIONARY_COVERAGE = 0.5  # Reduce from 0.6
```

### Too Many False Rejections

1. **For slower speech:**
   ```python
   MIN_WORD_DENSITY = 0.5  # Reduce from 0.8
   ```

2. **For short clips/commands:**
   ```python
   SHORT_CLIP_THRESHOLD = 5.0  # Increase from 3.0
   MIN_WORD_DENSITY_SHORT = 0.3  # More relaxed
   ```

3. **Disable multi-failure requirement (not recommended):**
   Edit `validate_input_audio()` to allow single failures

### NISQA Not Available

Ensure `nisqa.tar` is in one of these locations:
- Current directory
- Parent directory (`../nisqa.tar`)
- Script directory

Check the console for specific error messages.

### ScoreQ Dependency Warning

You may see: `scoreq 1.0.1 requires numpy>=2.0.0`

**This is safe to ignore.** ScoreQ works correctly with NumPy 1.26.4 despite the warning.

## 🧠 Design Philosophy

**Precision over Recall:**

This tool intentionally **rejects uncertain cases** to avoid false enhancements. Some valid speech may be conservatively rejected, including:
- Heavy accents or non-native speakers
- Code-mixed audio (multilingual)
- Domain-specific jargon or technical terms
- Very noisy but intelligible speech

This is **acceptable and by design** for quality-first applications where:
- Avoiding false enhancements > Catching all valid inputs
- User experience prioritizes reliability over coverage
- It's better to skip enhancement than corrupt audio

### When to Tune Thresholds

- **High-precision use case** (default): Keep current thresholds
- **High-recall use case**: Lower MIN_WORD_DENSITY, MIN_DICTIONARY_COVERAGE
- **Domain-specific**: Add terms to DOMAIN_ALLOWLIST
- **Multi-language**: Set `ENABLE_LANGUAGE_DETECTION = False`

## 📚 Known Limitations

1. **English-Only Mode**
   - Non-English audio is rejected by default
   - Configurable via `ENABLE_LANGUAGE_DETECTION` flag
   - Multi-language support planned for future releases

2. **Conservative Validation**
   - Heavily accented speech may be rejected
   - Technical/domain jargon may fail dictionary checks
   - Solution: Add terms to `DOMAIN_ALLOWLIST`

3. **NISQA Dependency**
   - Overall quality checks disabled without weights file
   - ScoreQ-NR still works for speech-specific assessment
   - Enhancement decisions still functional (degraded)

4. **NumPy 2.x Incompatibility**
   - Must use NumPy < 2.0 for PyTorch/Whisper
   - ScoreQ dependency warning is safe to ignore
   - Future: Wait for PyTorch/Whisper NumPy 2.x support

5. **Processing Speed**
   - Whisper transcription is the bottleneck (~5-10s per file)
   - NISQA adds ~2-3s per file
   - GPU acceleration helps but still serial processing
   - Future: Implement batch transcription for folders

## 🏗️ Architecture

### Validation Pipeline

```
Input Audio
    ↓
Stage 1: Input Validation (Pre-screening)
    ├─ Transcription (Whisper + language detection)
    ├─ Word density check (speech rate)
    ├─ Dictionary coverage (lexical validity)
    ├─ Acoustic fallback (for Whisper failures)
    └─ Multi-failure requirement
    ↓
[PASS] → Stage 2: VoiceFixer Enhancement
    ↓
Stage 3: Output Quality Assessment
    ├─ NISQA (overall quality)
    ├─ ScoreQ-NR (speech-specific, dual domain)
    ├─ Dictionary coverage re-check
    └─ Three-tier ScoreQ decision logic
    ↓
[ACCEPT] → Enhanced Audio
[REJECT] → Original Audio
```

### Three-Tier ScoreQ Decision Logic

```
ScoreQ Delta < -0.1     → Always Reject (significant degradation)
ScoreQ Delta -0.1 to 0.1 → Check NISQA tiebreaker (stable/minimal change)
ScoreQ Delta >= 0.1     → Always Accept (clear improvement)
```

### Dictionary Coverage Re-Check

```
Case 1: Input had low coverage (<60%)
    → Check if enhancement fixed it
    → Require BOTH NISQA + ScoreQ improvement if still low

Case 2: Input had insufficient words (<5)
    → HIGH RISK scenario
    → Strict AND logic (both metrics must improve significantly)

Case 3: Input passed (≥60%) but enhanced regressed
    → CRITICAL corruption detected
    → Automatic rejection unless both metrics improved significantly
```

## 🔬 Modularized Version

For production integration, use the modularized components:

```python
# audio_validation.py - Input validation logic
# quality_assessment.py - NISQA/ScoreQ assessments  
# streamlit_ui.py - UI layer (optional)
```

See individual module documentation for API details.

## 📄 License

See parent VoiceFixer repository license.

## 🙏 Acknowledgments

- **VoiceFixer** - Audio restoration framework
- **Whisper** - OpenAI's speech recognition model
- **NISQA** - Perceptual quality assessment model
- **ScoreQ** - Speech quality assessment framework

## 📧 Support

For issues or questions:
1. Check Troubleshooting section above
2. Review configuration options
3. Enable Debug Mode for detailed logs
4. Check console output for specific errors

## 🔄 Version History

### v2.0 - REFINED (February 2026)
- ✅ Added language detection (English-only mode)
- ✅ Multi-failure validation (reduced false rejections)
- ✅ Context-aware thresholds (short clips)
- ✅ Domain-aware dictionary (allowlist)
- ✅ Acoustic fallback (Whisper failures)
- ✅ Three-tier ScoreQ logic (loosened rejection)
- ✅ Comprehensive interpretation functions
- ✅ CSV export with human-readable metrics
- ✅ Dictionary coverage re-check post-enhancement
- ✅ NumPy compatibility fix (1.26.4)

### v1.0 - Basic (January 2026)
- Initial pre-screening approach
- NISQA + ScoreQ integration
- Single file processing
