# import os
# import time
# import tempfile
# from io import BytesIO

# import numpy as np
# import librosa
# import librosa.display
# import soundfile as sf
# import streamlit as st
# import torch
# from voicefixer import VoiceFixer
# import matplotlib.pyplot as plt

# TARGET_SR = 44100  # processing/output sample rate

# def init_voicefixer():
#     return VoiceFixer()

# voice_fixer = init_voicefixer()

# st.write("Audio restoration (16 kHz)")

# # Input mode selection
# input_mode = st.radio("Select input mode:", ["File Upload", "Byte Array (Hex)"], horizontal=True)

# if input_mode == "File Upload":
#     uploaded = st.file_uploader("Upload audio (.wav or .m4a)", type=["wav", "m4a"])
#     byte_input = None
# else:
#     uploaded = None
#     st.write("Enter audio data as hex string (e.g., paste hex bytes from another source):")
#     hex_input = st.text_area("Hex bytes:", height=150, placeholder="Enter hex string here...")
    
#     file_format = st.selectbox("Select audio format:", [".wav", ".m4a"])
    
#     if hex_input:
#         try:
#             # Convert hex string to bytes
#             byte_input = bytes.fromhex(hex_input.replace(" ", "").replace("\n", ""))
#             st.success(f"Successfully parsed {len(byte_input)} bytes")
#         except ValueError as e:
#             st.error(f"Invalid hex string: {e}")
#             byte_input = None
#     else:
#         byte_input = None

# def load_audio(raw_bytes: bytes, ext: str):
#     """Decode audio -> (audio, sr) without forced resample."""
#     ext = ext.lower()
#     if ext == ".wav":
#         bio = BytesIO(raw_bytes)
#         with sf.SoundFile(bio) as f:
#             audio = f.read(dtype="float32")
#             sr = f.samplerate
#         return audio, sr
#     # .m4a: use tempfile so audioread/ffmpeg can open it
#     with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
#         tmp.write(raw_bytes)
#         tmp.flush()
#         audio, sr = librosa.load(tmp.name, sr=None, mono=True)
#     return audio, sr

# def normalize_pred(pred):
#     """Return 1D float32 numpy array or raise ValueError."""
#     if isinstance(pred, torch.Tensor):
#         pred = pred.detach().cpu().numpy()
#     pred = np.asarray(pred)
#     pred = np.squeeze(pred)  # remove singleton dims
#     if pred.ndim == 2:
#         # If shape (channels, samples) with small channel count, transpose
#         if pred.shape[0] <= 8 and pred.shape[0] < pred.shape[1]:
#             pred = pred.T
#         # Collapse trailing singleton channel
#         if pred.shape[1] == 1:
#             pred = pred[:, 0]
#     if pred.ndim != 1:
#         raise ValueError(f"Unexpected audio shape {pred.shape}")
#     if pred.size == 0:
#         raise ValueError("Empty prediction audio")
#     return pred.astype("float32")


# def render_analysis(audio: np.ndarray, sr: int, title: str):
#     """Render metrics + spectrogram + pitch plot for a mono or multi-channel clip."""
#     st.markdown(f"**{title}**")
#     try:
#         audio_mono = audio if np.ndim(audio) == 1 else np.mean(audio, axis=1)
#         duration_s = float(len(audio_mono) / sr)
#         peak_amp = float(np.max(np.abs(audio_mono)))
#         rms_amp = float(np.sqrt(np.mean(np.square(audio_mono))))

#         frame_length = 2048
#         hop_length = 256
#         rms_frames = librosa.feature.rms(y=audio_mono, frame_length=frame_length, hop_length=hop_length, center=True)[0]
#         noise_rms = float(np.percentile(rms_frames, 10)) if rms_frames.size else float("nan")
#         snr_db = float(10 * np.log10((rms_amp ** 2) / (noise_rms ** 2))) if noise_rms and noise_rms > 0 else float("nan")

#         fmin = librosa.note_to_hz('C2')
#         fmax = librosa.note_to_hz('C7')
#         pitch = librosa.yin(audio_mono, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length)
#         pitch = pitch.astype('float32')
#         pitch[pitch <= 0] = np.nan
#         mean_pitch = float(np.nanmean(pitch)) if np.isnan(pitch).sum() < pitch.size else float('nan')
#         median_pitch = float(np.nanmedian(pitch)) if np.isnan(pitch).sum() < pitch.size else float('nan')

#         st.write(f"Duration: {duration_s:.2f} s | Peak: {peak_amp:.3f} | RMS: {rms_amp:.3f} | Est. SNR: {snr_db if not np.isnan(snr_db) else 'N/A'} dB")
#         st.write(f"Pitch (Hz): mean {mean_pitch if not np.isnan(mean_pitch) else 'N/A'} | median {median_pitch if not np.isnan(median_pitch) else 'N/A'}")

#         # Spectrogram
#         S = np.abs(librosa.stft(audio_mono, n_fft=1024, hop_length=hop_length))
#         S_db = librosa.amplitude_to_db(S, ref=np.max)
#         fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
#         img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax_spec)
#         ax_spec.set(title='Spectrogram (dB)')
#         fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
#         st.pyplot(fig_spec)
#         plt.close(fig_spec)

#         # Pitch curve
#         times = librosa.frames_to_time(np.arange(len(pitch)), sr=sr, hop_length=hop_length)
#         fig_pitch, ax_pitch = plt.subplots(figsize=(8, 2.5))
#         ax_pitch.plot(times, pitch, linewidth=1.2)
#         ax_pitch.set(title='Estimated Pitch', xlabel='Time (s)', ylabel='Hz')
#         ax_pitch.grid(True, alpha=0.3)
#         st.pyplot(fig_pitch)
#         plt.close(fig_pitch)
#     except Exception as e:
#         st.warning(f"{title} analysis failed: {e}")

# # Process based on input mode
# if input_mode == "File Upload" and uploaded:
#     ext = os.path.splitext(uploaded.name)[1].lower()
#     raw_bytes = uploaded.read()
# elif input_mode == "Byte Array (Hex)" and byte_input:
#     ext = file_format
#     raw_bytes = byte_input
# else:
#     raw_bytes = None

# if raw_bytes:

#     try:
#         audio, orig_sr = load_audio(raw_bytes, ext)
#     except Exception as e:
#         st.error(f"Failed to decode file: {e}. For .m4a ensure ffmpeg is installed (brew install ffmpeg).")
#     else:
#         if orig_sr != TARGET_SR:
#             audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)

#         mode = st.radio("Voice fixer modes (0,1,2)", [0, 1, 2])

#         if torch.cuda.is_available():
#             use_cuda = st.radio("Use GPU", [True, False])
#             if use_cuda != list(voice_fixer._model.parameters())[0].is_cuda:
#                 device = "cuda" if use_cuda else "cpu"
#                 voice_fixer._model = voice_fixer._model.to(device)
#         else:
#             use_cuda = False

#         t0 = time.time()
#         pred_wav = voice_fixer.restore_inmem(audio, mode=mode, cuda=use_cuda)
#         dt = time.time() - t0

#         st.write(f"Original sample rate: {orig_sr} Hz (processed at {TARGET_SR} Hz)")
#         st.write("Original Audio:")
#         st.audio(raw_bytes, format="audio/wav" if ext == ".wav" else "audio/mp4")

#         st.write("Restored Audio (16 kHz):")
#         pred_wav_norm = None
#         try:
#             pred_wav_norm = normalize_pred(pred_wav)
#             out_buf = BytesIO()
#             sf.write(out_buf, pred_wav_norm, TARGET_SR, format="WAV")
#             file_bytes = out_buf.getvalue()
#             file_size_kb = len(file_bytes) / 1024
#             st.write(f"Processing time: {dt:.3f}s")
#             st.write(f"Processed file size: {file_size_kb:.2f} KB")
#             st.audio(file_bytes, format="audio/wav")
#             st.download_button(
#                 label="Download processed audio",
#                 data=file_bytes,
#                 file_name="restored_audio.wav",
#                 mime="audio/wav"
#             )
#         except Exception as e:
#             st.error(f"Failed to write restored audio: {e}")

#         # ===== Comparison: Original vs Restored =====
#         if pred_wav_norm is not None:
#             st.subheader("Voice Analysis Comparison")
#             tab_orig, tab_restored = st.tabs(["Original (input)", "Restored"])
#             with tab_orig:
#                 render_analysis(audio, TARGET_SR, "Original (after any resample)")
#             with tab_restored:
#                 render_analysis(pred_wav_norm, TARGET_SR, "Restored output")

import os
import time
import tempfile
from io import BytesIO

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import streamlit as st
import torch
import matplotlib.pyplot as plt

from voicefixer import VoiceFixer

# =========================
# Global config
# =========================
TARGET_SR = 16000   # speech-safe conditioning SR

# =========================
# Load model once
# =========================
@st.cache_resource
def load_voicefixer():
    return VoiceFixer()

voice_fixer = load_voicefixer()

st.title("VoiceFixer – Safe Two-Pass Speech Enhancement")
st.caption("Codec noise removal + bandwidth extension with hallucination control")

# =========================
# Audio loading
# =========================
def load_audio(raw_bytes: bytes, ext: str):
    if ext == ".wav":
        bio = BytesIO(raw_bytes)
        with sf.SoundFile(bio) as f:
            audio = f.read(dtype="float32")
            sr = f.samplerate
    else:
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            audio, sr = librosa.load(tmp.name, sr=None, mono=True)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio.astype("float32"), sr

# =========================
# Adaptive parameter logic
# =========================
def compute_adaptive_params(audio: np.ndarray):
    abs_audio = np.abs(audio)
    mean_energy = np.mean(abs_audio)
    std_energy = np.std(audio) + 1e-6

    # Proxy for stability / quality
    snr_proxy = mean_energy / std_energy

    # Adaptive mixes (safe bounded ranges)
    pass1_mix = np.clip(0.20 + 0.20 * snr_proxy, 0.20, 0.40)
    pass2_mix = np.clip(0.15 + 0.10 * snr_proxy, 0.15, 0.30)

    # Adaptive voiced-frame gate
    energy_gate = 0.5 * np.percentile(abs_audio, 20)

    return pass1_mix, pass2_mix, energy_gate

# =========================
# Safe VoiceFixer wrapper
# =========================
def run_voicefixer(audio, mode, use_cuda):
    out = voice_fixer.restore_inmem(audio, mode=mode, cuda=use_cuda)
    if isinstance(out, torch.Tensor):
        out = out.cpu().numpy()
    out = np.squeeze(out).astype("float32")
    return out[:len(audio)]

# =========================
# Energy-gated safe mixing
# =========================
def safe_mix(original, enhanced, mix, energy_gate):
    mask = np.abs(original) > energy_gate
    mixed = original.copy()
    mixed[mask] = (1 - mix) * original[mask] + mix * enhanced[mask]
    return np.clip(mixed, -1.0, 1.0)

# =========================
# Two-pass VoiceFixer pipeline
# =========================
def two_pass_voicefixer(audio, use_cuda):
    duration = len(audio) / TARGET_SR
    mean_energy = np.mean(np.abs(audio))

    pass1_mix, pass2_mix, energy_gate = compute_adaptive_params(audio)

    # -------- PASS 1: Codec + noise cleanup --------
    vf1 = run_voicefixer(audio, mode=2, use_cuda=use_cuda)  # Conservative
    stage1 = safe_mix(audio, vf1, pass1_mix, energy_gate)

    # Safety: skip spectral enhancement if risky
    if duration < 1.5 or mean_energy < 0.008:
        return stage1, {
            "pass1_mix": float(pass1_mix),
            "pass2_mix": 0.0,
            "energy_gate": float(energy_gate),
            "note": "Spectral enhancement skipped (risk control)"
        }

    # -------- PASS 2: Spectral enhancement --------
    vf2 = run_voicefixer(stage1, mode=0, use_cuda=use_cuda)  # General
    final = safe_mix(stage1, vf2, pass2_mix, energy_gate)

    return final, {
        "pass1_mix": float(pass1_mix),
        "pass2_mix": float(pass2_mix),
        "energy_gate": float(energy_gate),
        "note": "Two-pass enhancement applied"
    }

# =========================
# Visualization
# =========================
def render_spectrogram(audio, sr, title):
    S = np.abs(librosa.stft(audio, n_fft=1024))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

# =========================
# UI
# =========================
uploaded = st.file_uploader("Upload audio (.wav / .m4a)", type=["wav", "m4a"])

if uploaded:
    ext = os.path.splitext(uploaded.name)[1].lower()
    raw = uploaded.read()

    audio, orig_sr = load_audio(raw, ext)

    # ✅ FIXED librosa 0.10+ resample call
    if orig_sr != TARGET_SR:
        audio = librosa.resample(
            y=audio,
            orig_sr=orig_sr,
            target_sr=TARGET_SR
        )

    st.audio(raw)

    use_cuda = torch.cuda.is_available()

    if st.button("Enhance"):
        t0 = time.time()
        enhanced, params = two_pass_voicefixer(audio, use_cuda)
        dt = time.time() - t0

        # Output audio
        buf = BytesIO()
        sf.write(buf, enhanced, TARGET_SR, format="WAV")
        st.audio(buf.getvalue())
        st.download_button("Download enhanced audio", buf.getvalue(), "enhanced.wav")

        # Info panel
        st.success(f"Processing time: {dt:.2f}s")
        st.markdown("### Applied Parameters")
        st.json(params)

        # Comparison
        st.markdown("### Spectrogram Comparison")
        col1, col2 = st.columns(2)
        with col1:
            render_spectrogram(audio, TARGET_SR, "Original")
        with col2:
            render_spectrogram(enhanced, TARGET_SR, "Enhanced")
