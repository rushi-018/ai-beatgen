import os
import io
import time
import torch
import numpy as np
import streamlit as st
from shutil import which
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ========== Environment / Runtime tweaks (helpful on Streamlit Community Cloud) ==========
torch.set_num_threads(1)  # be gentle with CPU
DEVICE = torch.device("cpu")

# ========== Ensure ffmpeg is available for pydub ==========
ffmpeg_path = which("ffmpeg")
if ffmpeg_path:
    # pydub uses this to find ffmpeg explicitly
    AudioSegment.converter = ffmpeg_path
else:
    st.error("ffmpeg not found. Please check packages.txt installation.")
    st.stop()

# ========== Audio post-processing ==========
def enhance_audio(audio_array_1d_float, sample_rate):
    """
    audio_array_1d_float: numpy float32/float64 in [-1, 1], mono
    Returns a pydub.AudioSegment (stereo) with light polish.
    """
    # convert float [-1, 1] -> int16 PCM for robust WAV export
    audio_int16 = np.clip(audio_array_1d_float, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

    # build mono, then upmix to stereo + light FX
    seg = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # int16
        channels=1
    ).set_channels(2)  # upmix to stereo

    # gentle polish: normalize + soft LPF + short fades
    seg = seg.normalize()
    reverbish = seg.low_pass_filter(5000).fade_in(100).fade_out(200)
    seg = seg.overlay(reverbish, gain_during_overlay=-3)
    return seg

# ========== Cached model loader ==========
@st.cache_resource
def load_model_and_processor():
    with st.spinner("Downloading and loading the MusicGen model…"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for percent in range(0, 50, 10):
            progress_bar.progress(percent)
            status_text.text(f"Loading model files… {percent}%")
            time.sleep(0.15)

        # Processor (handles text -> conditioning & sampling rate info)
        processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
        progress_bar.progress(70)
        status_text.text("Processor loaded. Loading model weights…")

        # Model (CPU, float32)
        model = MusicgenForConditionalGeneration.from_pretrained(
            "facebook/musicgen-medium"
        ).to(DEVICE)
        model.eval()

        progress_bar.progress(100)
        status_text.text("Model loaded successfully ✅")
        return model, processor

model, processor = load_model_and_processor()

# Derive the target sampling rate robustly across transformers versions
SAMPLE_RATE = getattr(
    getattr(processor, "feature_extractor", processor),  # older/newer compatibility
    "sampling_rate",
    32000,  # MusicGen default
)

# ========== UI ==========
st.title("🎵 AI BeatGen – Advanced Context-Based Music Generator")

col1, col2, col3, col4 = st.columns(4)
genre = col1.selectbox("Genre", ["Any", "Pop", "Rock", "Hip-Hop", "Jazz", "Classical", "Electronic", "Ambient", "Folk"])
mood = col2.selectbox("Mood", ["Any", "Happy", "Sad", "Energetic", "Calm", "Epic", "Romantic"])
tempo = col3.selectbox("Tempo", ["Any", "Slow", "Medium", "Fast"])
instrument = col4.selectbox("Instrument", ["Any", "Piano", "Guitar", "Violin", "Drums", "Synth", "Orchestra"])

prompt = st.text_area("Describe the music you want:", "", height=80)
length_choice = st.slider("Select track length (seconds)", 15, 60, 30)

if st.button("Generate Music 🎶"):
    if not prompt.strip():
        st.warning("Please enter a description to generate music.")
        st.stop()

    # Build prompt with style hints
    style_parts = []
    if genre != "Any":
        style_parts.append(f"genre: {genre}")
    if mood != "Any":
        style_parts.append(f"mood: {mood}")
    if tempo != "Any":
        style_parts.append(f"tempo: {tempo}")
    if instrument != "Any":
        style_parts.append(f"instrument: {instrument}")

    full_prompt = prompt
    if style_parts:
        full_prompt += " | " + ", ".join(style_parts)

    st.info(f"🎼 Generating music for: {full_prompt}")

    try:
        # Prepare inputs
        inputs = processor(text=[full_prompt], return_tensors="pt").to(DEVICE)

        # ~8 tokens/sec ≈ MusicGen guidance; keep it modest for Community Cloud
        max_new_tokens = int(length_choice * 8)

        with torch.inference_mode():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # audio_values shape is (batch, channels, samples); pick first batch + first channel
        if audio_values.ndim == 3:
            # (B, C, T)
            audio_np = audio_values[0, 0].detach().cpu().float().numpy()
        elif audio_values.ndim == 2:
            # (B, T) rare case
            audio_np = audio_values[0].detach().cpu().float().numpy()
        else:
            # fallback
            audio_np = audio_values.detach().cpu().float().numpy().reshape(-1)

        # Post-process and export WAV
        enhanced_audio = enhance_audio(audio_np, SAMPLE_RATE)
        audio_buffer = io.BytesIO()
        enhanced_audio.export(audio_buffer, format="wav")
        audio_buffer.seek(0)

        st.audio(audio_buffer, format="audio/wav")
        st.download_button(
            label="⬇ Download Music as WAV",
            data=audio_buffer,
            file_name="generated_music.wav",
            mime="audio/wav"
        )

    except Exception as e:
        st.error("Generation failed. See details below.")
        st.exception(e)

st.markdown("💡 *Tip: Combine descriptive words with genre, mood, tempo, and instruments for the best results.*")
