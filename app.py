import os
import io
import time
import torch
import numpy as np
import streamlit as st
from shutil import which
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from huggingface_hub import login

# ========== Streamlit / Device setup ==========
st.set_page_config(page_title="AI BeatGen", page_icon="🎵")
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

# ========== Hugging Face Token (from secrets or env) ==========
HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")
if HF_TOKEN:
    try:
        login(HF_TOKEN)
        st.sidebar.success("✅ Hugging Face token loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Failed to login with HF token: {e}")
else:
    st.sidebar.warning("⚠️ No HF_TOKEN found. Set it in Streamlit Secrets.")

# ========== Ensure ffmpeg is available ==========
ffmpeg_path = which("ffmpeg")
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
else:
    st.error("❌ ffmpeg not found. Please add it in packages.txt")
    st.stop()

# ========== Audio enhancement ==========
def enhance_audio(audio_array, sample_rate):
    # convert float waveform -> int16 PCM
    audio_int16 = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

    seg = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # int16
        channels=1,
    ).set_channels(2)

    seg = seg.normalize()
    reverbish = seg.low_pass_filter(5000).fade_in(100).fade_out(200)
    seg = seg.overlay(reverbish, gain_during_overlay=-3)
    return seg

# ========== Model loader (cached) ==========
@st.cache_resource(show_spinner=True)
def load_model_and_processor():
    st.write("⏳ Loading MusicGen model (first run may take a minute)...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium", use_auth_token=HF_TOKEN)
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-medium",
        use_auth_token=HF_TOKEN,
    ).to(DEVICE)
    model.eval()
    return model, processor

model, processor = load_model_and_processor()

# get sample rate safely
SAMPLE_RATE = getattr(getattr(processor, "feature_extractor", processor), "sampling_rate", 32000)

# ========== UI ==========
st.title("🎵 AI BeatGen – Context-Based Music Generator")

col1, col2, col3, col4 = st.columns(4)
genre = col1.selectbox("Genre", ["Any", "Pop", "Rock", "Hip-Hop", "Jazz", "Classical", "Electronic", "Ambient", "Folk"])
mood = col2.selectbox("Mood", ["Any", "Happy", "Sad", "Energetic", "Calm", "Epic", "Romantic"])
tempo = col3.selectbox("Tempo", ["Any", "Slow", "Medium", "Fast"])
instrument = col4.selectbox("Instrument", ["Any", "Piano", "Guitar", "Violin", "Drums", "Synth", "Orchestra"])

prompt = st.text_area("Describe the music you want:", "", height=80)
length_choice = st.slider("Track length (seconds)", 15, 60, 30)

if st.button("Generate Music 🎶"):
    if not prompt.strip():
        st.warning("Please enter a description to generate music.")
        st.stop()

    style_parts = []
    if genre != "Any": style_parts.append(f"genre: {genre}")
    if mood != "Any": style_parts.append(f"mood: {mood}")
    if tempo != "Any": style_parts.append(f"tempo: {tempo}")
    if instrument != "Any": style_parts.append(f"instrument: {instrument}")

    full_prompt = prompt
    if style_parts:
        full_prompt += " | " + ", ".join(style_parts)

    st.info(f"🎼 Generating music for: {full_prompt}")

    try:
        inputs = processor(text=[full_prompt], return_tensors="pt").to(DEVICE)
        max_new_tokens = int(length_choice * 8)

        with torch.inference_mode():
            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # handle shapes: (B, C, T) → 1D mono
        if audio_values.ndim == 3:
            audio_np = audio_values[0, 0].cpu().float().numpy()
        elif audio_values.ndim == 2:
            audio_np = audio_values[0].cpu().float().numpy()
        else:
            audio_np = audio_values.cpu().float().numpy().reshape(-1)

        enhanced_audio = enhance_audio(audio_np, SAMPLE_RATE)
        audio_buffer = io.BytesIO()
        enhanced_audio.export(audio_buffer, format="wav")
        audio_buffer.seek(0)

        st.audio(audio_buffer, format="audio/wav")
        st.download_button(
            "⬇ Download as WAV",
            audio_buffer,
            file_name="generated_music.wav",
            mime="audio/wav",
        )

    except Exception as e:
        st.error("❌ Music generation failed")
        st.exception(e)

st.markdown("💡 *Tip: Combine descriptive words with genre, mood, tempo, and instruments for the best results.*")
