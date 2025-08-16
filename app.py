import os
import io
import torch
import numpy as np
import streamlit as st
from shutil import which
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# -------------------- Setup --------------------
st.set_page_config(page_title="AI BeatGen", page_icon="🎵", layout="wide")
torch.set_num_threads(1)
DEVICE = torch.device("cpu")  # force CPU only

# -------------------- ffmpeg for audio --------------------
ffmpeg_path = which("ffmpeg")
if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
else:
    st.error("❌ Audio engine missing.")
    st.stop()

# -------------------- Audio Enhancement --------------------
def enhance_audio(audio_array, sample_rate):
    audio_int16 = np.clip(audio_array, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767.0).astype(np.int16)

    seg = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    ).set_channels(2)

    seg = seg.normalize()
    reverbish = seg.low_pass_filter(5000).fade_in(100).fade_out(200)
    return seg.overlay(reverbish, gain_during_overlay=-3)

# -------------------- Cached Model Loader --------------------
@st.cache_resource(show_spinner=True)
def load_model_and_processor(model_size: str):
    model_id = f"facebook/musicgen-{model_size}"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(DEVICE)
    model.eval()
    return model, processor

# -------------------- UI --------------------
st.title("🎵 AI BeatGen – Your Personal Music Generator")

# Model size selector (hidden from technical explanation)
model_size = st.sidebar.radio("Music Quality", ["Small (fast)", "Medium", "Large (best)"], index=0)
model_size = model_size.split()[0].lower()  # map to small/medium/large
model, processor = load_model_and_processor(model_size)
SAMPLE_RATE = getattr(getattr(processor, "feature_extractor", processor), "sampling_rate", 32000)

# Style controls
col1, col2, col3, col4 = st.columns(4)
genre = col1.selectbox("Genre", ["Any", "Pop", "Rock", "Hip-Hop", "Jazz", "Classical", "Electronic", "Ambient", "Folk"])
mood = col2.selectbox("Mood", ["Any", "Happy", "Sad", "Energetic", "Calm", "Epic", "Romantic"])
tempo = col3.selectbox("Tempo", ["Any", "Slow", "Medium", "Fast"])
instrument = col4.selectbox("Instrument", ["Any", "Piano", "Guitar", "Violin", "Drums", "Synth", "Orchestra"])

prompt = st.text_area("Describe your music idea:", "", height=80)
length_choice = st.slider("Track length (seconds)", 15, 60, 30)

# -------------------- Generate --------------------
if st.button("✨ Generate Music"):
    if not prompt.strip():
        st.warning("Please enter a description.")
        st.stop()

    # Build descriptive prompt
    style_parts = []
    if genre != "Any": style_parts.append(f"genre: {genre}")
    if mood != "Any": style_parts.append(f"mood: {mood}")
    if tempo != "Any": style_parts.append(f"tempo: {tempo}")
    if instrument != "Any": style_parts.append(f"instrument: {instrument}")
    full_prompt = prompt + (" | " + ", ".join(style_parts) if style_parts else "")

    st.info(f"🎼 Creating music for: *{full_prompt}*")

    try:
        inputs = processor(text=[full_prompt], return_tensors="pt").to(DEVICE)
        max_new_tokens = int(length_choice * 8)

        with torch.inference_mode():
            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
            audio_values = audio_values.to("cpu")

        # Convert output to numpy waveform
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

        st.success("✅ Music created!")
        st.audio(audio_buffer, format="audio/wav")
        st.download_button("⬇ Download Music", audio_buffer, file_name="beatgen.wav", mime="audio/wav")

    except Exception as e:
        st.error("⚠️ Something went wrong while generating music.")
