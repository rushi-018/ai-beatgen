import os
import io
import torch
import time
import streamlit as st
from shutil import which
from pydub import AudioSegment
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ===== Ensure ffmpeg is available =====
ffmpeg_path = which("ffmpeg")
if ffmpeg_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
else:
    st.error("ffmpeg not found. Please check packages.txt installation.")

# ===== Audio Enhancement =====
def enhance_audio(audio_array, sample_rate):
    audio = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    ).set_channels(2)  # Stereo
    audio = audio.normalize()  # Normalize volume
    reverb = audio.low_pass_filter(5000).fade_in(100).fade_out(200)
    audio = audio.overlay(reverb, gain_during_overlay=-3)
    return audio

# ===== Cached Model Loader =====
@st.cache_resource
def load_model():
    with st.spinner("Downloading and loading Hugging Face MusicGen model... Please wait."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        for percent in range(0, 50, 10):
            progress_bar.progress(percent)
            status_text.text(f"Loading model files... {percent}%")
            time.sleep(0.2)
        status_text.text("Model files loaded. Initializing processor...")

        processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
        progress_bar.progress(70)
        status_text.text("Processor loaded. Loading model weights...")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
        progress_bar.progress(100)
        status_text.text("Model loaded successfully ✅")
        return model, processor

# ===== Load model =====
model, processor = load_model()

# ===== Streamlit UI =====
st.title("🎵 AI BeatGen – Advanced Context-Based Music Generator")

# Style selectors
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
    else:
        # Build final prompt
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
        inputs = processor(text=[full_prompt], return_tensors="pt")

        with torch.inference_mode():
            audio_values = model.generate(**inputs, max_new_tokens=int(length_choice * 8))  # ~8 tokens/sec

        sample_rate = model.config.audio_encoder.sampling_rate

        # Enhance and play
        enhanced_audio = enhance_audio(audio_values[0].numpy(), sample_rate)
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

st.markdown("💡 *Tip: Combine descriptive words with genre, mood, tempo, and instruments for the best results.*")
