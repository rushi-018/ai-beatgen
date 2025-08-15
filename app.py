# import os
import os
from shutil import which

# Ensure ffmpeg is found
ffmpeg_path = which("ffmpeg")
if ffmpeg_path:
    os.environ["FFMPEG_BINARY"] = ffmpeg_path
else:
    raise RuntimeError("ffmpeg not found. Please check packages.txt installation.")

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import base64
import tempfile
from PIL import Image
import json
import re
import io
from test_model import generate_simple_audio
from transformers import pipeline
from frequency_generator import FrequencyGenerator
from context_analyzer import ContextAnalyzer
from utils import (
    plot_waveform, plot_spectrogram, analyze_frequency_content,
    analyze_specific_frequencies, get_audio_download_link, save_audio
)
from pydub.utils import which
AudioSegment.converter = which("ffmpeg")

from config import CONTEXT_MAPPINGS
from model import AudioGenerator, FrequencyModulationLayer

# Initialize components
freq_gen = FrequencyGenerator()
context_analyzer = ContextAnalyzer()

# Set page config
st.set_page_config(
    page_title="NeuroSound Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS to improve the appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4b67ad;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5e80ce;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .stAudio {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .freq-explanation {
        font-size: 0.9rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    .prompt-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sound-label {
        background-color: #f0f4ff;
        border-radius: 5px;
        padding: 3px 8px;
        margin-right: 5px;
        display: inline-block;
        font-size: 0.8rem;
        color: #4b67ad;
    }
</style>
""", unsafe_allow_html=True)

# Function to get a download link for a file
def get_audio_download_link(audio_data, filename, text):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        sf.write(tmp_file.name, audio_data, 22050)
        with open(tmp_file.name, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            b64 = base64.b64encode(audio_bytes).decode()
            href = f'<a href="data:audio/wav;base64,{b64}" download="{filename}">{text}</a>'
            return href

# Function to save matplotlib figure to an image
def fig_to_image(fig):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        fig.savefig(tmp_file.name, format='png', bbox_inches='tight')
        plt.close(fig)
        return Image.open(tmp_file.name)

# Context definitions
# CONTEXT_MAPPINGS = {
#     "sleep": ["rain", "water", "stream", "white noise", "ambient", "nature", "night", "crickets"],
#     "focus": ["white noise", "ambient", "hum", "air conditioner", "fan", "pink noise"],
#     "relaxation": ["birds", "nature", "water", "stream", "waves", "forest", "rain"],
#     "energy": ["beat", "rhythm", "percussion", "heartbeat", "upbeat"],
#     "meditation": ["chant", "singing bowl", "bell", "drone", "om", "binaural"]
# }

# Keywords for context detection
CONTEXT_KEYWORDS = {
    "sleep": {
        "keywords": ["sleep", "insomnia", "rest", "bed", "night", "tired", "relax", "calm", "peaceful", "dream"],
        "weight": 1.0
    },
    "focus": {
        "keywords": ["focus", "concentrate", "study", "work", "attention", "productivity", "deep work", "learning", "reading"],
        "weight": 1.0
    },
    "relaxation": {
        "keywords": ["relax", "calm", "chill", "unwind", "stress", "anxiety", "peace", "tranquil", "serene"],
        "weight": 1.0
    },
    "energy": {
        "keywords": ["energy", "active", "workout", "exercise", "motivation", "run", "gym", "power", "dynamic", "upbeat"],
        "weight": 1.0
    },
    "meditation": {
        "keywords": ["meditate", "mindful", "zen", "spiritual", "breath", "presence", "awareness", "consciousness"],
        "weight": 1.0
    }
}

# Instrument keywords and their contexts
INSTRUMENT_KEYWORDS = {
    "piano": ["piano", "keyboard", "keys", "classical", "melodic"],
    "flute": ["flute", "wind", "woodwind", "melodic", "peaceful"],
    "violin": ["violin", "strings", "classical", "emotional", "melodic"],
    "drums": ["drums", "percussion", "rhythm", "beat", "pulse"],
    "guitar": ["guitar", "strings", "acoustic", "electric", "chord"],
    "synth": ["synth", "electronic", "digital", "modern", "ambient"]
}

# Frequency band descriptions
FREQ_BAND_DESCRIPTIONS = {
    'Delta (1-4 Hz)': "Deep sleep, healing, dreamless sleep, immune system enhancement",
    'Theta (4-8 Hz)': "Meditation, creativity, REM sleep, deep relaxation, intuition",
    'Alpha (8-13 Hz)': "Relaxation, calmness, present focus, positive thinking",
    'Beta (13-30 Hz)': "Active thinking, focus, alertness, problem-solving",
    'Gamma (30-100 Hz)': "Higher cognitive processing, heightened perception, peak concentration"
}

# Specific frequency descriptions
SPECIFIC_FREQ_DESCRIPTIONS = {
    'Schumann Resonance (7.83 Hz)': "Earth's electromagnetic field resonance, grounding, well-being",
    'Solfeggio (528 Hz)': "Known as the 'miracle tone', associated with healing and DNA repair",
    'Earth Frequency (432 Hz)': "Alternative tuning standard, said to resonate with the universe",
    'Focus (14.1 Hz)': "Enhances focus and concentration, cognitive processing",
    'Deep Sleep (2.5 Hz)': "Promotes deep, restorative sleep phases",
    'Meditation (6.0 Hz)': "Facilitates meditative states and spiritual awareness"
}

def analyze_prompt(prompt):
    """Analyze the prompt to determine the most appropriate context."""
    # Initialize sentiment and emotion analysis
    sentiment_analyzer = pipeline("sentiment-analysis")
    emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    
    # Analyze sentiment and emotions
    sentiment = sentiment_analyzer(prompt)[0]
    emotions = emotion_analyzer(prompt)[0]
    
    # Keywords for different contexts
    context_keywords = {
        "sleep": ["sleep", "tired", "exhausted", "insomnia", "rest", "relax", "calm", "peaceful"],
        "focus": ["focus", "concentrate", "study", "work", "productive", "attention", "distracted"],
        "relaxation": ["relax", "stress", "anxiety", "peace", "calm", "tranquil", "serene"],
        "energy": ["energy", "motivation", "pump", "boost", "energetic", "active", "vitality", "power"],
        "meditation": ["meditate", "mindful", "peace", "zen", "spiritual", "awareness", "consciousness"]
    }
    
    # Calculate context scores based on keywords and emotions
    context_scores = {
        "sleep": 0,
        "focus": 0,
        "relaxation": 0,
        "energy": 0,
        "meditation": 0
    }
    
    # Check for keywords
    prompt_lower = prompt.lower()
    for context, keywords in context_keywords.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                context_scores[context] += 1
    
    # Consider emotions and sentiment
    if emotions['label'] in ['joy', 'love']:
        context_scores['energy'] += 2
        context_scores['focus'] += 1
    elif emotions['label'] in ['sadness', 'anger']:
        context_scores['energy'] += 2
        context_scores['relaxation'] += 1
    elif emotions['label'] in ['fear', 'surprise']:
        context_scores['relaxation'] += 2
        context_scores['meditation'] += 1
    
    # Consider sentiment
    if sentiment['label'] == 'POSITIVE':
        context_scores['energy'] += 1
        context_scores['focus'] += 1
    else:
        context_scores['relaxation'] += 1
        context_scores['meditation'] += 1
    
    # Get the context with highest score
    selected_context = max(context_scores.items(), key=lambda x: x[1])[0]
    
    # Define context relationships for more relevant alternatives
    context_relationships = {
        "sleep": ["relaxation", "meditation"],  # Sleep-related alternatives
        "focus": ["energy", "meditation"],      # Focus-related alternatives
        "relaxation": ["sleep", "meditation"],  # Relaxation-related alternatives
        "energy": ["focus", "relaxation"],      # Energy-related alternatives
        "meditation": ["relaxation", "sleep"]   # Meditation-related alternatives
    }
    
    # Get relevant alternatives based on the selected context
    alternatives = context_relationships[selected_context]
    
    return selected_context, alternatives

# Prepare the model for a specific context or load a cached version
@st.cache_resource
def load_model(context, latent_dim=100, sample_rate=22050, duration=30.0):
    try:
        condition_dim = len(CONTEXT_MAPPINGS[context])
        
        # Force CPU for better compatibility
        device = torch.device('cpu')
        
        # Initialize the generator
        generator = AudioGenerator(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            sample_rate=sample_rate,
            duration=duration,
            output_channels=1
        ).to(device)
        
        # If you have a trained model, load it
        model_path = f"./output/{context}/models/generator_final.pth"
        if os.path.exists(model_path):
            generator.load_state_dict(torch.load(model_path, map_location=device))
        
        generator.eval()
        
        return generator, device
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error loading model: {e}")
        print(f"Error details: {error_details}")
        st.error(f"Error loading model: {e}")
        st.expander("Error Details").code(error_details)
        return None, torch.device('cpu')

# Generate audio sample with improved quality
def generate_audio(generator, context, emphasis=None, custom_weights=None, latent_dim=100, 
                   sample_rate=22050, enable_looping=False, total_duration=30.0, 
                   crossfade_duration=1.0, apply_effects=True, use_simple_mode=False,
                   base_frequency=None):
    try:
        if use_simple_mode:
            print(f"Using simple audio generation for context: {context}")
            gen_audio = generator.generate_simple_audio(context, duration=total_duration)
            audio_np = gen_audio.squeeze().detach().cpu().numpy()
            return audio_np
            
        device = torch.device('cpu')
        generator = generator.to(device)
        
        # Sample random noise
        z = torch.randn(1, latent_dim, device=device)
        
        condition_dim = len(CONTEXT_MAPPINGS[context])
        
        # Create condition vector
        if emphasis is not None and emphasis < condition_dim:
            condition = torch.zeros(1, condition_dim, device=device)
            condition[0, emphasis] = 1.0
        else:
            condition = torch.ones(1, condition_dim, device=device) / condition_dim
        
        # Apply custom weights if provided
        if custom_weights:
            fm_layer = generator.fm_layer
            for band, weight in custom_weights['bands'].items():
                if band in fm_layer.band_weights:
                    fm_layer.band_weights[band].data = torch.tensor([float(weight)], device=device)
            for freq, weight in custom_weights['frequencies'].items():
                if freq in fm_layer.freq_weights:
                    fm_layer.freq_weights[freq].data = torch.tensor([float(weight)], device=device)
        
        print(f"Starting dynamic audio generation with context: {context}")
        
        # Use the new dynamic audio generation method
        gen_audio = generator.generate_dynamic_audio(z, condition, context, duration=total_duration)
        
        # Convert to numpy
        audio_np = gen_audio.squeeze().detach().cpu().numpy()
        
        # If base_frequency is provided, mix it with the generated audio
        if base_frequency is not None:
            t = np.linspace(0, total_duration, len(audio_np))
            base_wave = 0.1 * np.sin(2 * np.pi * base_frequency * t)
            audio_np = 0.7 * audio_np + 0.3 * base_wave
        
        return audio_np
        
    except Exception as e:
        print(f"Error in generate_audio: {e}")
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        
        # Fall back to simple generation with more informative message
        try:
            st.warning(f"Main generation failed: {str(e)}. Falling back to simple generation.")
            gen_audio = generator.generate_simple_audio(context, duration=total_duration)
            audio_np = gen_audio.squeeze().detach().cpu().numpy()
            return audio_np
        except Exception as simple_error:
            print(f"Simple generation also failed: {simple_error}")
            fallback_audio = np.sin(2 * np.pi * 440 * np.arange(sample_rate * 4) / sample_rate)
            st.error(f"Both generation methods failed. Using basic fallback audio.")
            st.expander("Error Details").code(error_details)
            return fallback_audio

# App title
st.markdown("<h1 class='main-header'>NeuroSound Generator 🧠🎵</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Generate optimized audio based on your needs by simply describing how you feel or what you want to achieve</p>", unsafe_allow_html=True)

# Prompt-based interface
st.markdown("<h2 class='sub-header'>Tell us what you need</h2>", unsafe_allow_html=True)

# User prompt input
user_prompt = st.text_area(
    "Describe how you're feeling or what you want to achieve:",
    placeholder="Example: I'm having trouble sleeping and need something to help me relax...",
    height=100
)

if user_prompt:
    # Analyze the prompt
    selected_context, alternatives = analyze_prompt(user_prompt)
    
    # Display the detected context and alternatives
    st.markdown("<div class='prompt-box'>", unsafe_allow_html=True)
    st.markdown(f"<strong>I'll create sounds for:</strong> <span class='sound-label'>{selected_context.capitalize()}</span>", unsafe_allow_html=True)
    
    if alternatives:
        alternatives_html = " ".join([f"<span class='sound-label'>{alt.capitalize()}</span>" for alt in alternatives])
        st.markdown(f"<strong>Alternative options:</strong> {alternatives_html}", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Sound Generation Settings")

# Advanced parameters
st.sidebar.markdown("### Advanced Parameters")

# Base duration and quality
quality_options = {
    "Standard": {"base_duration": 10.0, "sample_rate": 22050},
    "High": {"base_duration": 30.0, "sample_rate": 44100},
    "Ultra": {"base_duration": 60.0, "sample_rate": 48000}
}

quality = st.sidebar.radio("Sound Quality", list(quality_options.keys()), index=0)
quality_settings = quality_options[quality]

# Looping options
enable_looping = st.sidebar.checkbox("Enable Audio Looping", value=True)

# Only show these options if looping is enabled
if enable_looping:
    total_duration = st.sidebar.slider("Total Duration (seconds)", 30.0, 600.0, 120.0, 30.0)
    crossfade_duration = st.sidebar.slider("Crossfade Duration (seconds)", 0.5, 5.0, 2.0, 0.5)
else:
    total_duration = quality_settings["base_duration"]
    crossfade_duration = 2.0

# Add natural effects
apply_effects = st.sidebar.checkbox("Include Natural Sound Elements", value=True)

# Add simple mode option
use_simple_mode = st.sidebar.checkbox("Use Simple Mode", value=False,
                                     help="Generate audio using a simpler algorithm when experiencing performance issues")

# Seed setting for reproducibility
st.sidebar.markdown("### Randomization Settings")
use_random_seed = st.sidebar.checkbox("Use Fixed Random Seed", value=False,
                                    help="Enable to get reproducible results. Disable for more variety.")

if use_random_seed:
    seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
else:
    # Generate a random seed for this session
    seed = np.random.randint(0, 10000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    st.sidebar.info(f"Using random seed: {seed}")

# Audio Overlay
st.sidebar.markdown("### Audio Overlay")
user_audio_file = st.sidebar.file_uploader(
    "Upload audio to overlay (optional)",
    type=['wav', 'mp3', 'ogg'],
    help="Your audio will be mixed with the generated sound"
)

if user_audio_file:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(user_audio_file.getvalue())
        user_audio_path = tmp_file.name
    
    # Add a slider for mixing ratio
    mix_ratio = st.sidebar.slider(
        "Mix ratio (Generated:Your Audio)",
        0.0, 1.0, 0.7,
        help="Higher values give more prominence to the generated audio"
    )
else:
    user_audio_path = None
    mix_ratio = 0.7

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["Sound Generator", "Custom Frequencies", "About Sound Therapy", "Dataset Management"])

with tab1:
    st.markdown("<h2 class='sub-header'>Generate Optimized Audio</h2>", unsafe_allow_html=True)
    
    # Set the context based on prompt or selection
    if user_prompt:
        # Use the analyzed context and emphasis from the prompt
        selected_context = selected_context  # From analyze_prompt function
        emphasis_idx = None  # From analyze_prompt function
        
        # Display what was detected for transparency
        st.info(f"Using context '{selected_context.capitalize()}' based on your description.")
    else:
        # Manual selection if no prompt
        selected_context = st.selectbox(
            "Select a neurological context:",
            ["sleep", "focus", "relaxation", "meditation", "energy"],
            format_func=lambda x: x.capitalize()
        )
        
        # Emphasis option
        emphasis_options = ["Balanced mix"] + CONTEXT_MAPPINGS[selected_context]
        emphasis_choice = st.selectbox(
            "Emphasize a specific aspect:",
            range(len(emphasis_options)),
            format_func=lambda i: emphasis_options[i]
        )
        
        emphasis_idx = None if emphasis_choice == 0 else emphasis_choice - 1
    
    # Generation button
    generate_button = st.button("Generate Audio", key="quick_gen")
    
    if generate_button:
        with st.spinner(f"Generating {'looped ' if enable_looping else ''}audio for {selected_context}..."):
            try:
                # Load the model based on context
                generator, device = load_model(
                    selected_context, 
                    duration=quality_settings["base_duration"]
                )
                
                if generator is None:
                    st.error("Failed to load the audio generator model. Please check the error details above.")
                else:
                    # Generate base audio with single frequency
                    base_frequency = {
                        "sleep": 2.5,    # Delta wave
                        "focus": 14.1,   # Beta wave
                        "relaxation": 10.0,  # Alpha wave
                        "meditation": 6.0,   # Theta wave
                        "energy": 30.0   # Gamma wave
                    }[selected_context]
                    
                    # Generate the base audio
                    audio_data = freq_gen.generate_audio(
                        base_frequency=base_frequency,
                        duration=total_duration,
                        context=selected_context
                    )
                    
                    # Get relevant sample categories based on context
                    sample_categories = {
                        "sleep": ["ambient", "piano", "strings", "synth", "nature"],
                        "focus": ["piano", "guitar", "strings", "electronic", "ambient"],
                        "relaxation": ["piano", "guitar", "strings", "acoustic", "nature", "ambient"],
                        "meditation": ["ambient", "piano", "strings", "world", "nature"],
                        "energy": ["drums", "guitar", "bass", "rock", "electronic", "percussion"]
                    }
                    
                    # Load and mix samples with context-specific weights
                    sample_files = []
                    sample_weights = []
                    
                    for category in sample_categories[selected_context]:
                        sample_dir = os.path.join("samples", category)
                        if os.path.exists(sample_dir):
                            # Get all available samples for this category
                            available_samples = [f for f in os.listdir(sample_dir) if f.endswith(".wav")]
                            if available_samples:
                                # Context-specific sample filtering
                                if selected_context in ["relaxation", "focus"]:
                                    # Filter for gentle sounds
                                    gentle_samples = [f for f in available_samples if any(term in f.lower() for term in ["gentle", "soft", "calm", "peaceful", "quiet"])]
                                    if gentle_samples:
                                        available_samples = gentle_samples
                                elif selected_context == "energy":
                                    # Filter for energetic sounds
                                    energetic_samples = [f for f in available_samples if any(term in f.lower() for term in ["energetic", "upbeat", "dynamic", "powerful", "strong"])]
                                    if energetic_samples:
                                        available_samples = energetic_samples
                                
                                # Randomly select 2-3 samples from each category
                                num_samples = np.random.randint(2, 4)
                                selected_samples = np.random.choice(available_samples, min(num_samples, len(available_samples)), replace=False)
                                
                                for file in selected_samples:
                                    sample_files.append(os.path.join(sample_dir, file))
                                    # Adjust weights based on context and sample type
                                    if selected_context in ["relaxation", "focus"]:
                                        # Higher weights for gentle sounds
                                        weight = 1.2 if any(term in file.lower() for term in ["gentle", "soft", "calm", "peaceful", "quiet"]) else 0.8
                                    elif selected_context == "energy":
                                        # Higher weights for energetic sounds
                                        weight = 1.2 if any(term in file.lower() for term in ["energetic", "upbeat", "dynamic", "powerful", "strong"]) else 0.8
                                    else:
                                        # Equal weights for other contexts
                                        weight = 1.0
                                    sample_weights.append(weight)
                    
                    # Mix samples with base audio using weighted mixing
                    if sample_files:
                        sample_audio = freq_gen.mix_audio(sample_files, weights=sample_weights)
                        # Ensure lengths match
                        min_len = min(len(audio_data), len(sample_audio))
                        # Adjust mix ratio based on context with slight randomization
                        if selected_context == "energy":
                            # For energy, let the music be more prominent
                            base_weight = 0.1 + np.random.random() * 0.05  # Random between 0.1-0.15
                            final_audio = audio_data[:min_len] * base_weight + sample_audio[:min_len] * (1 - base_weight)
                        elif selected_context in ["relaxation", "focus"]:
                            # For relaxation and focus, balance between therapeutic and musical elements
                            base_weight = 0.15 + np.random.random() * 0.1  # Random between 0.15-0.25
                            final_audio = audio_data[:min_len] * base_weight + sample_audio[:min_len] * (1 - base_weight)
                        elif selected_context == "meditation":
                            # For meditation, slightly favor therapeutic frequencies
                            base_weight = 0.25 + np.random.random() * 0.1  # Random between 0.25-0.35
                            final_audio = audio_data[:min_len] * base_weight + sample_audio[:min_len] * (1 - base_weight)
                        else:  # sleep
                            # For sleep, favor therapeutic frequencies but keep musical elements present
                            base_weight = 0.3 + np.random.random() * 0.1  # Random between 0.3-0.4
                            final_audio = audio_data[:min_len] * base_weight + sample_audio[:min_len] * (1 - base_weight)
                    else:
                        final_audio = audio_data
                    
                    # Loop if needed
                    if total_duration > 30:
                        num_loops = int(total_duration / 30)
                        final_audio = freq_gen.loop_audio(final_audio, num_loops)
                    
                    # Normalize
                    final_audio = librosa.util.normalize(final_audio)
                    
                    # Save to temporary file
                    temp_file = "temp_output.wav"
                    freq_gen.save_audio(final_audio, temp_file)
                    
                    # Display audio player
                    st.audio(temp_file)
                    
                    # Download button
                    with open(temp_file, 'rb') as f:
                        st.download_button(
                            label="Download Audio",
                            data=f,
                            file_name="generated_audio.wav",
                            mime="audio/wav"
                        )
                    
                    # Clean up
                    os.remove(temp_file)
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Error in main generation button handler: {e}")
                print(f"Error details: {error_details}")
                st.error(f"An unexpected error occurred: {e}")
                st.expander("Error Details").code(error_details)

with tab2:
    st.markdown("<h2 class='sub-header'>Custom Frequency Settings</h2>", unsafe_allow_html=True)
    st.write("Fine-tune the base frequency and sample mixing for your personalized audio.")
    
    # Manual selection for custom frequencies tab
    tab2_context = st.selectbox(
        "Select a neurological context:",
        ["sleep", "focus", "relaxation", "meditation", "energy"],
        format_func=lambda x: x.capitalize(),
        key="tab2_context"
    )
    
    # Base frequency selection
    st.markdown("<h3>Base Frequency</h3>", unsafe_allow_html=True)
    
    # Default frequencies for each context
    default_frequencies = {
        "sleep": 2.5,    # Delta wave
        "focus": 14.1,   # Beta wave
        "relaxation": 10.0,  # Alpha wave
        "meditation": 6.0,   # Theta wave
        "energy": 30.0   # Gamma wave
    }
    
    base_frequency = st.slider(
        "Base Frequency (Hz)",
        0.1, 100.0,
        default_frequencies[tab2_context],
        0.1,
        help="The main frequency that will be used as the foundation of the audio"
    )
    
    # Sample mixing settings
    st.markdown("<h3>Sample Mixing</h3>", unsafe_allow_html=True)
    
    # Sample categories for the selected context
    sample_categories = {
        "sleep": ["ambient", "piano", "strings", "synth", "nature"],
        "focus": ["piano", "guitar", "strings", "electronic", "ambient"],
        "relaxation": ["piano", "guitar", "strings", "acoustic", "nature", "ambient"],
        "meditation": ["ambient", "piano", "strings", "world", "nature"],
        "energy": ["drums", "guitar", "bass", "rock", "electronic", "percussion"]
    }
    
    # Sample weights
    st.write("Adjust the mix of different instruments:")
    sample_weights = {}
    
    for category in sample_categories[tab2_context]:
        weight = st.slider(
            f"{category.capitalize()} Weight",
            0.0, 1.0,
            1.0,  # Equal weights by default
            0.1,
            help=f"Adjust how prominent {category} sounds are in the mix"
        )
        sample_weights[category] = weight
    
    # Base frequency vs samples mix
    st.markdown("<h3>Overall Mix</h3>", unsafe_allow_html=True)
    base_weight = st.slider(
        "Base Frequency vs Samples Mix",
        0.0, 1.0,
        0.3,
        0.1,
        help="Higher values make the base frequency more prominent"
    )
    
    # Generation button
    if st.button("Generate Custom Audio", key="custom_gen"):
        with st.spinner(f"Generating {'looped ' if enable_looping else ''}audio with custom settings..."):
            try:
                # Generate base audio
                audio_data = freq_gen.generate_audio(
                    base_frequency=base_frequency,
                    duration=total_duration,
                    context=tab2_context
                )
                
                # Load and mix samples
                sample_files = []
                weights = []
                
                for category, weight in sample_weights.items():
                    sample_dir = os.path.join("samples", category)
                    if os.path.exists(sample_dir):
                        for file in os.listdir(sample_dir):
                            if file.endswith(".wav"):
                                sample_files.append(os.path.join(sample_dir, file))
                                weights.append(weight)
                
                # Mix samples with base audio
                if sample_files:
                    sample_audio = freq_gen.mix_audio(sample_files, weights=weights)
                    min_len = min(len(audio_data), len(sample_audio))
                    final_audio = audio_data[:min_len] * base_weight + sample_audio[:min_len] * (1 - base_weight)
                else:
                    final_audio = audio_data
                
                # Loop if needed
                if total_duration > 30:
                    num_loops = int(total_duration / 30)
                    final_audio = freq_gen.loop_audio(final_audio, num_loops)
                
                # Normalize
                final_audio = librosa.util.normalize(final_audio)
                
                # Save to temporary file
                temp_file = "temp_output.wav"
                freq_gen.save_audio(final_audio, temp_file)
                
                # Display audio player
                st.audio(temp_file)
                
                # Download button
                with open(temp_file, 'rb') as f:
                    st.download_button(
                        label="Download Custom Audio",
                        data=f,
                        file_name="custom_audio.wav",
                        mime="audio/wav"
                    )
                
                # Clean up
                os.remove(temp_file)
                
            except Exception as e:
                st.error(f"Error generating custom audio: {e}")
                st.expander("Error Details").code(traceback.format_exc())

with tab3:
    st.markdown("<h2 class='sub-header'>Understanding Sound Therapy</h2>", unsafe_allow_html=True)
    st.write("""
    Sound therapy uses specific audio frequencies and patterns to influence brain activity and physiological states. 
    The brain has a natural tendency to synchronize with external rhythmic stimuli, a phenomenon known as 'entrainment'.
    """)
    
    # More detailed information about brainwaves
    st.markdown("<h3>Brain Frequency Bands and Their Effects</h3>", unsafe_allow_html=True)
    
    # Create a more visual explanation of frequency bands
    cols = st.columns(5)
    
    with cols[0]:
        st.markdown("#### Delta (1-4 Hz)")
        st.markdown("🌙 **Deep Sleep**")
        st.markdown("• Healing & restoration")
        st.markdown("• Dreamless sleep")
        st.markdown("• Immune system boost")
        st.markdown("• Pain reduction")
        st.markdown("• Unconscious processing")
    
    with cols[1]:
        st.markdown("#### Theta (4-8 Hz)")
        st.markdown("✨ **Meditation**")
        st.markdown("• Enhanced creativity")
        st.markdown("• Dream states")
        st.markdown("• Deep relaxation")
        st.markdown("• Intuition access")
        st.markdown("• Subconscious connection")
    
    with cols[2]:
        st.markdown("#### Alpha (8-13 Hz)")
        st.markdown("☁️ **Relaxation**")
        st.markdown("• Calm awareness")
        st.markdown("• Stress reduction")
        st.markdown("• Mind-body connection")
        st.markdown("• Positive thinking")
        st.markdown("• Learning readiness")
    
    with cols[3]:
        st.markdown("#### Beta (13-30 Hz)")
        st.markdown("🧠 **Focus**")
        st.markdown("• Active thinking")
        st.markdown("• Problem-solving")
        st.markdown("• Concentration")
        st.markdown("• Alertness")
        st.markdown("• Logical reasoning")
    
    with cols[4]:
        st.markdown("#### Gamma (30-100 Hz)")
        st.markdown("⚡ **Peak Processing**")
        st.markdown("• Higher consciousness")
        st.markdown("• Enhanced perception")
        st.markdown("• Complex problem solving")
        st.markdown("• Peak concentration")
        st.markdown("• Advanced cognition")
    
    # Information about special frequencies
    st.markdown("<h3>Special Frequencies in Sound Therapy</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Beyond the brain wave bands, specific frequencies have been studied for their unique properties:
    
    * **Schumann Resonance (7.83 Hz)** - The Earth's electromagnetic field resonance frequency. 
      Associated with grounding and alignment with natural rhythms.
      
    * **Solfeggio Frequencies** - Ancient scale frequencies used in sacred music, including the famous 
      528 Hz (transformation and "DNA repair") and 396 Hz (liberation from fear and guilt).
      
    * **Binaural Beats** - When two slightly different frequencies are presented separately to each ear, 
      the brain perceives a third "beat" frequency that can help induce specific states.
    
    * **Isochronic Tones** - Rhythmic pulsing of a single tone, creating strong entrainment effects 
      without requiring headphones.
    """)
    
    # Scientific basis
    st.markdown("<h3>Scientific Research</h3>", unsafe_allow_html=True)
    st.write("""
    The scientific understanding of sound therapy continues to evolve. Research areas include:
    
    1. **Neuroacoustics** - How sound affects neural activity and brain synchronization
    2. **Psychoacoustics** - The psychological response to sound and music
    3. **Chronobiology** - How rhythmic stimuli influence biological rhythms
    4. **Resonance Theory** - How vibrations affect physical structures including cells and organs
    """)
    
    # Applications
    st.markdown("<h3>Therapeutic Applications</h3>", unsafe_allow_html=True)
    st.write("""
    Sound therapy is being applied in various contexts:
    
    * **Sleep Enhancement** - Supporting deeper, more restorative sleep cycles
    * **Stress Reduction** - Activating the parasympathetic nervous system for relaxation
    * **Focus & Productivity** - Creating optimal brain states for concentration
    * **Pain Management** - Redirecting attention and promoting endorphin release
    * **Meditation Support** - Deepening meditative states and awareness
    """)
    
    # References
    st.markdown("<h3>Further Reading</h3>", unsafe_allow_html=True)
    st.markdown("""
    * [Neuroscience of Music: How Music Affects the Brain](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6165862/)
    * [Effects of Binaural Beats on EEG and Cognitive Performance](https://www.frontiersin.org/articles/10.3389/fnhum.2017.00557/full)
    * [Sound Frequency Exposure and Human Brain Activity](https://www.sciencedirect.com/science/article/abs/pii/S1388245718301949)
    * [Music as Medicine: The impact of healing harmonies](https://www.health.harvard.edu/blog/healing-through-music-2018072314496)
    """)
    
    # Cautionary note
    st.markdown("<h3>Important Considerations</h3>", unsafe_allow_html=True)
    st.write("""
    While sound therapy shows promise, it's important to note:
    
    * It should not replace professional medical treatment for serious conditions
    * Individual responses vary based on personal factors and preferences
    * Some frequencies may not be suitable for those with epilepsy or seizure disorders
    * Volume should be kept at comfortable levels to avoid hearing damage
    """)

with tab4:
    st.markdown("<h2 class='sub-header'>Dataset Management</h2>", unsafe_allow_html=True)
    st.write("Use this section to build and train on your own audio dataset.")
    
    # Dataset folder selection
    dataset_folder = st.text_input("Dataset Folder Path", 
                                  placeholder="Enter full path to your audio dataset folder")
    
    # Dataset organization options
    st.markdown("### Dataset Organization")
    st.write("""
    Your dataset folder should contain:
    - Audio files in formats: WAV, MP3, OGG, FLAC
    - Optionally organize files in subfolders by category (e.g., 'sleep', 'focus', etc.)
    """)
    
    # Dataset statistics (if folder provided)
    if dataset_folder and os.path.exists(dataset_folder):
        st.write(f"📁 Folder exists: {dataset_folder}")
        
        # Count files
        audio_files = []
        for root, dirs, files in os.walk(dataset_folder):
            audio_files.extend([f for f in files if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))])
        
        st.write(f"🎵 Found {len(audio_files)} audio files")
        
        # Display subfolders if any
        subfolders = [f for f in os.listdir(dataset_folder) if 
                     os.path.isdir(os.path.join(dataset_folder, f))]
        if subfolders:
            st.write(f"📂 Categories: {', '.join(subfolders)}")
        
        # Process dataset button
        if st.button("Process Dataset"):
            with st.spinner("Processing dataset files..."):
                try:
                    from model import build_dataset
                    processed_data, labels = build_dataset(dataset_folder)
                    st.success(f"Successfully processed {len(processed_data)} audio segments!")
                    
                    # Display unique labels
                    unique_labels = np.unique(labels)
                    st.write(f"Found categories: {', '.join(unique_labels)}")
                    
                    # Option to start training
                    if st.button("Train Model on Dataset"):
                        st.info("Model training feature coming soon!")
                except Exception as e:
                    st.error(f"Error processing dataset: {e}")
    else:
        if dataset_folder:
            st.error(f"Folder not found: {dataset_folder}")
        
        # Option to create sample dataset structure
        if st.button("Create Sample Dataset Structure"):
            try:
                # Create main dataset folder
                os.makedirs("dataset", exist_ok=True)
                
                # Create category subfolders
                categories = ["sleep", "focus", "relaxation", "meditation", "energy"]
                for category in categories:
                    os.makedirs(f"dataset/{category}", exist_ok=True)
                
                st.success(f"Created dataset structure at {os.path.abspath('dataset')}")
                st.info("Now you can add your audio files to each category folder.")
            except Exception as e:
                st.error(f"Error creating dataset structure: {e}")

# Footer
st.markdown("<div class='footer'>NeuroSound Generator - Advanced Sound Therapy for Optimal Brain States</div>", unsafe_allow_html=True)

def overlay_audio(generated_audio, user_audio_path, mix_ratio=0.7, sample_rate=22050):
    """
    Overlay user-provided audio with generated audio
    
    Args:
        generated_audio: NumPy array of generated audio
        user_audio_path: Path to user's audio file
        mix_ratio: 0-1 value determining how much of the generated audio to keep (vs user audio)
        sample_rate: Sample rate for loading user audio
        
    Returns:
        Mixed audio as NumPy array
    """
    try:
        # Load user audio
        user_audio, sr = librosa.load(user_audio_path, sr=sample_rate)
        
        # Ensure both audios are same length
        if len(user_audio) > len(generated_audio):
            user_audio = user_audio[:len(generated_audio)]
        else:
            # Pad user audio if shorter (loop it to fill the duration)
            repeats = int(np.ceil(len(generated_audio) / len(user_audio)))
            user_audio = np.tile(user_audio, repeats)[:len(generated_audio)]
            
        # Mix the audio
        mixed_audio = generated_audio * mix_ratio + user_audio * (1 - mix_ratio)
        return mixed_audio
    except Exception as e:
        st.error(f"Error overlaying audio: {e}")
        return generated_audio 