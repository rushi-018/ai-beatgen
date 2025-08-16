import shutil
import subprocess
import streamlit as st

st.title("🔍 Debug Environment")

# Check ffmpeg
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    st.success(f"✅ ffmpeg found at: {ffmpeg_path}")
    try:
        version = subprocess.check_output([ffmpeg_path, "-version"]).decode("utf-8").split("\n")[0]
        st.text(version)
    except Exception as e:
        st.error(f"Could not run ffmpeg: {e}")
else:
    st.error("❌ ffmpeg not found!")

# Check libsndfile
try:
    import soundfile as sf
    st.success("✅ libsndfile available via Python soundfile")
except Exception as e:
    st.error(f"❌ libsndfile missing: {e}")

# Check torch & transformers
try:
    import torch
    st.success(f"✅ torch available: {torch.__version__}")
except Exception as e:
    st.error(f"❌ torch missing: {e}")

try:
    import transformers
    st.success(f"✅ transformers available: {transformers.__version__}")
except Exception as e:
    st.error(f"❌ transformers missing: {e}")
