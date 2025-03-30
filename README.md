# AI Beat Generator

An AI-powered music generation system that creates context-aware ambient music for different purposes like sleep, focus, relaxation, meditation, and energy. The system uses deep learning to generate unique soundscapes tailored to specific contexts.

## Features

- **Context-Based Generation**: Create music for different purposes:
  - Sleep: Gentle, soothing sounds with delta and theta wave frequencies
  - Focus: Clear, structured sounds with alpha and beta frequencies
  - Relaxation: Calming ambient sounds with theta frequencies
  - Meditation: Deep, resonant tones with balanced frequencies
  - Energy: Dynamic, upbeat sounds with beta and gamma frequencies

- **Dynamic Audio Progression**:
  - Time-based modulation
  - Progressive mixing
  - Smooth transitions between audio segments
  - Context-specific audio processing

- **Multiple Instrument Categories**:
  - Traditional instruments (piano, strings, guitar)
  - Electronic sounds
  - World instruments
  - Ambient pads
  - Percussion and rhythmic elements

- **Real-time Audio Synthesis**:
  - Frequency modulation
  - Dynamic sample mixing
  - Adaptive sound generation
  - Custom audio effects

- **User-Friendly Interface**:
  - Streamlit web application
  - Simple context selection
  - Real-time audio preview
  - Download generated audio

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-beatgen.git
cd ai-beatgen
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create the necessary directory structure:

samples/
├── instruments/
│ ├── drums/
│ ├── bass/
│ ├── lead/
│ ├── pad/
│ ├── world/
│ └── electronic/
└── natural/
├── ambient/
├── textural/
├── rhythmic/
└── melodic/


5. Run the sample download script:
```bash
python download_samples.py
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed local URL (typically http://localhost:8501)

3. Select your desired context:
   - Choose from Sleep, Focus, Relaxation, Meditation, or Energy
   - Adjust any available parameters
   - Click "Generate" to create your music

4. Listen to the preview and download if desired

## Project Structure
ai-beatgen/
├── app.py # Main Streamlit application
├── model.py # Neural network model and audio generation
├── config.py # Configuration and context mappings
├── download_samples.py # Sample download script
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Technical Details

- **Audio Generation**: Uses PyTorch for neural network-based audio synthesis
- **Sample Rate**: 22050 Hz
- **Output Format**: WAV files
- **Duration**: Configurable, default 30 seconds
- **Frequency Ranges**:
  - Delta: 0.5-4 Hz
  - Theta: 4-8 Hz
  - Alpha: 8-13 Hz
  - Beta: 13-32 Hz
  - Gamma: 32+ Hz

## Troubleshooting

Common issues and solutions:

1. **Sample Download Fails**:
   - Check your internet connection
   - Ensure you have sufficient disk space
   - Try running the script with admin privileges

2. **Audio Generation Errors**:
   - Verify all dependencies are installed
   - Check available system memory
   - Ensure sample directories are properly structured

3. **Interface Issues**:
   - Clear browser cache
   - Restart the Streamlit server
   - Check console for error messages

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- Streamlit team for the web interface framework
- The open-source community for various audio processing libraries

## Contact

Rushiraj - [@rushi-018]((https://github.com/rushi-018))

Project Link: (https://github.com/rushi-018/ai-beatgen)
