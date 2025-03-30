# Context-Specific Audio Generator

A machine learning project that generates audio compositions with specific frequency profiles designed to induce particular neurological states (sleep, focus, relaxation, etc.).

## Overview

This project uses a conditional Generative Adversarial Network (GAN) to generate audio with specific frequency characteristics based on the FSD50K dataset. The model is designed to emphasize particular frequency bands and specific frequencies that are known to have effects on brain activity, based on neuroscientific research.

### Key Features

- **Context-specific audio generation**: Generate audio tailored for sleep, focus, relaxation, meditation, or energy
- **Frequency-aware neural network**: Special layers designed to enhance and modulate specific frequency bands
- **Customizable output**: Control which frequency characteristics to emphasize
- **Neurologically-informed design**: Based on established research on audio frequencies and brain activity

## How It Works

The system operates through several key components:

1. **Dataset Processing**: Filters the FSD50K dataset based on context labels and prepares audio data for training
2. **Conditional GAN**: A generator network creates audio conditioned on specific contexts, while a discriminator evaluates authenticity and context alignment
3. **Frequency Modulation Layer**: A specialized neural network layer that enhances specific frequency bands and target frequencies
4. **Analysis Tools**: Utilities to analyze the frequency characteristics of generated audio

### Neurological Frequency Bands

The system specifically targets these frequency ranges:

- **Delta (1-4 Hz)**: Associated with deep sleep and healing
- **Theta (4-8 Hz)**: Associated with meditation, creativity, and REM sleep
- **Alpha (8-13 Hz)**: Associated with relaxation, calmness, and present focus
- **Beta (13-30 Hz)**: Associated with alertness, focus, and active thinking
- **Gamma (30-100 Hz)**: Associated with higher cognitive processing

### Targeted Frequencies

Additionally, the model can emphasize specific frequencies known for particular effects:

- **Schumann Resonance (7.83 Hz)**: Earth's electromagnetic field resonance, associated with grounding
- **Solfeggio Frequency (528 Hz)**: Associated with healing and DNA repair
- **Earth Frequency (432 Hz)**: Alternative tuning frequency associated with harmony
- **Focus Frequency (14.1 Hz)**: Associated with increased concentration
- **Deep Sleep Frequency (2.5 Hz)**: Promotes deep, restorative sleep
- **Meditation Frequency (6.0 Hz)**: Enhances meditative states

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/context-audio-generator.git
cd context-audio-generator

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
# Train a model for sleep-inducing audio
python train.py --mode train --context sleep --num_epochs 100 --output_dir ./output/sleep

# Train a model for focus-enhancing audio
python train.py --mode train --context focus --num_epochs 100 --output_dir ./output/focus
```

### Generating Audio

```bash
# Generate sleep-inducing audio
python train.py --mode generate --context sleep --model_path ./output/sleep/models/generator_final.pth --num_samples 5 --output_dir ./generated/sleep

# Generate focus-enhancing audio
python train.py --mode generate --context focus --model_path ./output/focus/models/generator_final.pth --num_samples 5 --output_dir ./generated/focus
```

### Analyzing Generated Audio

```bash
# Analyze the frequency characteristics of generated audio
python -c "from utils import batch_analyze_frequencies; batch_analyze_frequencies('./generated/sleep')"
```

## Command Line Arguments

### Common Arguments

- `--data_dir`: Directory for the dataset (default: "./data")
- `--output_dir`: Directory to save outputs (default: "./output")
- `--context`: Context for audio generation, choose from ["sleep", "focus", "relaxation", "energy", "meditation"] (default: "sleep")
- `--sample_rate`: Audio sample rate (default: 22050)
- `--duration`: Duration of audio samples in seconds (default: 4.0)
- `--latent_dim`: Dimension of the latent space (default: 100)

### Training Arguments

- `--mode`: Mode of operation, either "train" or "generate" (default: "train")
- `--batch_size`: Batch size for training (default: 16)
- `--num_epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.0002)
- `--lambda_aux`: Weight for auxiliary loss (default: 1.0)
- `--num_workers`: Number of worker processes for data loading (default: 4)
- `--log_interval`: Interval for logging training metrics (default: 10)
- `--sample_interval`: Interval for generating sample audio (default: 1)
- `--save_interval`: Interval for saving model checkpoints (default: 10)

### Generation Arguments

- `--model_path`: Path to the trained generator model
- `--num_samples`: Number of audio samples to generate (default: 5)
- `--focus_condition`: Specific condition to focus on (-1 for random) (default: -1)

## Dataset

This project uses the FSD50K dataset:

> **FSD50K**: A dataset with 51,197 audio clips totaling over 100 hours of audio from the Freesound platform annotated using 200 audio classes derived from the AudioSet Ontology.

The dataset is available at: https://zenodo.org/records/4060432

## Important Notes

- Actual download and processing of the FSD50K dataset requires ~50GB of disk space
- Training a good model can take several hours on a GPU
- For optimal results, consider fine-tuning on specialized audio samples for your specific use case

## Scientific Basis

The relationship between audio frequencies and neurological states is based on research in:

1. **Brainwave entrainment**: The phenomenon where brainwaves synchronize to external stimuli
2. **Binaural beats**: When slightly different frequencies presented to each ear create a third "phantom" frequency
3. **Isochronic tones**: Regular patterns of sound followed by silence that can help entrain brainwaves

This project aims to create audio that can gently guide brainwave activity toward desired states.

## License

This project is available under the MIT License. 