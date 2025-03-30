from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from scipy import signal
import os
import random
from config import CONTEXT_MAPPINGS

class AudioGenerator(nn.Module):
    """
    Enhanced AudioGenerator for high-quality neurologically optimized sounds.
    Uses sample-based synthesis and advanced frequency modulation.
    """
    def __init__(self, 
                 latent_dim=100, 
                 condition_dim=5,
                 sample_rate=22050,
                 duration=30.0,
                 output_channels=1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sample_rate = sample_rate
        self.duration = duration
        self.output_channels = output_channels
        self.samples_per_audio = int(sample_rate * duration)
        
        # Load instrument samples with style categorization
        self.instrument_samples = self.load_instrument_samples()
        
        # Load natural sound samples
        self.natural_samples = self.load_natural_samples()
        
        # Enhanced frequency modulation layer
        self.fm_layer = FrequencyModulationLayer(
            output_size=min(88200, int(sample_rate * duration)), 
            channels=1, 
            sample_rate=sample_rate
        )
        
        # Enhanced time-based modulation network
        self.time_modulation = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # Outputs: tempo, rhythm, harmony, texture, dynamics, progression, modulation, energy
        )
        
        # Main generator network with increased capacity
        self.main = nn.Sequential(
            nn.Linear(latent_dim + condition_dim + 8, 2048),  # Added time modulation
            nn.ReLU(),
            nn.Unflatten(1, (1, 2048)),
            
            # Increased number of channels for more variation
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.ReLU(),
            
            # Final projection with residual connection
            nn.Conv1d(512, output_channels, kernel_size=1),
            nn.Tanh()
        )
        
        # Enhanced progressive mixing network
        self.progressive_mixer = nn.Sequential(
            nn.Conv1d(output_channels * 3, output_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(output_channels * 2, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Enhanced dynamic sample selection network
        self.sample_selector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Outputs selection weights
        )
        
        # Add musical progression network
        self.progression_network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),  # Outputs: chord progression, key changes, etc.
            nn.Tanh()
        )
        
        # Define progression patterns for each context
        self.progression_patterns = {
            "sleep": {
                "pattern": "ambient",
                "tempo": "slow",
                "evolution": "smooth",
                "harmony": "consonant"
            },
            "focus": {
                "pattern": "rhythmic",
                "tempo": "moderate",
                "evolution": "steady",
                "harmony": "balanced"
            },
            "relaxation": {
                "pattern": "flowing",
                "tempo": "slow",
                "evolution": "gentle",
                "harmony": "warm"
            },
            "meditation": {
                "pattern": "ambient",
                "tempo": "very_slow",
                "evolution": "minimal",
                "harmony": "pure"
            },
            "energy": {
                "pattern": "rhythmic",
                "tempo": "fast",
                "evolution": "dynamic",
                "harmony": "energetic"
            }
        }
        
        # Define instrument mappings for each context
        self.context_instrument_mapping = {
            "sleep": {
                "weights": {
                    "pad": 0.4,
                    "strings": 0.3,
                    "piano": 0.2,
                    "ambient": 0.1
                }
            },
            "focus": {
                "weights": {
                    "piano": 0.35,
                    "strings": 0.25,
                    "electronic": 0.25,
                    "ambient": 0.15
                }
            },
            "relaxation": {
                "weights": {
                    "piano": 0.3,
                    "strings": 0.3,
                    "guitar": 0.2,
                    "ambient": 0.2
                }
            },
            "meditation": {
                "weights": {
                    "pad": 0.35,
                    "world": 0.25,
                    "ambient": 0.25,
                    "bells": 0.15
                }
            },
            "energy": {
                "weights": {
                    "drums": 0.3,
                    "bass": 0.25,
                    "electronic": 0.25,
                    "guitar": 0.2
                }
            }
        }
    
    def load_instrument_samples(self):
        """Load instrument samples from the samples directory with style categorization."""
        instruments = {}
        sample_dir = "samples/instruments"
        
        if not os.path.exists(sample_dir):
            print(f"Warning: Instrument samples directory not found at {sample_dir}")
            return {}
        
        # Define instrument categories and their styles
        instrument_categories = {
            "drums": {
                "styles": ["ambient", "rhythmic", "textural", "energetic", "meditative"],
                "subcategories": ["kick", "snare", "hihat", "tom", "cymbal", "frame_drum", "hand_drum", "tribal"]
            },
            "bass": {
                "styles": ["ambient", "rhythmic", "melodic", "deep", "pulsing"],
                "subcategories": ["acoustic", "electric", "synth", "sub", "drone"]
            },
            "lead": {
                "styles": ["melodic", "textural", "rhythmic", "expressive", "atmospheric"],
                "subcategories": ["guitar", "piano", "synth", "flute", "violin", "cello", "sitar", "kalimba"]
            },
            "pad": {
                "styles": ["ambient", "textural", "atmospheric", "evolving", "harmonic"],
                "subcategories": ["strings", "synth", "atmospheric", "choir", "drone", "texture"]
            },
            "world": {
                "styles": ["meditative", "rhythmic", "atmospheric", "ceremonial", "healing"],
                "subcategories": ["bells", "bowls", "gongs", "flutes", "drums", "chimes"]
            },
            "electronic": {
                "styles": ["energetic", "ambient", "rhythmic", "textural", "atmospheric"],
                "subcategories": ["synth", "pad", "bass", "drone", "texture", "sequence"]
            }
        }
        
        for category, config in instrument_categories.items():
            category_dir = os.path.join(sample_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory not found: {category_dir}")
                continue
            
            instruments[category] = {
                "styles": {style: [] for style in config["styles"]},
                "subcategories": {sub: [] for sub in config["subcategories"]}
            }
            
            # Load samples for each subcategory
            for subcategory in config["subcategories"]:
                subcategory_dir = os.path.join(category_dir, subcategory)
                if not os.path.exists(subcategory_dir):
                    continue
                
                for file in os.listdir(subcategory_dir):
                    if file.endswith(('.wav', '.mp3')):
                        try:
                            audio, sr = librosa.load(os.path.join(subcategory_dir, file), sr=self.sample_rate)
                            
                            # Analyze the audio to determine its style
                            style = self.analyze_audio_style(audio)
                            
                            # Resample to match target duration
                            if len(audio) > self.samples_per_audio:
                                audio = audio[:self.samples_per_audio]
                            else:
                                audio = np.pad(audio, (0, self.samples_per_audio - len(audio)))
                            
                            # Add to appropriate style category
                            if style in config["styles"]:
                                instruments[category]["styles"][style].append(audio)
                            instruments[category]["subcategories"][subcategory].append(audio)
                        except Exception as e:
                            print(f"Error loading {file}: {e}")
        
        return instruments
    
    def analyze_audio_style(self, audio):
        """Analyze audio to determine its style category."""
        # Extract features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Calculate statistics
        mfcc_mean = np.mean(mfcc, axis=1)
        centroid_mean = np.mean(spectral_centroid)
        rolloff_mean = np.mean(spectral_rolloff)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Determine style based on features
        if zcr_mean < 0.1 and centroid_mean < 2000:
            return "ambient"  # Soft, atmospheric
        elif zcr_mean > 0.2 and rolloff_mean > 4000:
            return "rhythmic"  # Regular patterns
        elif np.std(mfcc_mean) > 5:
            return "melodic"  # Clear melodies
        else:
            return "textural"  # Background textures
    
    def load_natural_samples(self):
        """Load natural sound samples from the samples directory."""
        natural_sounds = {}
        sample_dir = "samples/natural"
        
        if not os.path.exists(sample_dir):
            print(f"Warning: Natural samples directory not found at {sample_dir}")
            return {}
        
        # Define natural sound categories
        natural_categories = {
            "ambient": ["wind", "rain", "waves", "forest", "crickets"],
            "textural": ["white_noise", "pink_noise", "brown_noise", "ambient_hum"],
            "rhythmic": ["heartbeat", "ocean_waves", "rain_steady", "wind_chimes"],
            "melodic": ["birds", "crickets", "wind_chimes", "bells"]
        }
        
        for category, subcategories in natural_categories.items():
            category_dir = os.path.join(sample_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory not found: {category_dir}")
                continue
            
            natural_sounds[category] = {}
            
            for subcategory in subcategories:
                subcategory_dir = os.path.join(category_dir, subcategory)
                if not os.path.exists(subcategory_dir):
                    continue
                
                samples = []
                for file in os.listdir(subcategory_dir):
                    if file.endswith(('.wav', '.mp3')):
                        try:
                            audio, sr = librosa.load(os.path.join(subcategory_dir, file), sr=self.sample_rate)
                            
                            # Resample to match target duration
                            if len(audio) > self.samples_per_audio:
                                audio = audio[:self.samples_per_audio]
                            else:
                                audio = np.pad(audio, (0, self.samples_per_audio - len(audio)))
                            
                            samples.append(audio)
                        except Exception as e:
                            print(f"Error loading {file}: {e}")
                
                if samples:
                    natural_sounds[category][subcategory] = samples
        
        return natural_sounds
    
    def get_time_modulation(self, t):
        """Generate time-based modulation parameters."""
        # Normalize time to [0, 1]
        t_normalized = torch.tensor([t / self.duration], dtype=torch.float32)
        return self.time_modulation(t_normalized)
    
    def progressive_mix(self, base_audio, instrument_audio, natural_audio, t):
        """Mix audio progressively based on time with enhanced dynamics."""
        # Get time-based mixing weights
        time_mod = self.get_time_modulation(t)
        
        # Extract individual modulation parameters
        tempo = time_mod[0, 0]  # Tempo variation
        rhythm = time_mod[0, 1]  # Rhythm intensity
        harmony = time_mod[0, 2]  # Harmonic content
        texture = time_mod[0, 3]  # Textural density
        dynamics = time_mod[0, 4]  # Dynamic range
        progression = time_mod[0, 5]  # Musical progression
        modulation = time_mod[0, 6]  # Modulation depth
        energy = time_mod[0, 7]  # Overall energy
        
        # Get musical progression
        prog_params = self.progression_network(torch.tensor([t / self.duration], dtype=torch.float32))
        
        # Create dynamic mixing weights
        base_weight = 0.4 + 0.2 * torch.sin(2 * np.pi * tempo * t / self.duration)
        instrument_weight = 0.3 + 0.2 * torch.sin(2 * np.pi * rhythm * t / self.duration)
        natural_weight = 0.3 + 0.2 * torch.sin(2 * np.pi * texture * t / self.duration)
        
        # Apply dynamic modulation
        mod_factor = 1.0 + modulation * torch.sin(2 * np.pi * energy * t / self.duration)
        
        # Apply progression influence
        prog_factor = 1.0 + 0.3 * torch.sin(2 * np.pi * progression * t / self.duration)
        
        # Combine all audio streams with enhanced mixing
        combined = torch.cat([
            base_audio * base_weight * mod_factor * prog_factor,
            instrument_audio * instrument_weight * mod_factor * prog_factor,
            natural_audio * natural_weight * mod_factor * prog_factor
        ], dim=1)
        
        # Apply progressive mixing with enhanced dynamics
        mixed = self.progressive_mixer(combined)
        
        # Apply final dynamic range compression
        mixed = mixed * (1.0 + dynamics * torch.sin(2 * np.pi * harmony * t / self.duration))
        
        return mixed
    
    def select_samples_dynamically(self, features, t):
        """Select samples dynamically based on features and time."""
        # Get selection weights
        weights = self.sample_selector(features)
        
        # Modify weights based on time
        time_factor = t / self.duration
        weights = weights * (1 + torch.sin(2 * np.pi * time_factor))
        
        return weights
    
    def forward(self, z, condition, t=0):
        # Generate base audio
        combined = torch.cat([z, condition, self.get_time_modulation(t)], dim=1)
        base_audio = self.main(combined)
        base_audio = base_audio.view(-1, self.output_channels, self.samples_per_audio)
        
        # Apply frequency modulation with time-based variation
        modulated_audio = self.fm_layer(base_audio)
        
        # Get instrument and natural samples with dynamic selection
        instrument_audio = self.add_instrument_sounds(modulated_audio, condition.argmax().item(), t)
        natural_audio = self.add_natural_samples(modulated_audio, condition.argmax().item(), t)
        
        # Progressive mixing
        final_audio = self.progressive_mix(modulated_audio, instrument_audio, natural_audio, t)
        
        return final_audio
    
    def add_instrument_sounds(self, audio, context, t):
        """Add instrument sounds with dynamic selection based on time."""
        if not self.instrument_samples:
            return audio
        
        # Get features for dynamic selection
        features = self.main[0](torch.cat([audio.mean(dim=2), self.get_time_modulation(t)], dim=1))
        
        # Get dynamic selection weights
        weights = self.select_samples_dynamically(features, t)
        
        # Define context-specific instrument configurations with time-based variation
        context_configs = {
            "sleep": {
                "drums": {"style": "ambient", "weight": 0.1 + 0.05 * np.sin(2 * np.pi * t / self.duration)},
                "pad": {"style": "ambient", "weight": 0.2 + 0.1 * np.cos(2 * np.pi * t / self.duration)},
                "lead": {"style": "textural", "weight": 0.15 + 0.05 * np.sin(4 * np.pi * t / self.duration)}
            },
            "focus": {
                "drums": {"style": "rhythmic", "weight": 0.2 + 0.1 * np.sin(2 * np.pi * t / self.duration)},
                "bass": {"style": "rhythmic", "weight": 0.15 + 0.05 * np.cos(2 * np.pi * t / self.duration)},
                "pad": {"style": "textural", "weight": 0.1 + 0.05 * np.sin(4 * np.pi * t / self.duration)}
            },
            "relaxation": {
                "pad": {"style": "ambient", "weight": 0.25 + 0.1 * np.sin(2 * np.pi * t / self.duration)},
                "lead": {"style": "melodic", "weight": 0.2 + 0.05 * np.cos(2 * np.pi * t / self.duration)},
                "bass": {"style": "ambient", "weight": 0.1 + 0.05 * np.sin(4 * np.pi * t / self.duration)}
            },
            "energy": {
                "drums": {"style": "rhythmic", "weight": 0.3 + 0.1 * np.sin(2 * np.pi * t / self.duration)},
                "bass": {"style": "rhythmic", "weight": 0.2 + 0.05 * np.cos(2 * np.pi * t / self.duration)},
                "lead": {"style": "rhythmic", "weight": 0.15 + 0.05 * np.sin(4 * np.pi * t / self.duration)}
            },
            "meditation": {
                "pad": {"style": "ambient", "weight": 0.3 + 0.1 * np.sin(2 * np.pi * t / self.duration)},
                "lead": {"style": "textural", "weight": 0.2 + 0.05 * np.cos(2 * np.pi * t / self.duration)},
                "drums": {"style": "ambient", "weight": 0.1 + 0.05 * np.sin(4 * np.pi * t / self.duration)}
            }
        }
        
        # Get configuration for current context
        config = context_configs.get(context, context_configs["relaxation"])
        
        # Mix with selected instruments using dynamic weights
        for instrument, settings in config.items():
            if instrument in self.instrument_samples:
                style = settings["style"]
                base_weight = settings["weight"]
                
                if style in self.instrument_samples[instrument]["styles"]:
                    samples = self.instrument_samples[instrument]["styles"][style]
                    if samples:
                        # Select sample based on dynamic weights
                        sample_idx = torch.multinomial(weights, 1).item() % len(samples)
                        sample = samples[sample_idx]
                        
                        # Apply time-varying weight
                        weight = base_weight * (1 + 0.2 * np.sin(2 * np.pi * t / self.duration))
                        audio = audio * (1 - weight) + sample * weight
        
        return audio
    
    def add_natural_samples(self, audio, context, t):
        """Add natural sound samples with dynamic selection based on time."""
        if not self.natural_samples:
            return audio
        
        # Get features for dynamic selection
        features = self.main[0](torch.cat([audio.mean(dim=2), self.get_time_modulation(t)], dim=1))
        
        # Get dynamic selection weights
        weights = self.select_samples_dynamically(features, t)
        
        # Define context-specific natural sound configurations with time-based variation
        context_configs = {
            "sleep": {
                "ambient": {"weight": 0.3 + 0.1 * np.sin(2 * np.pi * t / self.duration), "subcategories": ["rain", "waves", "wind"]},
                "textural": {"weight": 0.2 + 0.05 * np.cos(2 * np.pi * t / self.duration), "subcategories": ["white_noise", "pink_noise"]}
            },
            "focus": {
                "textural": {"weight": 0.25 + 0.1 * np.sin(2 * np.pi * t / self.duration), "subcategories": ["white_noise", "pink_noise"]},
                "ambient": {"weight": 0.15 + 0.05 * np.cos(2 * np.pi * t / self.duration), "subcategories": ["ambient_hum"]}
            },
            "relaxation": {
                "ambient": {"weight": 0.3 + 0.1 * np.sin(2 * np.pi * t / self.duration), "subcategories": ["forest", "waves", "wind"]},
                "melodic": {"weight": 0.2 + 0.05 * np.cos(2 * np.pi * t / self.duration), "subcategories": ["birds", "wind_chimes"]}
            },
            "energy": {
                "rhythmic": {"weight": 0.3 + 0.1 * np.sin(2 * np.pi * t / self.duration), "subcategories": ["heartbeat", "ocean_waves"]},
                "ambient": {"weight": 0.2 + 0.05 * np.cos(2 * np.pi * t / self.duration), "subcategories": ["wind", "rain_steady"]}
            },
            "meditation": {
                "ambient": {"weight": 0.3 + 0.1 * np.sin(2 * np.pi * t / self.duration), "subcategories": ["wind", "forest"]},
                "melodic": {"weight": 0.2 + 0.05 * np.cos(2 * np.pi * t / self.duration), "subcategories": ["bells", "wind_chimes"]}
            }
        }
        
        # Get configuration for current context
        config = context_configs.get(context, context_configs["relaxation"])
        
        # Mix with selected natural sounds using dynamic weights
        for category, settings in config.items():
            if category in self.natural_samples:
                base_weight = settings["weight"]
                for subcategory in settings["subcategories"]:
                    if subcategory in self.natural_samples[category]:
                        samples = self.natural_samples[category][subcategory]
                        if samples:
                            # Select sample based on dynamic weights
                            sample_idx = torch.multinomial(weights, 1).item() % len(samples)
                            sample = samples[sample_idx]
                            
                            # Apply time-varying weight
                            weight = base_weight * (1 + 0.2 * np.sin(2 * np.pi * t / self.duration))
                            audio = audio * (1 - weight) + sample * weight
        
        return audio
    
    def generate_simple_audio(self, context, duration=30.0):
        """Generate audio using a simpler algorithm."""
        # Generate base frequency
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Get context parameters
        if context == "sleep":
            base_freq = 432
            harmonics = [1, 2, 3, 4]
            weights = [0.4, 0.3, 0.2, 0.1]
        elif context == "focus":
            base_freq = 528
            harmonics = [1, 2, 3]
            weights = [0.5, 0.3, 0.2]
        elif context == "relaxation":
            base_freq = 396
            harmonics = [1, 2, 3, 4, 5]
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        elif context == "energy":
            base_freq = 528
            harmonics = [1, 2, 3, 4]
            weights = [0.35, 0.3, 0.2, 0.15]
        else:  # meditation
            base_freq = 432
            harmonics = [1, 2, 3]
            weights = [0.5, 0.3, 0.2]
        
        # Generate audio
        audio = np.zeros(len(t))
        for harmonic, weight in zip(harmonics, weights):
            freq = base_freq * harmonic
            audio += weight * np.sin(2 * np.pi * freq * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Add some variation
        mod_freq = base_freq * (1 + np.random.uniform(-0.02, 0.02))
        mod_audio = np.zeros(len(t))
        for harmonic, weight in zip(harmonics, weights):
            freq = mod_freq * harmonic
            mod_audio += weight * np.sin(2 * np.pi * freq * t)
        
        # Add some noise
        noise = np.random.normal(0, 0.05, len(t))
        mod_audio = mod_audio + noise
        
        # Normalize
        mod_audio = mod_audio / np.max(np.abs(mod_audio))
        
        return torch.FloatTensor(mod_audio).unsqueeze(0)
    
    def generate_looped_audio(self, z, condition, total_duration=30.0, crossfade_duration=1.0):
        """Generate looped audio with crossfading."""
        # Generate base segment
        base_audio = self(z, condition)
        
        # Calculate number of segments needed
        num_segments = int(np.ceil(total_duration / self.duration))
        
        # Initialize output tensor
        output = torch.zeros(1, self.output_channels, int(self.sample_rate * total_duration))
        
        # Generate and mix segments with crossfading
        for i in range(num_segments):
            # Generate new segment
            segment = self(z, condition)
            
            # Calculate crossfade positions
            start_idx = i * self.samples_per_audio
            end_idx = start_idx + self.samples_per_audio
            
            # Apply crossfading
            if i > 0:
                # Create crossfade window
                fade_in = torch.linspace(0, 1, int(self.sample_rate * crossfade_duration))
                fade_out = 1 - fade_in
                
                # Apply crossfade
                output[:, :, start_idx:start_idx + len(fade_in)] *= fade_out
                segment[:, :, :len(fade_in)] *= fade_in
            
            # Add segment to output
            output[:, :, start_idx:end_idx] += segment
        
        return output

    def generate_dynamic_audio(self, z, condition, context, duration=30.0):
        """Generate dynamic audio with improved progression and variation."""
        batch_size = z.shape[0]
        samples_per_chunk = int(self.sample_rate * 5.0)  # 5-second chunks
        num_chunks = int(duration / 5.0)
        
        # Initialize output tensor
        output = torch.zeros(batch_size, self.output_channels, int(duration * self.sample_rate), device=z.device)
        
        # Now use self.progression_patterns and self.context_instrument_mapping
        context_settings = self.progression_patterns[context]
        instrument_mapping = self.context_instrument_mapping[context]
        
        # Generate chunks with variation
        for i in range(num_chunks):
            # Calculate time position for this chunk
            t_start = i * 5.0
            t = torch.arange(t_start, t_start + 5.0, 1/self.sample_rate, device=z.device)
            
            # Get time-based modulation for this chunk
            time_mod = self.get_time_modulation(t)
            progression = self.get_progression(t)
            
            # Generate base audio for this chunk
            chunk_z = z + torch.randn_like(z) * 0.1  # Add variation to latent vector
            chunk = self.main(chunk_z)
            chunk = chunk.view(batch_size, self.output_channels, -1)
            
            # Apply frequency modulation
            chunk = self.fm_layer(chunk)
            
            # Mix instrument layers for this chunk
            chunk_mixed = torch.zeros_like(chunk)
            for category, weight in instrument_mapping["weights"].items():
                if category in self.instrument_samples:
                    samples = self.instrument_samples[category]
                    
                    # Select style based on context and progression
                    style = context_settings["pattern"]
                    if style in samples["styles"]:
                        style_samples = samples["styles"][style]
                    else:
                        style_samples = samples["styles"]["ambient"]
                    
                    if style_samples:
                        # Select and mix samples with variation
                        num_samples = min(3, len(style_samples))
                        selected_samples = random.sample(style_samples, num_samples)
                        
                        for sample in selected_samples:
                            # Apply dynamic modulation
                            mod = torch.sin(2 * np.pi * time_mod.mean() * t / 5.0)
                            modulated_sample = torch.from_numpy(sample[:samples_per_chunk]).to(z.device)
                            modulated_sample = modulated_sample * (1 + 0.2 * mod)
                            
                            # Add slight time offset for natural feel
                            offset = random.randint(0, samples_per_chunk // 4)
                            chunk_mixed[:, :, offset:offset + len(modulated_sample)] += modulated_sample * weight
            
            # Apply chunk-specific processing
            if context == "energy":
                # Add dynamic processing for energy
                chunk_mixed = chunk_mixed * (1 + 0.3 * torch.sin(2 * np.pi * 2 * t / 5.0))
            elif context in ["sleep", "meditation"]:
                # Smooth processing for calming contexts
                chunk_mixed = F.avg_pool1d(chunk_mixed, kernel_size=5, stride=1, padding=2)
            
            # Add natural samples for appropriate contexts
            if context in ["sleep", "relaxation", "meditation"]:
                for category in ["ambient", "textural"]:
                    if category in self.natural_samples:
                        for subcategory, samples in self.natural_samples[category].items():
                            if samples:
                                sample = random.choice(samples)
                                sample_tensor = torch.from_numpy(sample[:samples_per_chunk]).to(z.device)
                                
                                # Apply gentle modulation
                                mod = torch.sin(2 * np.pi * time_mod.mean() * t / 5.0)
                                modulated_sample = sample_tensor * (1 + 0.1 * mod)
                                
                                # Mix with chunk
                                chunk_mixed += modulated_sample * 0.1
            
            # Apply progressive mixing
            chunk_mixed = self.progressive_mixer(chunk_mixed)
            
            # Add to output with crossfade
            if i > 0:
                # Apply crossfade
                fade_samples = int(0.1 * self.sample_rate)  # 100ms crossfade
                fade_in = torch.linspace(0, 1, fade_samples, device=z.device)
                fade_out = torch.linspace(1, 0, fade_samples, device=z.device)
                
                # Apply crossfade to previous chunk
                output[:, :, i*samples_per_chunk-fade_samples:i*samples_per_chunk] *= fade_out
                # Apply crossfade to current chunk
                chunk_mixed[:, :, :fade_samples] *= fade_in
            
            # Add chunk to output
            output[:, :, i*samples_per_chunk:(i+1)*samples_per_chunk] = chunk_mixed
        
        return output

class FrequencyModulationLayer(nn.Module):
    """
    Enhanced layer that applies frequency modulation to specific frequency bands
    relevant to neurological effects. Uses advanced spectral processing techniques.
    """
    def __init__(self, output_size=88200, channels=1, sample_rate=22050):
        super().__init__()
        
        self.output_size = output_size
        self.channels = channels
        self.sample_rate = sample_rate
        
        # Enhanced frequency bands with more specific ranges
        self.freq_bands = {
            'delta': (1, 4),     # Deep sleep, healing
            'theta': (4, 8),     # Meditation, creativity
            'alpha': (8, 13),    # Relaxation, calmness
            'beta': (13, 30),    # Focus, alertness
            'gamma': (30, 100),  # Cognitive processing
            'high_gamma': (100, 200)  # Peak performance
        }
        
        # Create learnable parameters for each frequency band
        self.band_weights = nn.ParameterDict({
            band: nn.Parameter(torch.tensor([0.5])) 
            for band in self.freq_bands
        })
        
        # Enhanced specific frequencies with more musical relationships
        self.specific_freqs = {
            'schumann': 7.83,    # Earth's resonance
            'healing': 528,      # DNA repair
            'solfeggio': 396,    # Liberation
            'sleep': 2.5,        # Deep sleep
            'focus': 14.1,       # Focus
            'harmony': 432,      # Universal harmony
            'meditation': 6.0,   # Meditation
            'energy': 30.0       # Energy boost
        }
        
        # Create learnable parameters for each specific frequency
        self.freq_weights = nn.ParameterDict({
            freq: nn.Parameter(torch.tensor([0.3])) 
            for freq in self.specific_freqs
        })
        
        # Add harmonic progression parameters
        self.harmonic_progression = nn.Parameter(torch.randn(8, 4))  # 8 chords, 4 notes each
        
        # Add time-based modulation network
        self.time_modulation = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Outputs: progression speed, modulation depth, etc.
        )
    
    def get_harmonic_progression(self, t):
        """Generate harmonic progression based on time."""
        # Normalize time to [0, 1]
        t_normalized = torch.tensor([t / self.output_size], dtype=torch.float32)
        
        # Get time-based modulation parameters
        mod_params = self.time_modulation(t_normalized)
        
        # Select chord based on time and modulation
        chord_idx = int((t / self.output_size) * 8) % 8
        chord = self.harmonic_progression[chord_idx]
        
        # Apply time-based modulation
        mod_depth = mod_params[0, 1]  # Use second parameter for modulation depth
        chord = chord * (1 + mod_depth * torch.sin(2 * np.pi * t / self.output_size))
        
        return chord
    
    def forward(self, x):
        batch_size = x.shape[0]
        result = x.clone()
        
        # Apply band-specific modulation in frequency domain
        for i in range(batch_size):
            for c in range(self.channels):
                # Get the actual length of this audio segment
                actual_length = x[i, c].shape[0]
                
                # Get device from input tensor
                device = x.device
                
                # Convert to frequency domain
                n_fft = 2**int(np.ceil(np.log2(actual_length)))
                padded = F.pad(x[i, c], (0, n_fft - actual_length))
                x_fft = torch.fft.rfft(padded)
                
                # Get frequency bins
                freqs = torch.fft.rfftfreq(n_fft, d=1.0/self.sample_rate).to(device)
                
                # Apply enhanced band modulation with smoother transitions
                for band, (low, high) in self.freq_bands.items():
                    # Create a smoother mask for this frequency band
                    center = (low + high) / 2
                    width = (high - low) / 2
                    
                    # Use a more musical transition curve
                    mask_values = torch.exp(-0.5 * ((freqs - center) / (width/2))**2)
                    mask_values = torch.where(freqs < low, torch.zeros_like(mask_values), mask_values)
                    mask_values = torch.where(freqs > high, torch.zeros_like(mask_values), mask_values)
                    
                    # Get time-based modulation
                    t = torch.arange(actual_length, device=device)
                    time_mod = torch.sin(2 * np.pi * center * t / self.sample_rate)
                    
                    # Apply modulated weight
                    band_weight = self.band_weights[band].to(device)
                    modulated_weight = band_weight * (1 + 0.2 * time_mod)
                    
                    if torch.any(mask_values > 0.01):
                        x_fft = x_fft + x_fft * mask_values * modulated_weight
                
                # Apply enhanced specific frequency modulation with harmonic progression
                for name, freq in self.specific_freqs.items():
                    if freq >= self.sample_rate / 2:
                        continue
                    
                    # Get harmonic progression for this frequency
                    progression = self.get_harmonic_progression(torch.arange(actual_length, device=device))
                    
                    # Find the closest bin to the target frequency
                    idx = torch.argmin(torch.abs(freqs - freq))
                    
                    # Get frequency weight
                    freq_weight = self.freq_weights[name].to(device)
                    
                    # Create a more musical enhancement with harmonics and progression
                    for harmonic_multiple in [1.0, 2.0, 3.0, 4.0, 5.0]:
                        harmonic_freq = freq * harmonic_multiple
                        if harmonic_freq >= self.sample_rate / 2:
                            continue
                            
                        # Find the closest bin to the harmonic frequency
                        harmonic_idx = torch.argmin(torch.abs(freqs - harmonic_freq))
                        
                        # Create a musical window around that frequency
                        window_size = min(15, n_fft // 100)
                        if window_size == 0:
                            window_size = 1
                            
                        start_idx = max(0, harmonic_idx - window_size)
                        end_idx = min(len(freqs) - 1, harmonic_idx + window_size)
                        
                        if start_idx >= end_idx:
                            continue
                        
                        # Musical strength based on harmonic relationship
                        harmonic_strength = 1.0 / (harmonic_multiple ** 1.2)
                        
                        # Create a musical window with progression influence
                        indices = torch.arange(start_idx, end_idx+1, device=device)
                        window = torch.exp(-0.5 * ((indices - harmonic_idx) / (window_size/3))**2)
                        
                        if window.shape[0] > 0:
                            # Apply the enhancement with progression
                            enhancement = freq_weight * window * harmonic_strength * (1 + progression.mean())
                            x_fft[start_idx:end_idx+1] = x_fft[start_idx:end_idx+1] * (1.0 + enhancement)
                
                # Apply phase correction and smoothing
                phase = torch.angle(x_fft)
                magnitude = torch.abs(x_fft)
                
                # Enhanced magnitude smoothing
                magnitude_smoothed = torch.cat([
                    magnitude[:1],
                    magnitude[1:-1] * 0.7 + magnitude[:-2] * 0.15 + magnitude[2:] * 0.15,
                    magnitude[-1:]
                ])
                
                # Reconstruct with smoothed magnitude and original phase
                x_fft_clean = magnitude_smoothed * torch.exp(1j * phase)
                
                # Convert back to time domain
                result_padded = torch.fft.irfft(x_fft_clean, n=n_fft)
                
                # Trim back to original size
                result[i, c] = result_padded[:actual_length]
        
        return result

class AudioDiscriminator(nn.Module):
    """
    Discriminator for audio generation. 
    Evaluates whether an audio sample is real or generated.
    Also predicts the context labels for conditional generation.
    """
    def __init__(self, input_size=88200, condition_dim=8, output_channels=1):
        super(AudioDiscriminator, self).__init__()
        
        self.input_size = input_size
        self.condition_dim = condition_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, output_channels, input_size)
            nn.Conv1d(output_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 16, input_size/2)
            
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 32, input_size/4)
            
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 64, input_size/8)
            
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 128, input_size/16)
            
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 256, input_size/32)
            
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 512, input_size/64)
        )
        
        # Calculate the flattened size after convolutions
        self.flat_size = 512 * (input_size // 64)
        
        # Dense layers for real/fake prediction
        self.validity_layers = nn.Sequential(
            nn.Linear(self.flat_size, 1),
            nn.Sigmoid()
        )
        
        # Dense layers for context prediction (auxiliary classifier)
        self.context_layers = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, condition_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(-1, self.flat_size)
        
        # Get validity prediction
        validity = self.validity_layers(x)
        
        # Get context prediction
        context = self.context_layers(x)
        
        return validity, context 

class InstrumentEnhancedGenerator(nn.Module):
    """
    Enhanced audio generator that uses real instrument samples and advanced synthesis techniques.
    """
    def __init__(self, 
                 latent_dim=100, 
                 condition_dim=8,
                 sample_rate=22050,
                 duration=4.0,
                 output_channels=1):
        super(InstrumentEnhancedGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.sample_rate = sample_rate
        self.duration = duration
        self.output_channels = output_channels
        
        # Calculate final output size
        self.output_size = int(sample_rate * duration)
        
        # Instrument embedding layer
        self.instrument_embedding = nn.Embedding(6, 64)  # 6 different instruments
        
        # Base frequency generator
        self.freq_generator = FrequencyModulationLayer(self.output_size)
        
        # Instrument synthesis network
        self.instrument_net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2)
        )
        
        # Mixing network
        self.mixer = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
        
        # Load instrument samples
        self.instrument_samples = self.load_instrument_samples()
        
    def load_instrument_samples(self):
        """Load pre-processed instrument samples"""
        samples = {}
        instrument_paths = {
            'piano': 'samples/piano/',
            'flute': 'samples/flute/',
            'violin': 'samples/violin/',
            'drums': 'samples/drums/',
            'guitar': 'samples/guitar/',
            'synth': 'samples/synth/'
        }
        
        for instrument, path in instrument_paths.items():
            if os.path.exists(path):
                samples[instrument] = []
                for file in os.listdir(path):
                    if file.endswith('.wav'):
                        audio, _ = librosa.load(os.path.join(path, file), sr=self.sample_rate)
                        samples[instrument].append(audio)
        
        return samples
    
    def generate_instrument_sound(self, instrument, duration):
        """Generate sound for a specific instrument"""
        if instrument not in self.instrument_samples:
            return torch.zeros(1, int(duration * self.sample_rate))
        
        # Select a random sample from the instrument's samples
        sample = np.random.choice(self.instrument_samples[instrument])
        
        # Resample to match duration
        target_length = int(duration * self.sample_rate)
        if len(sample) > target_length:
            start = np.random.randint(0, len(sample) - target_length)
            sample = sample[start:start + target_length]
        else:
            # Loop the sample if it's too short
            repeats = int(np.ceil(target_length / len(sample)))
            sample = np.tile(sample, repeats)[:target_length]
        
        return torch.from_numpy(sample).float()
    
    def forward(self, z, condition, instrument_idx=None):
        """Generate audio based on latent vector, condition, and instrument"""
        device = z.device
        
        # Generate base frequency
        base_audio = self.freq_generator(z)
        
        # Get instrument embedding
        if instrument_idx is not None:
            instrument_emb = self.instrument_embedding(instrument_idx)
        else:
            instrument_emb = torch.zeros(1, 64, device=device)
        
        # Process base audio through instrument network
        instrument_features = self.instrument_net(base_audio)
        
        # Mix features
        mixed_features = torch.cat([
            instrument_features.mean(dim=2),
            instrument_emb
        ], dim=1)
        
        # Generate final audio
        output = self.mixer(mixed_features)
        
        # Add instrument sound if specified
        if instrument_idx is not None:
            instrument_name = list(self.instrument_samples.keys())[instrument_idx]
            instrument_sound = self.generate_instrument_sound(instrument_name, self.duration)
            instrument_sound = instrument_sound.to(device)
            
            # Mix with base audio
            output = output * 0.7 + instrument_sound * 0.3
        
        return output

# Update the generate_audio function to use the new generator
def generate_audio(generator, context, emphasis=None, custom_weights=None, latent_dim=100, 
                   sample_rate=22050, enable_looping=False, total_duration=30.0, 
                   crossfade_duration=1.0, apply_effects=True, use_simple_mode=False,
                   instrument_idx=None):
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
        
        print(f"Starting audio generation with context: {context}, instrument: {instrument_idx}")
        
        # Generate audio
        if enable_looping:
            gen_audio = generator.generate_looped_audio(z, condition, total_duration, crossfade_duration)
        else:
            gen_audio = generator(z, condition, instrument_idx)
        
        # Add natural samples
        if apply_effects:
            gen_audio = generator.add_natural_samples(gen_audio, context)
        
        # Convert to numpy
        audio_np = gen_audio.squeeze().detach().cpu().numpy()
        
        return audio_np
        
    except Exception as e:
        print(f"Error in generate_audio: {e}")
        # Fall back to simple generation
        try:
            gen_audio = generator.generate_simple_audio(context, duration=total_duration)
            audio_np = gen_audio.squeeze().detach().cpu().numpy()
            st.warning("Using simplified audio generation due to hardware constraints.")
            return audio_np
        except Exception as simple_error:
            print(f"Simple generation also failed: {simple_error}")
            fallback_audio = np.sin(2 * np.pi * 440 * np.arange(sample_rate * 4) / sample_rate)
            st.error(f"Both generation methods failed. Using basic fallback audio.")
            return fallback_audio 