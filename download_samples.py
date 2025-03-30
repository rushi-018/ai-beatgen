import os
import requests
import json
import time
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

class SampleDownloader:
    def __init__(self, api_key, base_dir="samples"):
        self.api_key = api_key
        self.base_dir = base_dir
        self.sample_rate = 22050
        self.api_url = "https://freesound.org/apiv2"
        
        # Create base directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Define sample categories and their search terms
        self.categories = {
            "ambient": {
                "drone": ["ambient drone", "meditation drone", "deep drone"],
                "atmospheric": ["ambient pad", "atmospheric pad", "ambient soundscape"],
                "textural": ["ambient texture", "soundscape texture", "atmospheric texture"],
                "minimalist": ["minimal ambient", "minimalist music", "ambient minimal"]
            },
            "piano": {
                "classical": ["classical piano", "piano solo", "piano classical"],
                "jazz": ["jazz piano", "piano jazz", "jazz piano solo"],
                "new_age": ["new age piano", "ambient piano", "relaxing piano"],
                "contemporary": ["modern piano", "piano contemporary", "piano improvisation"]
            },
            "strings": {
                "violin": ["violin solo", "classical violin", "violin melody"],
                "cello": ["cello solo", "cello classical", "cello melody"],
                "viola": ["viola solo", "viola classical", "viola melody"],
                "harp": ["harp solo", "celtic harp", "harp melody"]
            },
            "natural": {
                "rain": ["rain ambience", "rain soundscape", "rain nature"],
                "ocean": ["ocean waves", "ocean ambience", "waves nature"],
                "forest": ["forest ambience", "forest soundscape", "nature forest"],
                "birds": ["birds nature", "birds ambience", "birds soundscape"],
                "wind": ["wind ambience", "wind nature", "wind soundscape"]
            },
            "world": {
                "indian": ["indian classical", "sitar music", "tabla music"],
                "african": ["african drums", "djembe music", "african percussion"],
                "middle_eastern": ["middle eastern music", "oud music", "arabic music"],
                "tibetan": ["tibetan bowls", "tibetan bells", "tibetan meditation"],
                "bells": ["meditation bells", "temple bells", "zen bells"]
            },
            "electronic": {
                "pad": ["ambient pad", "synth pad", "atmospheric pad"],
                "synth": ["ambient synth", "atmospheric synth", "electronic ambient"],
                "bass": ["ambient bass", "deep bass", "electronic bass"],
                "drums": ["ambient drums", "electronic percussion", "atmospheric drums"],
                "textural": ["electronic texture", "synth texture", "ambient texture"]
            },
            "percussion": {
                "drums": ["ambient drums", "atmospheric drums", "textural drums"],
                "hand_percussion": ["hand drums", "frame drum", "hand percussion"],
                "ethnic_percussion": ["ethnic drums", "world percussion", "tribal drums"],
                "electronic_percussion": ["electronic drums", "synth percussion", "digital drums"]
            }
        }
    
    def search_freesound(self, query, filter_params=None):
        """Search for sounds on Freesound."""
        if filter_params is None:
            filter_params = {}
        
        url = f"{self.api_url}/search/text/"
        
        params = {
            "query": query,
            "fields": "id,name,tags,description,username,previews",  # Simplified fields
            "page_size": 15,
            "sort": "rating_desc",
            "filter": "duration:[1 TO 30]"  # Simplified filter
        }
        
        headers = {
            "Authorization": f"Token {self.api_key}"
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 400:
                print(f"Request URL: {response.url}")
                print(f"Response: {response.text}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return {"results": []}
    
    def download_file(self, url, filename):
        """Download a file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def process_audio_file(self, input_file, output_file):
        """Process audio file: normalize, convert to WAV, resample."""
        try:
            # Load audio
            audio, sr = librosa.load(input_file, sr=self.sample_rate)
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Save as WAV
            sf.write(output_file, audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return False
    
    def save_metadata(self, category, subcategory, sample_name, metadata):
        """Save metadata for a sample."""
        metadata_file = os.path.join(self.base_dir, category, subcategory, f"{sample_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def search_and_download(self, category, subcategory, search_terms, num_samples=3):
        """Search and download samples for a category and subcategory."""
        category_dir = os.path.join(self.base_dir, category)
        subcategory_dir = os.path.join(category_dir, subcategory)
        
        # Create directories
        os.makedirs(subcategory_dir, exist_ok=True)
        
        downloaded = 0
        for term in search_terms:
            if downloaded >= num_samples:
                break
            
            print(f"\nSearching for {term}...")
            try:
                results = self.search_freesound(term)
                
                if "results" not in results or not results["results"]:
                    print(f"No results found for {term}")
                    continue
                
                # Download remaining samples needed
                remaining = num_samples - downloaded
                for i, sound in enumerate(results["results"][:remaining]):
                    try:
                        # Get preview URL (more reliable than download URL)
                        preview_url = sound["previews"]["preview-hq-mp3"]
                        
                        # Generate filenames
                        temp_file = os.path.join(subcategory_dir, f"temp_{downloaded + i}.mp3")
                        final_file = os.path.join(subcategory_dir, f"{subcategory}_{downloaded + i}.wav")
                        
                        # Download preview
                        print(f"Downloading {sound['name']}...")
                        self.download_file(preview_url, temp_file)
                        
                        # Process audio
                        if self.process_audio_file(temp_file, final_file):
                            metadata = {
                                "id": sound["id"],
                                "name": sound["name"],
                                "tags": sound["tags"],
                                "description": sound["description"],
                                "username": sound["username"]
                            }
                            self.save_metadata(category, subcategory, f"{subcategory}_{downloaded + i}", metadata)
                            print(f"Successfully processed {sound['name']}")
                            downloaded += 1
                        
                        # Clean up temp file
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
                    except Exception as e:
                        print(f"Error downloading {sound['name']}: {e}")
                        continue
                    
                    # Rate limiting
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error searching for {term}: {e}")
                continue
    
    def download_all_samples(self):
        """Download samples for all categories and subcategories."""
        for category, subcategories in self.categories.items():
            print(f"\nProcessing category: {category}")
            for subcategory, search_terms in subcategories.items():
                print(f"\nProcessing subcategory: {subcategory}")
                self.search_and_download(category, subcategory, search_terms)

def main():
    # Get API key from environment variable or input
    api_key = input("Please enter your Freesound API key: ")
    
    try:
        downloader = SampleDownloader(api_key)
        
        # Test the API connection first with proper error handling
        print("Testing API connection...")
        test_result = downloader.search_freesound("piano")
        
        if "results" not in test_result:
            print("Error: Invalid API response. Your API key might be incorrect.")
            print("Please get a valid API key from: https://freesound.org/apiv2/apply/")
            return
            
        if not test_result["results"]:
            print("Warning: No results found, but API connection successful.")
        else:
            print("API connection successful! Starting download...")
            downloader.download_all_samples()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTroubleshooting steps:")
        print("1. Go to https://freesound.org/apiv2/apply/")
        print("2. Log in or create an account")
        print("3. Create a new API key")
        print("4. Copy the API key and try again")

if __name__ == "__main__":
    main() 