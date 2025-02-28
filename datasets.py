import os
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import Dataset

# Load audio
import torchaudio

# Convert to mel-spectrogram
import librosa

# Mel-spectrogram save format
from PIL import Image

# Progress bar
import tqdm







augmentations = Compose(
        [
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

class GTZANDataset(Dataset):
    def __init__(self, resolution, root_dir, spectrogram_dir=None):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sr (int, optional): Sample rate for loading audio files.
            n_mels (int, optional): Number of mel bands to generate.
            fmax (int, optional): Maximum frequency for mel spectrogram.
        """
        # File management
        self.root_dir = root_dir # Directory where the audio genre folders are stored. Each genre folder should contain the list of audio files to use in the dataset.
        self.spectrogram_dir = spectrogram_dir # Directory to store mel-spectrograms of the audio files (Done to reduce loading times)
        self.genres = os.listdir(root_dir) #Testing only 1 genre (blues) os.listdir(root_dir)
        self.files = [] # Stores the file names of each waveform sample

        # Mel-spectrogram time-slice conversion
        self.resolution = resolution # (64, 64) # Resolution of each time slice (number of frequency bins, number of time steps / frames)
        self.hop_length = 1024
        self.sample_rate = 22050
        self.n_fft = 2048
        self.top_db = 80
        self.n_iter = 32 # Number of iterations for Griffin Linn mel inversion
        self.spectrogram_slices = [] # Stores each time slice of the converted mel-spectrograms. These are the actual items used in the diffusion process

        # Add each file to the dataset
        print("GENRES: ", self.genres)
        for genre in self.genres:
            genre_dir = os.path.join(root_dir, genre)
            if os.path.isdir(genre_dir):
                for file in os.listdir(genre_dir):
                    if file.endswith('.au'):
                        self.files.append((os.path.join(genre_dir, file), genre))

        # Add any existing mel-spectrogram slices to the dataset
        if os.path.exists(spectrogram_dir):
            for file in os.listdir(spectrogram_dir):
                if file.endswith('.png'):
                    self.spectrogram_slices.append(os.path.join(spectrogram_dir, file))


        # Create a mapping from unique genres to class labels
        self.genre_to_class = {genre: idx for idx, genre in enumerate(self.genres)}

    def __len__(self):
        return len(self.spectrogram_slices)

    def waveform_transform(self, waveform, target_length):
        """Transforms the loaded waveform to pad and crop to the target_seconds length.
           This may be used when the mel_spectrograms have not been saved to disk."""
        
        y = waveform.numpy().squeeze()
        
        # Pad or truncate the waveform to the target length
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')

            # Repeat the sample until the target length is reached
            num_repeats = int(np.ceil(target_length / len(y)))
            y = np.tile(y, num_repeats)
            y = y[:target_length]  # Truncate to the exact target length

        else:
            y = y[:target_length]

        # Convert the transformed waveform back to a tensor
        waveform = torch.tensor(y).unsqueeze(0)
        
        return waveform

    def save_mel_spectrograms(self):
        """Creates mel-spectrogram slices of each waveform file found in self.files and saves the individual slices."""

        for idx in tqdm.tqdm(range(len(self.files)), desc="Processing files"):
            audio_path, genre = self.files[idx]

            # Transform the waveform
            waveform, sample_rate = torchaudio.load(audio_path)
            #waveform = self.waveform_transform(waveform, 30)
            waveform = waveform.numpy()

            if waveform.ndim == 2:  # Stereo audio
                #print("Stereo audio detected! Mixing down to mono...")
                waveform = np.mean(waveform, axis=0)  # Mixdown to mono

            # Compute mel-spectrogram slices for the waveform
            #print("Waveform: ", waveform.shape, len(waveform))
            number_of_samples = len(waveform)#waveform.shape[1]
            slice_size = self.resolution[1] * self.hop_length - 1
            number_of_time_slices = number_of_samples // slice_size

            for slice in range(number_of_time_slices):
                spectrogram_path = os.path.join(self.spectrogram_dir, f"{os.path.basename(audio_path)}_{slice}.png")

                # Create directories if they don't exist
                if not os.path.exists(os.path.dirname(spectrogram_path)):
                    print("Directory does not exist! Making directory...")
                    os.makedirs(os.path.dirname(spectrogram_path))

                # Skip if files already exist
                if os.path.exists(spectrogram_path):
                    #print(f"File at {spectrogram_path} already exists!")
                    continue

                # Convert time-slice to spectrogram image
                audio_slice = waveform[slice_size * slice : slice_size * (slice + 1)]#.numpy()
                #print("Audio-slice: ", audio_slice.shape, slice, slice_size)
                # # mel_spectrogram_power = librosa.feature.melspectrogram(
                # #     y=audio_slice,
                # #     sr=self.sample_rate,
                # #     n_fft=self.n_fft,
                # #     hop_length=self.hop_length,
                # #     n_mels=self.resolution[1]
                # # )
                # # log_mel_spectrogram = librosa.power_to_db(mel_spectrogram_power, ref=np.max, top_db=self.top_db)
                # # bytedata = (((log_mel_spectrogram + self.top_db) * 255 / self.top_db).clip(0, 255) + 0.5).astype(np.uint8)
                # # image = Image.fromarray(bytedata)
                image = self.audio_to_mel_spectrogram(audio_slice)

                #print("Time-slice: ", slice, "Image-size: ", image.width, image.height)
                
                # Remove any silent time slices
                if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                    print(f"File: {audio_path} slice {slice} is silent!")
                    continue

                # Save the time slice image
                #print("Saving image to: ", spectrogram_path)
                image.save(spectrogram_path, format="PNG")
                
        return
    
    def audio_to_mel_spectrogram(self, input_audio):
        """Converts a waveform of a given size into an equivalent size mel-spectrogram. Inspired by this git repository: https://github.com/teticio/audio-diffusion/blob/main/audiodiffusion/mel.py"""
        mel_spectrogram_power = librosa.feature.melspectrogram(
                    y=input_audio,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.resolution[0]
                )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram_power, ref=np.max, top_db=self.top_db)
        bytedata = (((log_mel_spectrogram + self.top_db) * 255 / self.top_db).clip(0, 255) + 0.5).astype(np.uint8)
        image = Image.fromarray(bytedata)
        return image
    
    def mel_spectrogram_to_audio(self, mel_spectrogram):
        """Converts a mel-spectrogram of a given size into an equivalent size waveform. Inspired by this git repository: https://github.com/teticio/audio-diffusion/blob/main/audiodiffusion/mel.py"""
        bytedata = np.frombuffer(mel_spectrogram.tobytes(), dtype="uint8").reshape((mel_spectrogram.height, mel_spectrogram.width))
        log_S = bytedata.astype("float") * self.top_db / 255 - self.top_db
        S = librosa.db_to_power(log_S)
        audio = librosa.feature.inverse.mel_to_audio(
            S, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_iter=self.n_iter
        )
        return audio



    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #audio_path, genre = self.files[idx]
        image_path = self.spectrogram_slices[idx]
        sample_rate = 22050 # all files should have the same sample rate

        # Load saved mel-spectrogram slice if it exists
        if os.path.exists(image_path):
            #print("Cached files found! Loading...")
            image = Image.open(image_path)

        # Transform the spectrogram slices correctly
        image = augmentations(image)#image.convert("RGB"))

        sample = {'image': image, 'sample_rate': sample_rate}

        return sample