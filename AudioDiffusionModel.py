# Dataset and dataloader
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

import librosa
import numpy as np

# Preprocessing
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize, RandomHorizontalFlip

# Visualise dataset
import matplotlib.pyplot as plt
import sounddevice as sd

# Model
from torch import nn
from functools import partial # to create partial functions, allowing pre-filling of certain arguments
from einops import rearrange # needed by model blocks and training code
from torch import einsum  # needed by model blocks and training code
import math # Used within the positional embeddings

# Optimiser
from torch.optim import Adam

# Learning rate scheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# General
import torch.nn.functional as F # to access various functions for neural networks, like activation functions and loss calculations
import random

# Weights and Biases
import wandb
from itertools import product
from datetime import datetime

# Frechet Audio Distance measure 
from scipy.linalg import sqrtm
from torchaudio.transforms import MelSpectrogram
from torchvision.models import inception_v3
from torch.utils.data import DataLoader

# TODO: wanb.watch(model, ...) # Use to track exploding gradients etc.
# Also document my decisions and hypotheses as I go, so that I can use it within the writeup later!!!!!!!!!
# E.G. When testing different hyperparameters I noticed that there was a large impact to the model performance when adjusting the noise schedule, particularly by reducing the amount of noise added in each timestep. (This should be in line with the research in which the model can better approximate smaller amounts of noise)
# Test with just tones
# Test with different noise schedules

# Utility methods

from inspect import isfunction # to check if an object is a function, used in the UNet implementation.
from tqdm.auto import tqdm # for showing progress bar

def exists(x):
    """Utility function to check if a value exists (is not None)."""
    return x is not None

def default(val, d):
    """Utility function to return a value if it exists; otherwise, return a default value. If the default is a function, it calls the function to get the value."""
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)



# Dataset

class GTZANDataset(Dataset):
    def __init__(self, root_dir, cache_dir=None, db_cache_path=None, transform=None, sample_rate=22050, n_mels=256, fmax=15000, target_seconds=30):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sr (int, optional): Sample rate for loading audio files.
            n_mels (int, optional): Number of mel bands to generate.
            fmax (int, optional): Maximum frequency for mel spectrogram.
        """

        
        self.root_dir = root_dir
        self.cache_dir = cache_dir # directory to store mel-spectrograms of the audio files (Done to reduce loading times)
        self.cache_dir_db = db_cache_path # directory to store mel-spectrograms of the audio files in the DB scale (Done to reduce loading times)
        self.transform = transform 
        self.genres = [os.listdir(root_dir)[0]] #Testing only 1 genre (blues) os.listdir(root_dir)
        self.files = []
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmax = fmax
        self.target_length = sample_rate * target_seconds

        # Add each file to the dataset
        print("GENRES: ", self.genres)
        for genre in self.genres:
            genre_dir = os.path.join(root_dir, genre)
            if os.path.isdir(genre_dir):
                for file in os.listdir(genre_dir):
                    if file.endswith('.au'):
                        self.files.append((os.path.join(genre_dir, file), genre))

        # Create a mapping from unique genres to class labels
        self.genre_to_class = {genre: idx for idx, genre in enumerate(self.genres)}

    def __len__(self):
        return len(self.files)

    def pre_processing_transform(self, waveform, sample_rate, target_length, n_mels, fmax, pitch_shift_range=(-6, 6), time_stretch_range=(0.8, 1.0)):
        """Transforms the loadedwaveform to pitch-shift and time-stretch, then pads and crops to the target_seconds length.
           This will be used when the mel_spectrograms have not been saved to disk."""
        
        y = waveform.numpy().squeeze()

        # Apply random pitch shift within the specified range
        pitch_shift_steps = random.uniform(pitch_shift_range[0], pitch_shift_range[1])
        #y = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=pitch_shift_steps)
        
        # Apply random time stretch within the specified range
        time_stretch_factor = random.uniform(time_stretch_range[0], time_stretch_range[1])
        #y = librosa.effects.time_stretch(y, rate=time_stretch_factor)
        
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

    def pre_process_mel_spectrogram_db(self, mel_spectrogram_db):
        """Processes the mel_spectrogram in the db scale to augment the dataset"""

        # Apply random pitch shift (-2 to +2 semitones)
        shift_bins = random.choice([-2, -1, 0, 1, 2])
        S_shift = frequency_shift(mel_spectrogram_db, shift_bins)

        # Extract a random 5-second sample (assuming original length is 30 seconds)
        sample = extract_random_sample(S_shift)

        # Perform a frequency mask over the sample
        #time_masked_sample = time_mask(sample)

        #freq_masked_mel = frequency_mask(time_masked_sample)

        #print("New shape now: ", freq_masked_mel.shape)
        
        return sample #freq_masked_mel

    def save_mel_spectrograms(self):
        for idx in tqdm(range(len(self.files)), desc="Processing files"):
            audio_path, genre = self.files[idx]
            cache_path = os.path.join(self.cache_dir, f"{os.path.basename(audio_path)}.npy")
            db_cache_path = os.path.join(self.cache_dir_db, f"{os.path.basename(audio_path)}_db.npy")

            # Create directories if they don't exist
            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            if not os.path.exists(os.path.dirname(db_cache_path)):
                os.makedirs(os.path.dirname(db_cache_path))

            # Skip if files already exist
            if os.path.exists(cache_path) and os.path.exists(db_cache_path):
                continue

            self.save_mel_spectrogram(audio_path, cache_path, db_cache_path)

    def save_mel_spectrogram(self, audio_path, cache_path, db_cache_path):

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self.pre_processing_transform(waveform, sample_rate, self.target_length, self.n_mels, self.fmax)

        # Compute mel-spectrogram
        mel_spectrogram = create_mel_spectogram_from_waveform(waveform, sample_rate)

        # Convert to dB scale
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Save both mel-spectrogram and dB scale version to disk
        np.save(cache_path, mel_spectrogram)
        np.save(db_cache_path, mel_spectrogram_db)

        return mel_spectrogram, mel_spectrogram_db

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path, genre = self.files[idx]
        cache_path = os.path.join(self.cache_dir, f"{os.path.basename(audio_path)}.npy")
        db_cache_path = os.path.join(self.cache_dir_db, f"{os.path.basename(audio_path)}_db.npy")

        # Load saved mel-spectrogram and mel-spectrogram_db if they exist
        if os.path.exists(cache_path) and os.path.exists(db_cache_path):
            #print("Cached files found! Loading...")

            #mel_spectrogram = torch.tensor(np.load(cache_path))
            mel_spectrogram_db = torch.tensor(np.load(db_cache_path))

            #print("Shape before transformation: ", mel_spectrogram_db.shape)

            mel_spectrogram_db = self.pre_process_mel_spectrogram_db(mel_spectrogram_db)

            #print("Shape after transformation: ", mel_spectrogram_db.shape)
        else:
            print("File ", idx, "does not have a saved spectrogram! Remaking spectrograms...")

            mel_spectrogram, mel_spectrogram_db = self.save_mel_spectrogram(audio_path, cache_path, db_cache_path)

        # Normalize the mel-spectrogram
        mel_spectrogram_db = min_max_normalize(mel_spectrogram_db)

        # 'mel_spectrogram_db': mel_spectrogram_db
        sample_rate = 22050 # all files should have the same sample rate

        sample = {'mel_spectrogram': mel_spectrogram_db, 'sample_rate': sample_rate, 'genre': self.genre_to_class[genre]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitemdynamically__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path, genre = self.files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self.pre_processing_transform(waveform, sample_rate, self.target_length, self.n_mels, self.fmax)


        # # # step2 - converting audio np array to spectrogram
        mel_spectrogram = create_mel_spectogram_from_waveform(waveform, sample_rate)
        # # mel_spectogram = librosa.feature.melspectrogram(y=waveform.numpy().squeeze(),
        # #                                         sr=sample_rate, 
        # #                                             n_fft=2048, 
        # #                                             hop_length=512, 
        # #                                             win_length=None, 
        # #                                             window='hann', 
        # #                                             center=True, 
        # #                                             pad_mode='reflect', 
        # #                                             power=2.0,
        # #                                     n_mels=128)

        # # # step3 converting mel-spectrogrma back to wav file
        #reconstruction = reconstruct_waveform_from_mel_spectogram(mel_spectrogram, sample_rate)
        # # reconstruction = librosa.feature.inverse.mel_to_audio(mel_spectogram, 
        # #                                         sr=sample_rate, 
        # #                                         n_fft=2048, 
        # #                                         hop_length=512, 
        # #                                         win_length=None, 
        # #                                         window='hann', 
        # #                                         center=True, 
        # #                                         pad_mode='reflect', 
        # #                                         power=2.0, 
        # #                                         n_iter=32)
        
        # Play the reconstruction and compare to the original
        #play_audio_from_waveform(reconstruction, sample_rate)
        #play_audio_from_waveform(waveform.numpy().squeeze(), sample_rate)

        # 'audio_path': audio_path, 'genre': genre, 'waveform': waveform,

        sample = { 'mel_spectrogram': mel_spectrogram, 'sample_rate': sample_rate, 'genre': self.genre_to_class[genre]}

        if self.transform:
            sample = self.transform(sample)

        return sample


def frequency_mask(mel_spectrogram_db, F=15, num_masks=1):
    """
    Applies frequency masking to a mel spectrogram.
    
    Args:
        mel_spectrogram_db (torch.Tensor): Mel spectrogram in dB scale.
        F (int): Maximum width of the frequency mask.
        num_masks (int): Number of frequency masks to apply.
    
    Returns:
        torch.Tensor: Mel spectrogram with frequency masks applied.
    """
    # Ensure input is a torch tensor
    if not isinstance(mel_spectrogram_db, torch.Tensor):
        mel_spectrogram_db = torch.tensor(mel_spectrogram_db)
    
    cloned = mel_spectrogram_db.clone()
    
    num_mels = cloned.shape[0]

    min_value = cloned.min()
    
    for _ in range(num_masks):
        f = np.random.uniform(low=0.0, high=F)
        f = int(f)
        f_zero = np.random.uniform(low=0.0, high=num_mels - f)
        f_zero = int(f_zero)
        
        # Apply the mask
        cloned[f_zero:f_zero + f, :] = min_value #0
        
    return cloned

def time_mask(mel_spectrogram_db, T=40, num_masks=1):
    """
    Applies time masking to a mel spectrogram.
    
    Args:
        mel_spectrogram_db (torch.Tensor): Mel spectrogram in dB scale.
        T (int): Maximum width of the time mask.
        num_masks (int): Number of time masks to apply.
    
    Returns:
        torch.Tensor: Mel spectrogram with time masks applied.
    """
    # Ensure input is a torch tensor
    if not isinstance(mel_spectrogram_db, torch.Tensor):
        mel_spectrogram_db = torch.tensor(mel_spectrogram_db)
    
    cloned = mel_spectrogram_db.clone()
    
    num_frames = cloned.shape[1]

    min_value = cloned.min()
    
    for _ in range(num_masks):
        t = np.random.uniform(low=0.0, high=T)
        t = int(t)
        t_zero = np.random.uniform(low=0.0, high=num_frames - t)
        t_zero = int(t_zero)
        
        # Apply the mask
        cloned[:, t_zero:t_zero + t] = min_value #0
        
    return cloned

# # def frequency_shift(S, shift_bins):
# #     """Shifts the frequency bins of a mel_spectrogram_db"""
# #     S_shift = np.roll(S, shift_bins, axis=0)
# #     if shift_bins > 0:
# #         S_shift[:shift_bins, :] = 0
# #     else:
# #         S_shift[shift_bins:, :] = 0
# #     return S_shift

from scipy.ndimage import affine_transform

def frequency_shift(S, shift_bins):
    """
    Shifts the frequency bins of a mel_spectrogram_db by using interpolation to avoid artifacts.
    
    Args:
        S (np.ndarray): Mel spectrogram in dB scale.
        shift_bins (int): Number of frequency bins to shift.
    
    Returns:
        np.ndarray: Shifted mel spectrogram.
    """
    # Create an affine transformation matrix for vertical shift
    shift_matrix = np.array([[1, 0, 0], [0, 1, shift_bins], [0, 0, 1]])

    # Apply the affine transformation
    S_shift = affine_transform(S, matrix=shift_matrix, offset=0, output_shape=S.shape, order=1, mode='constant', cval=0)
    
    return S_shift

def extract_random_sample(mel_spectrogram_db, sample_duration=5, sr=22050, hop_length=512):
    """
    Extracts a random sample of specified duration from a mel spectrogram.
    
    Args:
        mel_spectrogram_db (np.ndarray): Mel spectrogram in dB scale.
        sample_duration (int): Duration of the sample in seconds. Default is 5 seconds.
        sr (int): Sampling rate of the audio. Default is 22050 Hz.
        hop_length (int): Hop length used to compute the mel spectrogram. Default is 512.
    
    Returns:
        np.ndarray: Mel spectrogram of the extracted sample in dB scale.
    """
    # Calculate the number of columns corresponding to the sample duration
    n_samples = int(sample_duration * sr / hop_length)
    
    # Calculate the maximum possible starting index for the sample
    max_start = mel_spectrogram_db.shape[1] - n_samples
    
    # Ensure there's enough data to extract the desired sample duration
    if max_start < 0:
        raise ValueError("The mel spectrogram is shorter than the desired sample duration.")
    
    # Randomly select the starting index for the sample
    start_idx = np.random.randint(0, max_start)
    
    # Extract the sample
    mel_sample = mel_spectrogram_db[:, start_idx:start_idx + n_samples + 1]
    
    return torch.tensor(mel_sample)#.to(device)

def min_max_normalize(mel_spectrogram):
    min_val = mel_spectrogram.min()
    max_val = mel_spectrogram.max()
    normalized_mel = (mel_spectrogram - min_val) / (max_val - min_val)
    return normalized_mel

def min_max_denormalize(normalized_mel, min_val=-80.0, max_val=0.0):
    """
    Denormalize the normalized mel-spectrogram.
    
    Args:
        normalized_mel (np.ndarray or torch.Tensor): Normalized mel-spectrogram.
        min_val (float): Minimum value for denormalization (default: -80.0).
        max_val (float): Maximum value for denormalization (default: 0.0).
        
    Returns:
        np.ndarray or torch.Tensor: Denormalized mel-spectrogram.
    """
    return normalized_mel * (max_val - min_val) + min_val

def normalize_audio(audio):
    """Normalize the audio (amplitude) to [-1, 1] range. This should make it possible to hear extremely quiet waveforms"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        return audio

def create_mel_spectogram_from_waveform(waveform, sample_rate):
    """Creates a mel-spectrogram in the power range using the input audio"""

    mel_spectogram = librosa.feature.melspectrogram(y=waveform.numpy().squeeze(),
                                        sr=sample_rate, 
                                            n_fft=2048, 
                                            hop_length=512, 
                                            win_length=None, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0,
                                    n_mels=128)
    
    return mel_spectogram

def reconstruct_waveform_from_mel_spectogram(mel_spectogram, sample_rate):
    reconstruction = librosa.feature.inverse.mel_to_audio(mel_spectogram, 
                                        sr=sample_rate, 
                                        n_fft=2048, 
                                        hop_length=512, 
                                        win_length=None, 
                                        window='hann', 
                                        center=True, 
                                        pad_mode='reflect', 
                                        power=2.0, 
                                        n_iter=32)
    
    return reconstruction

# # def convert_to_mel_spectrogram(audio_path, sample_rate, n_mels, fmax):
# #     y, sr = librosa.load(audio_path, sr=sample_rate)
# #     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
# #     S_dB = librosa.power_to_db(S, ref=np.max)
# #     return S_dB

def convert_to_mel_spectrogram(audio_path, sample_rate, n_mels, fmax, n_fft=2048, hop_length=512):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sample_rate)

    waveform = torch.tensor(y).unsqueeze(0)
    
    return convert_waveform_to_mel_spectrogram(waveform, sr, n_mels, fmax, n_fft, hop_length)

def convert_waveform_to_mel_spectrogram(waveform, sample_rate, n_mels, fmax, n_fft=2048, hop_length=512):
    # Load the audio file
    y, sr = waveform.numpy().squeeze(), sample_rate
    
    # Compute the Short-Time Fourier Transform (STFT) of the audio
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # Extract the magnitude (S) and phase (angle) of the STFT
    S, phase = np.abs(D), np.angle(D)
    
    # Convert the magnitude to a mel-spectrogram
    mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels, fmax=fmax)
    
    # Convert the mel-spectrogram to a decibel (dB) scale
    mel_S_dB = librosa.power_to_db(mel_S, ref=np.max)
    
    # Return the mel-spectrogram in dB, the phase information, and the sample rate
    return mel_S_dB, phase

def mel_spectrogram_to_audio(mel_spectrogram, phase, sr, n_fft=2048, hop_length=512, win_length=2048):
    # Convert the mel-spectrogram from dB back to power
    S = librosa.db_to_power(mel_spectrogram)
    
    # Convert the mel-spectrogram back to STFT magnitude
    S_inv = librosa.feature.inverse.mel_to_stft(S, sr=sr, n_fft=n_fft)
    
    # Combine the magnitude with the phase information
    D = S_inv * np.exp(1j * phase)
    
    # Use the inverse STFT to reconstruct the audio waveform
    y = librosa.istft(D, hop_length=hop_length, win_length=win_length)
    
    # Return the reconstructed audio waveform
    return y

# # def mel_spectrogram_to_audio(mel_spectrogram, sr, n_iter=32, n_fft=2048, hop_length=512, win_length=2048):
# #     # Invert the mel-spectrogram
# #     S = librosa.db_to_power(mel_spectrogram)
# #     S = librosa.feature.inverse.mel_to_stft(S, sr=sr, n_fft=n_fft)
# #     # Use Griffin-Lim algorithm for phase reconstruction
# #     y = librosa.griffinlim(S, n_iter=n_iter, hop_length=hop_length, win_length=win_length)
# #     return y

def plot_mel_spectrogram(mel_spectrogram, sr, display_plot=True, output_file=None):
    """Plots a mel-spectrogram and optionally save locally""" #after converting to the DB scale"""

    # Remove the extra dimension if present
    if len(mel_spectrogram.shape) == 3:
        mel_spectrogram = mel_spectrogram.squeeze(0)

    # Convert to the decibel scale to plot
    #mel_spectrogram_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Only attempt to convert to a NumPy array if it is not already a NumPy array
    #if not isinstance(mel_spectrogram, np.ndarray):
    mel_spectrogram_dB = mel_spectrogram.numpy() # The mel-spectrogram is already in dB scale
    #else:
    #    mel_spectrogram_dB = mel_spectrogram

    print("Mel-spectrogram shape: ", mel_spectrogram_dB.shape)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()

    # Save the mel spectrogram if an output location is provided
    if output_file != None:
        plt.savefig(output_file)
        print(f'Saved mel_spectrogram_db to {output_file}')


    if display_plot:
        plt.show()

    # free up memory after plotting and clear the image   
    plt.close()

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

# # def play_audio(sample):
# #     waveform = sample['waveform'].numpy().squeeze()
# #     sample_rate = sample['sample_rate']

# #     # Play audio using sounddevice
# #     sd.play(waveform, sample_rate)
# #     sd.wait()

def play_audio_from_waveform(waveform, sample_rate):
    # Play audio using sounddevice
    print("Sample rate: ", sample_rate)
    sd.play(waveform, sample_rate)
    sd.wait()




# Positional embeddings and timestep schedulers

class SinusoidalPositionEmbeddings(nn.Module):
    """Class to generate sinusoidal position embeddings for time or positional data.
       Used to encode time/position information."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim # Dimension of the embedding space.

    def forward(self, time):
        # Get the device of the input tensor to ensure computations are on the correct device.
        device = time.device
        half_dim = self.dim // 2 # Compute half the dimension to create sin and cos embeddings.
        embeddings = math.log(10000) / (half_dim - 1) # Scale factor for the frequencies of the sinusoidal embeddings.
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # Compute exponential terms for the positional encodings.
        embeddings = time[:, None] * embeddings[None, :] # Multiply the time input by the frequency terms to create embeddings.
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # Concatenate sine and cosine embeddings along the last dimension.
        return embeddings # Return the sinusoidal position embeddings.

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    #beta_start = 0.0001
    #beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):
    #beta_start = 0.0001
    #beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    #beta_start = 0.0001
    #beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start



# Up-blocks and down-blocks

class Block(nn.Module):
    """Defines a Block with a convolution, normalization, and activation."""

    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1) # 3x3 convolution
        self.norm = nn.GroupNorm(groups, dim_out) # Group normalization
        self.act = nn.SiLU() # SiLU activation

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        # Applies scale and shift if provided
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x) # Apply activation
        return x

class ResnetBlock(nn.Module):
    """Residual block using two Block layers and a residual connection.
       https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        
        # Conditional MLP for time embedding if provided
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups) # First Block layer
        self.block2 = Block(dim_out, dim_out, groups=groups) # Second Block layer
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity() # Adjusts input if needed

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        # Adds time embedding if it exists
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h) # Second Block
        return h + self.res_conv(x) # Residual connection

class ConvNextBlock(nn.Module):
    """Convolutional block inspired by ConvNeXt, with depthwise and residual connections.
       https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        
        # Conditional MLP for time embedding if provided
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim) # Depthwise convolution

        # Main convolutional network with normalization and activation
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity() # Adjust input if needed

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        # Adds time embedding if it exists
        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h) # Apply main convolutional network
        return h + self.res_conv(x) # Residual connection

def Upsample(dim):
    """Upsample function using a transpose convolution layer.
       This doubles the spatial dimensions (e.g., height and width) of the input."""
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    """Downsample function using a convolution layer.
       This halves the spatial dimensions (e.g., height and width) of the input."""
    return nn.Conv2d(dim, dim, 4, 2, 1)



# Residual connections and attention blocks

class Residual(nn.Module):
    """Residual class that wraps a function (fn) and adds the input (x) to its output.
       This is commonly used in residual networks (ResNets) to facilitate gradient flow."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    # Forward pass applies the function and adds the input to the result.
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    """The idea of pre-normalization (normalizing before applying the main function)
       helps to ensure that the inputs to the subsequent layers are standardised.
       Pre-normalisation is performed on the attention blocks when processing the residual blocks"""

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Attention(nn.Module):
    """Multi-head self-attention mechanism with scaling, for 2D inputs."""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5 # Scaling factor for query
        self.heads = heads # Number of attention heads
        hidden_dim = dim_head * heads # Dimension for multi-head projections
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False) # Compute Q, K, V
        self.to_out = nn.Conv2d(hidden_dim, dim, 1) # Output projection

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) # Split Q, K, V along the channel dimension
        
        # Rearrange the tensors q, k, v to separate the heads and flatten the spatial dimensions.
        # The input shape is (batch, heads * channels, height, width), and it is rearranged to 
        # (batch, heads, channels, height * width), where each head operates independently.
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale # Scale query for stable gradients

        # Compute attention scores using dot product and apply softmax
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach() # Normalize scores for stability
        attn = sim.softmax(dim=-1)

        # Compute weighted sum of values
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w) # Reshape back to original dimensions
        return self.to_out(out) # Output projection

class LinearAttention(nn.Module):
    """Linear attention mechanism to reduce complexity in self-attention calculations."""

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5 # Scaling factor for query
        self.heads = heads # Number of attention heads
        hidden_dim = dim_head * heads # Dimension for multi-head projections
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False) # Compute Q, K, V

        # Output projection with normalization
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1) # Split Q, K, V along the channel dimension
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2) # Apply softmax on queries along the spatial dimension
        k = k.softmax(dim=-1) # Apply softmax on keys along the spatial dimension

        q = q * self.scale # Scale queries
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v) # Compute context as K-V product

        # Multiply context with queries to get the final attention output
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q) 
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w) # Reshape back to original dimensions
        return self.to_out(out) # Output projection with normalization



# Models

class Unet(nn.Module):
    """A U-Net implementation with options for ResNet or ConvNeXt blocks and attention."""

    def __init__(
        self,
        dim,                         # Base dimension size
        init_dim=None,               # Initial convolution dimension
        out_dim=None,                # Output dimension (number of channels)
        dim_mults=(1, 2, 4, 8),      # Multipliers for the dimensions at each stage
        channels=3,                  # Number of input channels (e.g., RGB = 3)
        with_time_emb=True,          # Whether to include time embeddings
        resnet_block_groups=8,       # Groups for ResNet blocks (GroupNorm)
        use_convnext=True,           # Whether to use ConvNeXt blocks
        convnext_mult=2,             # Multiplier for ConvNeXt blocks
    ):
        super().__init__()

        # Initial convolution layer
        #self.channels = channels # determine dimensions

        init_dim = default(init_dim, dim // 3 * 2) # Default to 2/3 of the base dimension
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        # Compute dimensions for each stage
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Select block type (ResNet or ConvNeXt)
        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Time embedding layer (if enabled)
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        
        self.time_dim = time_dim

        self.downs = nn.ModuleList([]) # Define downsampling layers
        self.ups = nn.ModuleList([]) # Define upsampling layers
        num_resolutions = len(in_out)

        # Create downsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Bottleneck layer with attention and ConvNeXt/ResNet blocks
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Create upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Final convolution layer to match the output dimensions
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time): # Forward pass
        x = self.init_conv(x) # Initial convolution

        # Compute time embedding if enabled
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = [] # to store intermediate results for skip connections

        # Downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x) # Final convolution to get output

class ClassConditionalUnet(Unet):
    """Class conditioned unet with the class condition dimensions appended to the input at the start of the model and positional embeddings at each layer of the unet.
       Defines the U-Net model architecture for the diffusion process.
       Arguments:
       - dim: Base dimensionality (often corresponds to image size).
       - channels: Number of input/output channels (e.g., 3 for RGB images).
       - dim_mults: Multipliers for feature map dimensions at different U-Net layers."""
    
    def __init__(self,
        dim,                         # Base dimension size
        init_dim=None,               # Initial convolution dimension
        out_dim=None,                # Output dimension (number of channels)
        dim_mults=(1, 2, 4, 8),      # Multipliers for the dimensions at each stage
        channels=3,                  # Number of input channels (e.g., RGB = 3)
        with_time_emb=True,          # Whether to include time embeddings
        resnet_block_groups=8,       # Groups for ResNet blocks (GroupNorm)
        use_convnext=True,           # Whether to use ConvNeXt blocks
        convnext_mult=2,             # Multiplier for ConvNeXt blocks
        num_classes=None,
        emb_dim=0
    ):

        self.num_classes = num_classes
        emb_dim = emb_dim # Hyperparameter
        print("EMBEDDING DIMENSION: ", emb_dim)
        if self.num_classes is not None:
            #self.label_emb = nn.Embedding(num_classes, num_classes) #self.time_dim)

            # Reshape the initial layer to account for the additional conditional embeddings being concatenated to the input image
            #self.init_conv = nn.Conv2d(channels+num_classes, init_dim, 7, padding=3)
            channels+=emb_dim


        super().__init__(dim,        # Base dimension size
        init_dim=init_dim,               # Initial convolution dimension (Output from the initial layer, compared to channels being the input to this layer)
        out_dim=out_dim,#1                # Output dimension (number of channels)
        dim_mults=dim_mults,      # Multipliers for the dimensions at each stage
        channels=channels,                  # Number of input channels (e.g., RGB = 3)
        with_time_emb=with_time_emb,          # Whether to include time embeddings
        resnet_block_groups=resnet_block_groups,       # Groups for ResNet blocks (GroupNorm)
        use_convnext=use_convnext,           # Whether to use ConvNeXt blocks
        convnext_mult=convnext_mult,             # Multiplier for ConvNeXt blocks)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, emb_dim) #self.time_dim)


    def forward(self, x, time, label=None): # y is the label

        # Compute time embedding if enabled
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # Add the label embedding to the time embedding
        #if label is not None and t is not None:
        #    t += self.label_emb(label)

        # Concatenate class embedding to the input image
        if label is not None:
            label_emb = self.label_emb(label)
            label_emb = label_emb.unsqueeze(-1).unsqueeze(-1)           # Reshape to [batch_size, emb_dim, 1, 1]
            x = torch.cat((x, label_emb.expand(-1, -1, x.shape[2], x.shape[3])), dim=1)

        #print("X Starting Shape: ", x.shape)
        
        x = self.init_conv(x) # Initial convolution

        #print("X Shape: ", x.shape)

        h = [] # to store intermediate results for skip connections

        # Downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)

            #print("X Shape: ", x.shape)
            #print("Skip connection shape: ", x.shape) # Warning: If you get an odd number of dimensions, this process will break and not produce the same number of dimensions during upsampling!

            h.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Upsampling
        for block1, block2, attn, upsample in self.ups:
            skip_connection = h.pop()
            
            #print("X Shape: ", x.shape)
            #print("Skip connection shape: ", skip_connection.shape) # The spatial dimensions of the skip connctions should match the x at this block

            x = torch.cat((x, skip_connection), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x) # Final convolution to get output



# Forward diffusion process: generates a noisy version of the input image at a given timestep `t`.

def q_sample(x_start, t, noise=None): #, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod): The remaining variables are taken from the constants section below
    # If no noise is provided, generate random Gaussian noise.
    if noise is None:
        noise = torch.randn_like(x_start)

    # Extract the square root of the cumulative product of alphas for the given timestep `t`.
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    
    # Extract the square root of one minus the cumulative product of alphas for the given timestep `t`.
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    # Combine the scaled input `x_start` and scaled noise to produce the noisy sample.
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_spectrogram(x_start, t):#, reverse_transform):
  
  #print("Getting noisy image!!!")
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  #noisy_image = reverse_transform(x_noisy.squeeze())

  return x_noisy#noisy_image



# Reverse diffusion process

@torch.no_grad() # Disable gradient calculations for efficiency during inference.
def p_sample(model, x, t, t_index, labels=None): #, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance): The remaining variables are taken from the constants section below
    """
    Perform a single reverse diffusion step.
    
    Args:
        model: The diffusion model (noise predictor).
        x: Current noisy image tensor.
        t: Current timestep tensor.
        t_index: Index of the timestep.

    Returns:
        The denoised image tensor at the previous timestep.
    """
    
    # Extract relevant parameters for the current timestep.
    betas_t = extract(betas, t, x.shape) # Noise variance at this timestep.
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    ) # Precomputed term for efficiency.
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape) # Reciprocal alpha.

    # Equation 11 from the diffusion paper: Predict the mean of the posterior.
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, labels) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        # For the final step, return the mean directly as no further noise is added.
        return model_mean
    else:
        # Otherwise, sample from the posterior distribution.
        posterior_variance_t = extract(posterior_variance, t, x.shape) # Variance.
        noise = torch.randn_like(x) # Sample noise.
        # Add noise to the mean to simulate the reverse process (Algorithm 2, line 4).
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, labels=None, timesteps=0): # Algorithm 2 but save all images:
    """
    Perform the entire reverse diffusion process.
    
    Args:
        model: The diffusion model.
        shape: Shape of the desired output images (batch_size, channels, height, width).

    Returns:
        A list of sampled images at each timestep.
    """
    
    device = next(model.parameters()).device # Ensure computations are on the model's device.

    b = shape[0] # Batch size.
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)# Just random gaussian noise
    #img = torch.rand(shape, device=device) # Uniform noise in the range [0, 1]
    imgs = [] # to store images at each timestep for visualization or analysis.

    # Iterate over timesteps in reverse (from T to 0).
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, labels)
        #imgs.append(img.cpu().numpy()) # Store the current image (move to CPU for storage).
        
        # Save image at each 1/5 of the timesteps
        if i % (timesteps // 5) == 0:
            imgs.append(img.cpu())#.numpy())

    #print("LENGTH: ", len(imgs))
    return imgs#[img.cpu()]#.numpy()]

@torch.no_grad()
def sample(model, image_width, image_height, batch_size=16, channels=3, labels=None, timesteps=0):
    """
    Generate a batch of images by performing reverse diffusion.
    
    Args:
        model: The diffusion model.
        image_size: Height and width of the output images.
        batch_size: Number of images to generate.
        channels: Number of image channels (e.g., 3 for RGB).

    Returns:
        A list of generated images at each timestep.
    """
    
    return p_sample_loop(model, shape=(batch_size, channels, image_height, image_width), labels=labels, timesteps=timesteps)



# Loss functions

def p_losses(denoise_model, x_start, t, labels = None, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    # print("Noisy_X: ", x_noisy, "\n", x_noisy.shape)

    predicted_noise = denoise_model(x_noisy, t, labels)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss



import torch
import torchlibrosa
from torchlibrosa.augmentation import SpecAugmentation

# # # Define a simple convolutional model from PANNs
# # class PANNs_CNN14(torch.nn.Module):
# #     def __init__(self):
# #         super(PANNs_CNN14, self).__init__()
# #         # Using a simple model from PANNs for demonstration
# #         self.model = torch.hub.load('qiuqiangkong/panns_train', 'Cnn14', pretrained=True)
        
# #     def forward(self, x):
# #         # Extract features
# #         x = self.model(x)
# #         return x['embedding']

def extract_features(audio_embedding_model, mel_spectrograms):
    """Extract features from mel-spectrograms using PANNs model"""
    features = []
    for mel in mel_spectrograms:
        mel = mel.unsqueeze(0)  # Add batch dimension
        with torch.inference_mode():
            feature = audio_embedding_model(mel)
        features.append(feature.numpy())
    return np.array(features)

def calculate_fad(real_features, generated_features):
    """Calculate Frechet Audio Distance (FAD)"""
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_generated = np.mean(generated_features, axis=0)
    sigma_generated = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_generated
    covmean = sqrtm(sigma_real.dot(sigma_generated))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = np.sum(diff**2) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fad




# Optimisers

def get_optimiser(model, learning_rate, optimiser_type='adam'):
    if optimiser_type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimiser_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimiser_type == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimiser_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimiser_type}")



# Saving models and samples

def save_model(model, run_name="Audio class conditioned", epochs=0, upload_model=True):
    """Save locally, then upload to Weights and Biases as an artifact"""

    # Setup directory to store the model parameters
    filepath = "Saved Models"
    filename = f'{run_name}_{epochs}_epochs.pth'

    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    fullpath = os.path.join(filepath, filename)

    # Save the model state
    torch.save(model.state_dict(), fullpath)

    print("Model saved locally!")

    if upload_model:
        # Upload artifact to Weights and Biases
        at = wandb.Artifact("model", type="model", description="Class Conditioned Audio Diffusion Model.", metadata={"epoch": epochs})
        #at.add_dir(os.path.join("models", run_name))
        at.add_file(fullpath)
        wandb.log_artifact(at)

        print("Model uploaded!")

def save_samples(mel_spectrograms, upload_artifacts=True, play_audio=False):
    """Save locally, then upload to Weights and Biases as an artifact"""

    # Setup directory to store the generated audio mel-spectrograms
    folder1 = "Generated_Images"
    folder2 = "Class_Conditioned_Audio"
    folder = os.path.join(folder1, folder2)
    #filename = "Audio class conditioned_"+epochs+"_epochs.pth"

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    upload_mel_spectrogram_files = []
    upload_audio_files = []

    #print("Number of samples: ", len(mel_spectrograms))


    # Process mel-spectrograms from the batch back into audio
    for sample_idx, sample in enumerate(mel_spectrograms): 
        for i, mel_spectrogram_db in enumerate(sample): # Each sample is a list of mel-spectrograms
            #print("Sample: ", sample_idx, f"{i}_from_5_timesteps")
            #print(mel_spectrogram_db, mel_spectrogram_db.shape)

            #mel_spectrogram_db = sample[i] # In the db scale

            # Normalize the mel-spectrogram
            mel_spectrogram_db = min_max_normalize(mel_spectrogram_db)

            # Plot the mel-spectrogram and save it
            output_file = os.path.join(folder, f'output_audio_{sample_idx}_{i}_from_5_timesteps_{epochs}_epochs')
            output_mel_spectrogram = output_file + ".png"
            upload_mel_spectrogram_files.append(output_mel_spectrogram)
            plot_mel_spectrogram(mel_spectrogram_db, sample_rate, display_plot=False, output_file=output_mel_spectrogram)


            # sanitise the output to remove invalid values and replace them with 0
            #mel_spectrogram = np.nan_to_num(mel_spectrogram, nan=0.0, posinf=0.0, neginf=0.0)

            # De-normalise the mel-spectrogram
            mel_spectrogram_db = min_max_denormalize(mel_spectrogram_db).numpy().astype(np.float32)# must be numpy array to convert back to power scale
            
            # Convert mel-spectrogram back to power scale
            mel_spectrogram_power = librosa.db_to_power(mel_spectrogram_db, ref=1.0)
            
            # Play the audio reconstructed from the mel-spectrogram
            reconstruction = reconstruct_waveform_from_mel_spectogram(mel_spectrogram_power, sample_rate)
            reconstruction = normalize_audio(reconstruction)
            if play_audio:
                play_audio_from_waveform(reconstruction, sample_rate)

            # Compare to the original waveform
            #waveform = batch['waveform'][i].numpy().squeeze()
            #play_audio_from_waveform(waveform, batch['sample_rate'][i].item())

            # Save the new audio
            reconstruction = torch.tensor(reconstruction).unsqueeze(0) # Ensure the tensor is a 2D tensor to save
            output_audio_file = output_file + ".wav"
            upload_audio_files.append(output_audio_file)
            torchaudio.save(output_audio_file, reconstruction, sample_rate)
            print(f'Saved reconstructed audio to {output_audio_file}')


    if upload_artifacts:
        # log mel-spectrograms to wandb
        #wandb.log({f'{output_file}': wandb.Image(output_file)})
        wandb.log({"sampled_mel_spectrograms":     [wandb.Image(output_file) for output_file in upload_mel_spectrogram_files]})

        # log audio to wandb
        wandb.log({"sampled_audio":     [wandb.Audio(output_file, sample_rate=sample_rate) for output_file in upload_audio_files]})

        print("Samples uploaded!")



if __name__ == '__main__':



    # # # Hyperparameters

    epochs = 2000 # 20 epochs or above starts to produce 'reasonable' quality images but it takes longer time
    timesteps = [4000] # 1000
    beta_starts = [0.00001] # Noise scheduler start
    beta_ends = [0.0004] # Noise scheduler end
    learning_rates = [1e-4]#, 1e-3] # 1e-5, too low with a learning rate scheduler, 1e-3 too high
    warmup = 0.0
    dim_mults = [(1, 2, 4,)]# , (1, 2, 4, 8,) makes no noticable improvement] #(1, 2, 4,) # Model structure (number of layers and what size should each layer be)
    emb_dims = [1] # Cannot be set to 0 for no class conditioning#5]#, 10 makes no noticable improvement] # num_classes // 2 # The number of (conditional / class / genre) dimensions that are appended to the input image
    loss_type=["huber"] #["l1", "l2"]


    # TODO: Test different loss types, optimisers, learning rate schedules, noise schedules, weight initialisation techniques
    # Next step: learning rate scheduler, perhaps try just one genre of music, get overleaf for writeup

    # Constants

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # target_seconds = 5 should be (128, 216)
    image_size = 128  # Number of mel bands
    image_width = 216 # number of time windows
    channels = 1 # Single-channel for mel-spectrograms
    batch_size = 4 #8#32#128
    
    num_classes = 10

    torch.manual_seed(0) # use seed for reproducability
    sample_rate = 22050 # All files in the  dataset should have this sample_rate

    target_seconds=5 # Length of the mel-spectrograms to be generated and used as input to the model

    # Calculate total number of runs
    total_runs = len(list(product(timesteps, beta_starts, beta_ends, learning_rates, dim_mults, emb_dims, loss_type)))

    # Perform multiple runs sequencially
    for i, (timestep, beta_start, beta_end, learning_rate, architecture, emb_dim, loss_type) in enumerate(product(timesteps, beta_starts, beta_ends, learning_rates, dim_mults, emb_dims, loss_type)):
        print(f"Preparing to run {i+1}/{total_runs} with parameters: timestep={timestep}, beta_start={beta_start}, beta_end={beta_end}, learning_rate={learning_rate}, architecture={architecture}, emb_dim={emb_dim}, loss_type={loss_type}")


        # define beta schedule (a bad beta schedule that contains too much noise too early may make the model learn to produce blank mel-spectrograms)
        betas = linear_beta_schedule(timesteps=timestep, beta_start=beta_start, beta_end=beta_end) # cosine_beta_schedule(timesteps=timesteps, s=0.0001)

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # Get current date and time
        now = datetime.now()

        # Format time as hh:mm:ss
        time_str = now.strftime("%H_%M_%S")

        # Format date as dd/mm/yyyy
        date_str = now.strftime("%d_%m_%Y")

        print("Current Time:", time_str)
        print("Current Date:", date_str)


        run_name = f'Run_{i+1}_Date_{date_str}_{epochs}_epochs_{timestep}_timesteps_{beta_start}_beta_start_{beta_end}_beta_end_{learning_rate}_lr_{len(architecture)}_layers_{emb_dim}_emb_dim_{loss_type}_loss_blues_only' # MAKE SURE THIS NAME IS NOT TOO LONG!!!

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="Final-Year_university-Project",

            # Set the name of the run
            name = run_name,

            # track hyperparameters and run metadata
            config={
            "epochs": epochs,
            "batch_size": batch_size,
            "timesteps": timestep,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "learning_rate": learning_rate,
            "learning_rate_scheduler": True,
            "learning_rate_warmup": {warmup},
            "architecture": "CNN",
            "number_of_layers": len(architecture),
            "loss_type": loss_type,
            "dataset": "GTZAN",
            }
        )



        # Model Setup

        # Setup pre-processing
        # Done within the dataset itself

        # Import dataset and setup dataloader
        dataset = GTZANDataset(root_dir="GTZAN_Genre_Collection/genres", cache_dir="GTZAN_Genre_Collection/cached_mel_spectrograms", db_cache_path="GTZAN_Genre_Collection/cached_mel_spectrograms_db", target_seconds=target_seconds)#, transform=transform) # 5 seconds since this produces a round width on the mel-spectrograms that can be split many times
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, prefetch_factor=2) # 2 workers if using large epochs TODO: Augment the dataset

        # Save mel-spectrograms to disk if they are not present
        print("Saving mel spectrograms")

        dataset.save_mel_spectrograms()

        print("Done.")



        # Example: Make sure everything is setup correctly before continuing

        

        # Print the first item's genre and shape of the waveform
        print("Getting first item...")
        first_item = dataset[0]
        print(f"Genre: {first_item['genre']}, Mel-Spectrogram shape: {first_item['mel_spectrogram'].shape}, Sample Rate: {first_item['sample_rate']}")

        # # # Display the forward diffusion process for the mel-spectrogram
        # TODO: upload an example of the reverse diffusion process (after training) to weights and biases
        plot([get_noisy_spectrogram(first_item['mel_spectrogram'], torch.tensor([t])) for t in [0, timestep // 5, 2*timestep//5, 3*timestep//5, 4*timestep//5, timestep-1]])#[0, 50, 100, 150, 199]])


        # Plot the mel-spectrograms and save them
        folder1 = "Generated_Images"
        folder2 = "Class_Conditioned_Audio"
        folder = os.path.join(folder1, folder2)
        output_file = os.path.join(folder, f'forward_diffusion_{timestep}_timesteps_{beta_start}_beta_start_{beta_end}_beta_ends')
        output_mel_spectrogram = output_file + ".png"
        
        plt.savefig(output_mel_spectrogram, bbox_inches='tight') # Should remove the white padding
        print(f'Saved mel_spectrogram_db to {output_mel_spectrogram}')
        plt.show() # The image must be saved before plotting to prevent the canvas from being cleared

        # Upload the mel-spectrograms
        wandb.log({"forward_diffusion": wandb.Image(output_mel_spectrogram)})
        
        # # # TODO: Make a test that will compare the audio similarity between the reconstructed audio and the original waveform
        # # # Display the original mel-spectogram to compare
        # # plot_mel_spectrogram(first_item['mel_spectrogram'], first_item['sample_rate'])
        # # #plot_mel_spectrogram(create_mel_spectogram_from_waveform(first_item['waveform'], first_item['sample_rate']), first_item['sample_rate'])

        
        # # # De-normalise the mel-spectrogram
        # # mel_spectrogram_db = first_item['mel_spectrogram']# db scale, normalised mel-spectrogram
        # # mel_spectrogram_db = min_max_denormalize(mel_spectrogram_db).numpy().astype(np.float32)# must be numpy array to convert back to power scale
        
        # # # Convert mel-spectrogram back to power scale
        # # mel_spectrogram_power = librosa.db_to_power(mel_spectrogram_db, ref=1.0)
        
        # # # Play the audio reconstructed from the mel-spectrogram
        # # reconstruction = reconstruct_waveform_from_mel_spectogram(mel_spectrogram_power, sample_rate)
        # # play_audio_from_waveform(normalize_audio(reconstruction), sample_rate)


        # # # Test retrieving a batch from the dataset
        # # batch = next(iter(dataloader))
        # # #print("BATCH: ", batch) # [0] is images, [1] is labels
        # # print(batch.keys())
        # # #print("Labels from this batch: ", batch[1])#batch["label"])


        # Process mel-spectrograms from the batch back into audio
        # # for i, mel_spectrogram in enumerate(batch['mel_spectrogram']):

        # #     # De-normalise the mel-spectrogram
        # #     mel_spectrogram_db = min_max_denormalize(mel_spectrogram).numpy().astype(np.float32)# must be numpy array to convert back to power scale
            
        # #     # Convert mel-spectrogram back to power scale
        # #     mel_spectrogram_power = librosa.db_to_power(mel_spectrogram_db, ref=1.0)
            
        # #     # Play the audio reconstructed from the mel-spectrogram
        # #     reconstruction = reconstruct_waveform_from_mel_spectogram(mel_spectrogram_power, sample_rate)
        # #     play_audio_from_waveform(normalize_audio(reconstruction), batch['sample_rate'][i].item())

        # #     # Compare to the original waveform
        # #     #waveform = batch['waveform'][i].numpy().squeeze()
        # #     #play_audio_from_waveform(waveform, batch['sample_rate'][i].item())



        # Load model
        model = ClassConditionalUnet(
        dim=image_size,
        channels=channels,
        out_dim=channels,
        dim_mults=architecture, #(1, 2, 4,),
        num_classes=num_classes,#len(dataset.caption_to_class)
        emb_dim=emb_dim # num_classes // 2
        #init_dim=1+
        )
        model.to(device)

        # Define the optimizer for training the model.
        # - model.parameters(): Parameters of the U-Net model to optimize.
        # - lr=1e-3: Learning rate for the optimizer.
        optimizer = Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-5) # Weight decay is L2 regularisation (a term added to the loss function being optimised)


        # Learning rate scheduler
        num_steps_per_epoch = len(dataloader)
        total_training_steps = num_steps_per_epoch * epochs
        warmup_ratio = warmup#0.1 # 10% of total training steps
        warmup_steps = int(total_training_steps * warmup_ratio)

        # Learning rate scheduler
        lr_scheduler = get_cosine_schedule_with_warmup( # Used to vary the learning rate at each step to better tune the training process
           optimizer=optimizer,
           num_warmup_steps=warmup_steps,
           num_training_steps=total_training_steps,
        )

        # # import tensorflow_hub as hub

        # Load pre-trained audio embedding model
        #audio_embedding_model = hub.load('https://tfhub.dev/google/vggish/1')#inception_v3(pretrained=True) # has default size of 96x64, which does not match the mel-spectrograms for this project
        #audio_embedding_model.eval()
        # Load the pre-trained PANNs model
        #audio_embedding_model = PANNs_CNN14()
        #audio_embedding_model.eval()

        # # # Load real and generated mel-spectrograms
        # # real_mels = batch = next(iter(dataloader))  # Replace with your real mel-spectrograms
        # # labels = batch['genre']#.to(device)
        # # generated_mels = sample(model, image_height=image_size, image_width=image_width, batch_size=batch_size, channels=channels, labels=labels)  # Replace with your generated mel-spectrograms

        # # # Extract features
        # # real_features = extract_features(real_mels)
        # # generated_features = extract_features(generated_mels)

        # # # Calculate FAD
        # # fad_score = calculate_fad(real_features, generated_features)
        # # print(f'Frechet Audio Distance (FAD): {fad_score}')



        # Model Training

        # Set the model to train mode (Ensures layers such as dropout and batch normalisation work correctly)
        model.train()

        #all_losses = []

        for epoch in range(epochs):
            #for step, batch in enumerate(dataloader):
            running_loss = 0.0
            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", total=len(dataloader))):
                optimizer.zero_grad()

                # Get batch size and move data to device
                batch_size = batch['mel_spectrogram'].shape[0]
                batch_mels = batch['mel_spectrogram'].to(device)

                # Add a channel dimension to mel-spectrograms: [batch_size, 1, 128, 1292]
                if batch_mels.dim() == 3:
                    batch_mels = batch_mels.unsqueeze(1) # The un-processed mel-spectrograms have shape [batch_size, 128, 1292], without an explicit channel dimension used within images


                #batch_labels = torch.tensor(batch['genre'], dtype=torch.long).to(device) # Sample t uniformly for every example in the batch
                batch_labels = batch['genre'].to(device) # Add labels to the training batch

                #print("Batch keys: ", batch.keys())
                #print("Labels: ", batch_labels.shape)

                #batch_labels = batch[1].to(device) # Add labels to the training batch

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, timestep, (batch_size,), device=device).long()

                loss = p_losses(model, batch_mels, t, batch_labels, loss_type=loss_type)

                # Log the loss from the batch
                #all_losses.append(loss.item()) # Append loss to the list
                running_loss+=loss.item()
                
                # log metrics to wandb
                wandb.log({f'{loss_type}_loss': loss})
                wandb.log({f'learning_rate': lr_scheduler.get_last_lr()[0]})

                # Update the model weights and optimiser
                loss.backward()

                # Clip the calculated gradients to prevent the exploding gradients problem
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Uncomment for the learning rate scheduler to be active
                #lr_scheduler.step()

            average_loss = running_loss / len(dataloader)
            print("Average loss for epoch: ", epoch + 1, " = ", average_loss)
            wandb.log({f'average_{loss_type}_loss': average_loss})

            # Load real and generated mel-spectrograms
            #real_mels = batch = next(iter(dataloader))  # Replace with your real mel-spectrograms
            #labels = batch['genre']#.to(device)
            #generated_mels = sample(model, image_height=image_size, image_width=image_width, batch_size=batch_size, channels=channels, labels=batch_labels)  # Replace with your generated mel-spectrograms
            #generated_mel_spectrograms = generated_mels[0].squeeze(1)


            # Extract features
            #real_features = extract_features(audio_embedding_model, batch['mel_spectrogram'])
            #generated_features = extract_features(audio_embedding_model, generated_mel_spectrograms)

            # Calculate FAD
            #fad_score = calculate_fad(real_features, generated_features)
            #print(f'Frechet Audio Distance (FAD): {fad_score}')

            # log metrics to wandb
            #wandb.log({"fad": fad_score})


        # # # Plotting the loss
        # # plt.figure(figsize=(10, 5))
        # # plt.plot(all_losses, label='Loss')
        # # plt.xlabel('Batch number')
        # # plt.ylabel('Loss')
        # # plt.title('Training Loss Over Time')
        # # plt.legend()
        # # plt.show()



        # Sample new audio using the trained model

        model.eval()

        # Generate labels, 10 of each class
        class_indices = []
        for i in range(1): # Only sampling blues #(10):
            class_indices.extend([i] * 1)

        # Convert the list to a tensor
        labels = torch.tensor(class_indices, device=device)

        # Check the labels
        print(labels)

        # Save the trained model weights
        save_model(model, run_name=run_name, epochs=epochs)

        # # # Load the model state

        # # # Load the state dictionary into the model
        # # model.load_state_dict(torch.load(fullpath))
        # # model.eval() # Set the model to evaluation mode

        # # print("Model loaded!")

        # randomly generate 10 mel_spectrograms
        samples = sample(model, image_height=image_size, image_width=image_width, batch_size=len(class_indices), channels=channels, labels=labels, timesteps=timestep)# These samples will be un-normalised

        #mel_spectrograms = samples[0].squeeze(1) # if there is only 1 image per sample
        mel_spectrograms = [sample_imgs.squeeze(1) for sample_imgs in samples] # Each sample contains a list of 5 images demonstrating the reverse diffusion process


        save_samples(mel_spectrograms)

        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()