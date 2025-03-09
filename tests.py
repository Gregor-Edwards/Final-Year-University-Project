# Forward diffusion
import matplotlib.pyplot as plt
import torch

# FAD score

import torchaudio
import numpy as np
from scipy.linalg import sqrtm
import sys
import os

# Add the PANNs folder to the system path
sys.path.append(os.path.abspath("PANNs/audioset_tagging_cnn-master/pytorch"))

# Import Cnn14 directly from models.py within the downloaded PANNs git repo
from models import Cnn14

# Step 1: Load the PANNs Pre-trained Model
def load_panns_model():
    """
    Load the pre-trained PANNs (Cnn14) model for audio feature extraction.
    """
    model = Cnn14(
        sample_rate=32000,     # The sample rate used during training
        window_size=1024,      # The FFT window size
        hop_size=320,          # The hop size (step size between FFT windows)
        mel_bins=64,           # Number of mel bins in the spectrogram
        fmin=50,               # Minimum frequency for the mel scale
        fmax=14000,            # Maximum frequency for the mel scale
        classes_num=527        # Number of output classes in the PANNs model
    )

    # Load the checkpoint
    checkpoint = torch.load(os.path.abspath("PANNs/Cnn14_mAP=0.431.pth"), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()  # Set the model to evaluation mode
    return model

# Step 2: Preprocess Audio
def preprocess_audio(waveform, sample_rate, target_sample_rate=32000):
    """
    Resample waveform to the target sample rate, and ensure mono audio. Waveform should be a torch tensor.
    """
    #print("Part 1: ", waveform.shape)
    
    # Resample if the sample rate is not the target sample rate
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)

    #print("Part 2: ", waveform.shape)
    
    if waveform.ndim == 2:  # Stereo audio
        #print("Stereo audio detected! Mixing down to mono...")
        waveform = np.mean(waveform, axis=0)  # Mixdown to mono

    #print("Part 3: ", waveform.shape)
    
    return waveform

# Step 3: Extract Embeddings Using PANNs
def extract_panns_embeddings(waveform, model):
    """
    Extract audio feature embeddings using the PANNs model.
    """
    with torch.no_grad():
        #print("Waveform shape: ", waveform.shape)
        # Pass the waveform through the PANNs model to get embeddings
        embeddings = model(waveform.unsqueeze(0))['embedding']
    return embeddings.numpy()

# Step 4: Compute FAD Score
def compute_fad(embeddings1, embeddings2):
    """
    Calculate the Fréchet Audio Distance (FAD) between two sets of embeddings.
    """

    # Duplicate the single embedding for each sample to simulate at least two samples per distribution. This is because Covariance computation requires at least two samples (rows) for statistical variability.
    if embeddings1.shape[0] == 1:
        embeddings1 = np.vstack([embeddings1, embeddings1])
    if embeddings2.shape[0] == 1:
        embeddings2 = np.vstack([embeddings2, embeddings2])


    mu1, sigma1 = np.mean(embeddings1, axis=0), np.cov(embeddings1, rowvar=False)
    mu2, sigma2 = np.mean(embeddings2, axis=0), np.cov(embeddings2, rowvar=False)
    
    # Compute the difference between the means
    diff = mu1 - mu2
    
    # Compute the square root of the product of the covariance matrices
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # Handle numerical instability
    
    # Apply the FAD formula
    fad = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fad

# Step 5: Main Function to Calculate FAD
def calculate_fad_between_samples(waveform_1, waveform_2, sample_rate):
    """
    Calculate the FAD score between two audio samples of a given sample rate.
    """
    # Load the PANNs model
    panns_model = load_panns_model()
    
    # Preprocess the audio files
    waveform1 = preprocess_audio(waveform_1, sample_rate=sample_rate)
    waveform2 = preprocess_audio(waveform_2, sample_rate=sample_rate)
    
    # Extract embeddings for both audio files
    embeddings1 = extract_panns_embeddings(waveform1, panns_model)
    embeddings2 = extract_panns_embeddings(waveform2, panns_model)

    print("Embeddings 1: ", embeddings1.shape)
    print("Embeddings 2: ", embeddings2.shape)
    
    # Compute the FAD score
    fad_score = compute_fad(embeddings1, embeddings2)
    return fad_score



# Noise scheduler tests

def test_forward_diffusion(dataset, device, timesteps, noise_scheduler):
    print("Testing forward diffusion process!")

    clean_sample = dataset[0]["image"]  # Gets the first file from the dataset
    clean_sample = clean_sample.unsqueeze(0)  # Add batch dimension (1, channels, height, width)
    clean_sample = clean_sample.to(device)

    # Number of timesteps to visualize (e.g., 5 intermediate steps)
    num_visual_steps = 10
    timesteps_to_visualize = torch.linspace(0, timesteps - 1, num_visual_steps).long().to(device)

    # Visualize noise addition at each step
    fig, axes = plt.subplots(1, num_visual_steps, figsize=(15, 5))
    for i, t in enumerate(timesteps_to_visualize):
        # Add noise to the clean sample at timestep `t`
        noisy_sample = noise_scheduler.add_noise(clean_sample, torch.randn_like(clean_sample), t)
        
        # Convert noisy sample to numpy for visualization
        noisy_sample_np = noisy_sample.squeeze(0).cpu().numpy()  # Remove batch dimension

        # Plot the noisy sample
        axes[i].imshow(noisy_sample_np[0], cmap="magma")  # Assuming single-channel audio
        axes[i].set_title(f"Timestep {t.item()}")
        axes[i].axis("off")

    plt.show()



# Dataset tests

def test_dataset_setup(dataset):
    # Assert that the folder does not already exist
    folder = dataset.spectrogram_dir
    if os.path.exists(folder):
        print("The folder already exists before saving slices.")

    # Attempt to save the slices
    dataset.save_mel_spectrograms()

    # Assert that the folder now exists
    assert os.path.exists(folder), "The folder does not exist after saving slices."

    # Assert that the first slice can be displayed
    slices = os.listdir(folder)
    assert len(slices) > 0, "No slices were saved in the folder."
    first_slice_path = os.path.join(folder, slices[0])
    assert os.path.isfile(first_slice_path), "The first slice is not a valid file."
    print(f"First slice saved at: {first_slice_path}")

def test_mel_spectrogram_conversion(dataset):
    import matplotlib.pyplot as plt
    import torchaudio
    import librosa.display
    from IPython.display import Audio, display
    import sounddevice as sd

    # Get the first file in the dataset
    if len(dataset.files) == 0:
        raise ValueError("No audio files found in the dataset!")
    
    audio_path, genre = dataset.files[0]  # Retrieve the first audio file and its genre
    print(f"Testing file: {audio_path}, Genre: {genre}")

    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Loaded waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

    # Ensure the waveform is mono
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)  # Convert to mono by averaging channels

    # Convert the waveform to a mel-spectrogram
    waveform_numpy = waveform.numpy()
    mel_spectrogram = dataset.audio_to_mel_spectrogram(waveform_numpy)
    print(f"Generated mel-spectrogram of size: {mel_spectrogram.size}")

    # Display the mel-spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, cmap='inferno', aspect='auto')
    plt.title(f"Mel-Spectrogram for File: {audio_path}")
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(label='Intensity')
    plt.show()

    # Convert the mel-spectrogram back to audio
    reconstructed_audio_numpy = dataset.mel_spectrogram_to_audio(mel_spectrogram)
    print(f"Reconstructed audio length: {len(reconstructed_audio_numpy)}, shape {reconstructed_audio_numpy.shape} Sample rate: {dataset.sample_rate}")

    # Compare the original and reconstructed audio
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    plt.title("Original Audio")
    plt.plot(waveform_numpy)
    plt.subplot(2, 1, 2)
    plt.title("Reconstructed Audio")
    plt.plot(reconstructed_audio_numpy)
    plt.tight_layout()
    plt.show()

    # Both are now torch tensors
    reconstructed_audio = torch.tensor(reconstructed_audio_numpy)
    print(waveform)
    print(reconstructed_audio)

    # Calculate the FAD score between the original and reconstructed audio
    fad_score_1 = calculate_fad_between_samples(waveform, waveform, sample_rate)
    print(f"Fréchet Audio Distance (FAD) Score between the original waveform and itself: {fad_score_1}") # This should be 0.0 and deterministic

    fad_score_2 = calculate_fad_between_samples(waveform, reconstructed_audio, sample_rate)
    print(f"Fréchet Audio Distance (FAD) Score between the original waveform and the reconstructed waveform: {fad_score_2}") # This will vary slightly due to the Griffin Lim algorithm used to convert to a mel-spectrogram is non-deterministic

    reconstructed_audio = reconstructed_audio / torch.max(torch.abs(reconstructed_audio))
    fad_score_3 = calculate_fad_between_samples(waveform, reconstructed_audio, sample_rate)
    print(f"Fréchet Audio Distance (FAD) Score between the original waveform and the normalised reconstructed waveform: {fad_score_3}")


    # Listen to original and reconstructed audio (use IPython.display for playback in notebooks)
    print("Playing Original Audio:")
    display(Audio(waveform, rate=dataset.sample_rate))
    sd.play(waveform, dataset.sample_rate)
    sd.wait()
    print("Playing Reconstructed Audio:")
    display(Audio(reconstructed_audio, rate=dataset.sample_rate))
    sd.play(reconstructed_audio, dataset.sample_rate)
    sd.wait()

