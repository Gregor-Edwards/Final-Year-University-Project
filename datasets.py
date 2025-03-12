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
            ToTensor(), # range from 0 to 1
            Normalize([0.5], [0.5]), # range from -1 to 1
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
        self.genres = os.listdir(root_dir)
        self.files = [] # Stores the file names of each waveform sample

        # Mel-spectrogram time-slice conversion
        self.resolution = resolution # (64, 64) # Resolution of each time slice (number of frequency bins, number of time steps / frames)
        self.hop_length = 1024 #512
        self.sample_rate = 22050
        self.n_fft = 2048
        self.top_db = 80
        self.n_iter = 32 #100 # Number of iterations for Griffin Linn mel inversion
        self.spectrogram_slices = [] # Stores each time slice of the converted mel-spectrograms. These are the actual items used in the diffusion process

        # Add each file to the dataset
        print("GENRES: ", self.genres) # GENRES:  ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
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
            #number_of_time_slices = number_of_samples // slice_size
            step_size = slice_size // 2  # 50% overlap
            number_of_time_slices = (number_of_samples - slice_size) // step_size + 1

            for slice in range(number_of_time_slices):
                spectrogram_path = os.path.join(self.spectrogram_dir, f"{os.path.basename(audio_path)}_{slice}.png")
                start_idx = slice * step_size
                end_idx = start_idx + slice_size


                # Create directories if they don't exist
                if not os.path.exists(os.path.dirname(spectrogram_path)):
                    print("Directory does not exist! Making directory...")
                    os.makedirs(os.path.dirname(spectrogram_path))

                # Skip if files already exist
                if os.path.exists(spectrogram_path):
                    #print(f"File at {spectrogram_path} already exists!")
                    continue

                # Convert time-slice to spectrogram image
                audio_slice = waveform[start_idx : end_idx]#.numpy()
                image = self.audio_to_mel_spectrogram(audio_slice)

                #print("Time-slice: ", slice, "Image-size: ", image.width, image.height)
                
                # Remove any silent time slices
                if all(np.frombuffer(image.tobytes(), dtype=np.uint8) == 255):
                    print(f"File: {audio_path} slice {slice} is silent!")
                    continue

                # Save the time slice image
                #print("Saving image to: ", spectrogram_path)
                image.save(spectrogram_path, format="PNG")
        
        # If there are currently no samples in the self.spectrogram_slices parameter, then add them here
        if len(self.spectrogram_slices) == 0:
            print("Slices have not yet been added to the dataset! Adding them now...")
            if os.path.exists(self.spectrogram_dir):
                for file in os.listdir(self.spectrogram_dir):
                    if file.endswith('.png'):
                        self.spectrogram_slices.append(os.path.join(self.spectrogram_dir, file))
            print("Done.")


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

        # Extract the file name (e.g., "blues.00000.au_0.png") regardless of folder structure
        file_name = os.path.basename(image_path)
        genre, file_number, slice_number = file_name.split('.')[0], file_name.split('.')[1], file_name.split('_')[1].split('.')[0]
        
        class_label = self.genre_to_class[genre]  # Map genre to class label

        # Load saved mel-spectrogram slice if it exists
        if os.path.exists(image_path):
            #print("Cached files found! Loading...")
            image = Image.open(image_path)

        # Transform the spectrogram slices correctly
        image = augmentations(image)#image.convert("RGB"))

        # Convert the slice number to a tensor
        slice_number = torch.tensor(int(slice_number))

        sample = {'image': image, 'label': class_label, 'position': slice_number, 'sample_rate': sample_rate}

        return sample
    

if __name__ == '__main__':
    # Testing
    import tests

    # Initialise dataset
    root_dir = "GTZAN_Genre_Collection/genres"
    spectrogram_dir = "GTZAN_Genre_Collection/test_slices"
    resolution = (256, 256)  # Resolution of mel-spectrogram slices
    dataset = GTZANDataset(resolution, root_dir, spectrogram_dir)
    dataset.hop_length = 512 #1024
    dataset.n_fft = 2048
    dataset.top_db = 80
    dataset.n_iter = 100 # Number of iterations for Griffin Linn mel inversion

    # TODO: Test the dataset setup
    # TODO: Assert that the folder does not already exist
    tests.test_dataset_setup(dataset)

    tests.test_mel_spectrogram_conversion(dataset)
