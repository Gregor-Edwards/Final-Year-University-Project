import diffusion_models
import datasets
import torch
import os
from tqdm.auto import tqdm
from diffusers import DDPMScheduler
from PIL import Image
import torchaudio




timesteps = 4000 # 1000
# noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps) # 1000 timesteps
noise_scheduler = DDPMScheduler(
    num_train_timesteps=4000,
    beta_start=0.00005,  # Adjusted for 4000 timesteps
    beta_end=0.01,       # Adjusted for 4000 timesteps
    beta_schedule="linear"
) # 4000 timesteps

resolution = (256, 256) # (64, 64)
num_classes = 10
#max_slice_position = 18
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0
torch.manual_seed(seed) # use seed for reproducability

model = diffusion_models.ClassConditionedAudioUnet2D(sample_size=resolution, num_classes=num_classes).to(device)#, max_position=max_slice_position).to(device)
dataset = datasets.GTZANDataset(resolution=resolution, root_dir="GTZAN_Genre_Collection/genres", spectrogram_dir="GTZAN_Genre_Collection/slices")


# Setup directory to store the model parameters
filepath = "Saved Models"
# filename = 'Date_02_03_2025_100_epochs_1000_timesteps_0.95_0.0001_lr_class_conditioned_GTZAN_100_epochs.pth' #f'{run_name}_{epochs}_epochs.pth' # Test 1000 timesteps 64x64
#filename = 'Date_03_03_2025_100_epochs_4000_timesteps_0.95_0.0001_lr_class_conditioned_GTZAN_100_epochs.pth' #f'{run_name}_{epochs}_epochs.pth' # Test 4000 timesteps 256x256
# filename = 'Date_05_03_2025_16_18_22_100_epochs_4000_timesteps_class__position_embeddings_GTZAN_100_epochs_64_x_res_position_embed.pth' # Positional embedding part 1
# filename = 'Date_06_03_2025_11_25_30_100_epochs_4000_timesteps_class__position_embeddings_GTZAN_100_epochs_64_x_res_position_embed.pth' # Positional embedding part 2
filename = 'Date_06_03_2025_23_31_27_200_epochs_4000_timesteps_class_embeddings_GTZAN_200_epochs_64_x_res.pth' # 4000 timesteps more epochs

# Create the directory if it doesn't exist
os.makedirs(filepath, exist_ok=True)
fullpath = os.path.join(filepath, filename)

# Load the state dictionary into the model
model.load_state_dict(torch.load(fullpath))
model.eval() # Set the model to evaluation mode



# Sample new audio using the trained model

# Generate labels: 1 sample for each class (len(dataset.genres) total classes)
class_indices = [i for i in range(10)] # First n classes # len(dataset.genres))]  # One sample per class

# Generate positional indices
#slice_positions = torch.tensor([0] * len(class_indices), device=device)  # Shape: (batch_size,)

# Batch size is equal to the number of classes
batch_size = len(class_indices)
shape = (batch_size, 1, resolution[0], resolution[1])  # Shape of spectrogram: (batch_size, channels, height, width/length)

# Convert class indices into a tensor of shape (batch_size,)
class_labels = torch.tensor(class_indices, device=device)  # Shape: (batch_size,)

# Check the labels
print("Class labels: ", class_labels)

# Initialize noise for the batch
images = torch.randn(shape, device=device)



# Initialize noise
#images = torch.randn(shape, device=device)

for step in tqdm(reversed(range(0, timesteps)), desc="Sampling timestep", unit="step"):

    # Create a tensor filled with the current timestep for the batch
    batch_step = torch.full((shape[0],), step, device=device, dtype=torch.long)
    
    model_output = model(images, batch_step, class_labels=class_labels).detach()#, slice_positions=slice_positions).detach() # Detaching prevents memory build up at each step

    # Perform the scheduler step and update images in-place
    images.copy_(noise_scheduler.step(
        model_output=model_output,
        timestep=step,
        sample=images,
        generator=None,
    )["prev_sample"].detach())

images = (images / 2 + 0.5).clamp(0, 1)
images = images.cpu().permute(0, 2, 3, 1).numpy()
images = (images * 255).round().astype("uint8")
images = list(
    map(lambda _: Image.fromarray(_[:, :, 0]), images)
    if images.shape[3] == 1
    else map(lambda _: Image.fromarray(_, mode="RGB").convert("L"), images)
)



# Setup directory to store the generated audio mel-spectrograms
folder1 = "Generated_Images"
folder2 = "Class_Conditioned_Audio" #"Unconditional_Audio"
folder = os.path.join(folder1, folder2)
#filename = "Audio class conditioned_"+epochs+"_epochs.pth"

# Create the directory if it doesn't exist
os.makedirs(folder, exist_ok=True)

# Save final images
for idx, image in enumerate(images):
    output_file = os.path.join(folder, f'TEST_IMAGE_{idx}_seed_{seed}')
    output_mel_spectrogram = output_file + ".png"

    # Save the generated spectrogram image
    image.save(output_mel_spectrogram)
    print(f"Generated spectrogram {idx} saved!")

    # Load the generated spectrograms and convert to audio
    generated_spectrogram_file = output_mel_spectrogram#os.path.join(folder, "output_audio_0_100_epochs_64_x_res_640_y_res.png")
    image = Image.open(generated_spectrogram_file)
    audio = dataset.mel_spectrogram_to_audio(image)
    audio_tensor = torch.tensor(audio).unsqueeze(0) # Add channel dimension due to the mono audio output (duplicate the channel so that the audio can be played)

    # Save the audio file
    output_audio_path = generated_spectrogram_file.replace(".png", ".wav")
    torchaudio.save(output_audio_path, audio_tensor, dataset.sample_rate)
    print(f"Audio {idx} saved!")