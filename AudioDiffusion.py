import datasets
import models
import torch.nn.functional as F # to access various functions for neural networks, like activation functions and loss calculations
from tqdm.auto import tqdm # for showing progress bar
import os
from PIL import Image

# Play generated audio
import torchaudio
import sounddevice as sd

# Dataloader
import torch
from torch.utils.data import DataLoader

# Optimiser
from torch.optim import Adam

# Learning rate scheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

# Forward diffusion
from diffusers import DDPMScheduler

# Initialise forward diffusion parameters

timesteps = 1000 # 4000
noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
# # beta_start = 0.00001 #, 0.00002]#[0.000005] # Noise scheduler start
# # beta_end = 0.001 #[0.0004, 0.0006, 0.0008, 0.0003] # Noise scheduler end

# # # define beta schedule (a bad beta schedule that contains too much noise too early may make the model learn to produce blank mel-spectrograms)
# # betas = models.linear_beta_schedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end) # cosine_beta_schedule(timesteps=timesteps, s=0.0001)

# # # define alphas
# # alphas = 1. - betas
# # alphas_cumprod = torch.cumprod(alphas, axis=0)
# # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# # sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# # # calculations for diffusion q(x_t | x_{t-1}) and others
# # sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# # sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# # # calculations for posterior q(x_{t-1} | x_t, x_0)
# # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


# Initialise dataset and dataloader

resolution = (64, 64) #dataset[0]["image"].height, dataset[0]["image"].width
batch_size = 16

dataset = datasets.GTZANDataset(resolution=resolution, root_dir="GTZAN_Genre_Collection/genres", spectrogram_dir="GTZAN_Genre_Collection/slices")

# Save mel-spectrograms to disk if they are not present
print("Saving mel spectrograms")

dataset.save_mel_spectrograms()

print("Done.")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)



# Initialise model

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0) # use seed for reproducability
# sample_rate = 22050 # All files in the  dataset should have this sample_rate

channels = 1 # Mono audio
num_classes = len(dataset.genres)

epochs = 100#2000#2000 # 20 epochs or above starts to produce 'reasonable' quality images but it takes longer time
learning_rate = 1e-4#, 1e-3] # 1e-5, too low with a learning rate scheduler, 1e-3 too high
#warmup = 0.0
#dim_mults = (1, 2, 4,) #, (1, 2, 4, 8,)] # , (1, 2, 4, 8,) makes no noticable improvement] #(1, 2, 4,) # Model structure (number of layers and what size should each layer be)
#emb_dim = 10 #[1, 2, 5, 10, 36] # Cannot be set to 0 for no class conditioning#5]#, 10 makes no noticable improvement] # num_classes // 2 # The number of (conditional / class / genre) dimensions that are appended to the input image
#loss_type= "huber" #["l1", "l2"]

# # model = models.Unet(
# # dim=resolution, # Image size
# # channels=channels,
# # out_dim=channels,
# # dim_mults=dim_mults, #(1, 2, 4,),
# # #num_classes=num_classes,#len(dataset.caption_to_class)
# # #emb_dim=emb_dim # num_classes // 2
# # )
# # model.to(device)

#model = models.SimplifiedUNet2D(resolution).to(device)
model = models.AudioUnet2D(resolution).to(device)





# Define the optimizer for training the model.
# model.parameters(): Parameters of the U-Net model to optimize.
beta_start = 0.95
beta_end = 0.999
weight_decay = 1e-6
epsilon = 1e-08
optimizer = Adam(model.parameters(), lr=learning_rate, betas=(beta_start, beta_end), weight_decay=weight_decay, eps=epsilon,) # Weight decay is L2 regularisation (a term added to the loss function being optimised)


# Learning rate scheduler
warmup_steps = 500
num_steps_per_epoch = len(dataloader)
total_training_steps = num_steps_per_epoch * epochs

lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps,) # Used to vary the learning rate at each step to better tune the training process




# Train the diffusion model
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    #for step, batch in enumerate(dataloader):
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", total=len(dataloader))):

        # Get batch size and move data to device
        batch_size = batch['image'].shape[0]
        batch_mel_spectrograms = batch['image'].to(device)

        #print("Size of the batch images: ", batch_mel_spectrograms.shape)

        # Noise to be added to the images
        noise = torch.randn(batch_mel_spectrograms.shape).to(device)

        # Sample a random timestep for each image
        timestep = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        
        # Forward diffusion process
        noisy_mel_spectrograms = noise_scheduler.add_noise(batch_mel_spectrograms, noise, timestep) # Adds noise depending on the timestep

        # Backward diffusion process
        noise_prediction = model(noisy_mel_spectrograms, timestep)
        loss = F.mse_loss(noise_prediction, noise)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Accumulate the loss
        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    print("Average loss for epoch: ", epoch + 1, " = ", average_loss)






# Save and load the trained model

def save_model(model, fullpath, upload_model=True):
    """Save locally, then upload to Weights and Biases as an artifact"""

    # Save the model state
    torch.save(model.state_dict(), fullpath)

    print("Model saved locally!")

    # # if upload_model:
    # #     # Upload artifact to Weights and Biases
    # #     at = wandb.Artifact("model", type="model", description="Class Conditioned Audio Diffusion Model.", metadata={"epoch": epochs})
    # #     #at.add_dir(os.path.join("models", run_name))
    # #     at.add_file(fullpath)
    # #     wandb.log_artifact(at)

    # #     print("Model uploaded!")

# Save the trained model weights
run_name = "Please_Work"

# Setup directory to store the model parameters
filepath = "Saved Models"
filename = f'{run_name}_{epochs}_epochs.pth'

# Create the directory if it doesn't exist
os.makedirs(filepath, exist_ok=True)
fullpath = os.path.join(filepath, filename)


save_model(model, fullpath)

# Load the state dictionary into the model
model.load_state_dict(torch.load(fullpath))
model.eval() # Set the model to evaluation mode



# Sample new audio using the trained model

# Generate labels, 10 of each class
class_indices = []
for i in range(len(dataset.genres)):#2):#len(dataset.notes)): # Only sampling blues #(10):
    class_indices.extend([i] * 1)

# Convert the list to a tensor
labels = torch.tensor(class_indices, device=device)

# Check the labels
print(labels)

# Generate new audio from the model

shape = (1, 1, resolution[0], resolution[1] * 10)  # Shape of the output spectrogram shape[0] = batch_size

# Initialize noise
images = torch.randn(shape, device=device)

for step in tqdm(reversed(range(0, timesteps)), desc="Sampling timestep", unit="step"):

    # Create a tensor filled with the current timestep for the batch
    batch_step = torch.full((shape[0],), step, device=device, dtype=torch.long)
    
    model_output = model(images, batch_step).detach() # Detaching prevents memory build up at each step

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
folder2 = "Unconditional_Audio"
folder = os.path.join(folder1, folder2)
#filename = "Audio class conditioned_"+epochs+"_epochs.pth"

# Create the directory if it doesn't exist
os.makedirs(folder, exist_ok=True)

# Save final images
for idx, image in enumerate(images):
    output_file = os.path.join(folder, f'output_audio_{idx}_{epochs}_epochs_{shape[2]}_x_res_{shape[3]}_y_res_PLEASE_WORK')
    output_mel_spectrogram = output_file + ".png"

    image.save(output_mel_spectrogram)

# Save the generated spectrogram image
#image.save(output_mel_spectrogram)
print("Generated spectrograms saved!")


# Load the generated spectrograms and convert to audio
generated_spectrogram_file = output_file#os.path.join(folder, "output_audio_0_100_epochs_64_x_res_640_y_res.png")
image = Image.open(generated_spectrogram_file)
audio = dataset.mel_spectrogram_to_audio(image)
audio_tensor = torch.tensor(audio).unsqueeze(0) # Add channel dimension due to the mono audio output (duplicate the channel so that the audio can be played)

# Save the audio file
output_audio_path = generated_spectrogram_file.replace(".png", ".wav")
torchaudio.save(output_audio_path, audio_tensor, dataset.sample_rate)
print("Audio saved!")


# Play the audio
def play_audio_from_waveform(waveform, sample_rate):
    # Convert waveform to 1D numpy array if it's mono
    if waveform.shape[0] == 1:
        waveform = waveform.squeeze().numpy()
    else:
        waveform = waveform.T.numpy()

    # Play audio using sounddevice
    print("Sample rate: ", sample_rate)
    sd.play(waveform, sample_rate)
    sd.wait()

waveform, sample_rate = torchaudio.load(output_audio_path)
play_audio_from_waveform(waveform, dataset.sample_rate)