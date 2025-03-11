import diffusion_models
import datasets
import torch
import os
from tqdm.auto import tqdm
from diffusers import DDPMScheduler
from PIL import Image
import torchaudio


def generate(model, device, noise_scheduler, max_timesteps, shape, class_labels, display_steps=False):
    """ Performs the reverse diffusion for a given model, starting with random noise in the shape of the batch to be generated, then repeatedly removing the predicted noise from the images until a class conditioned audio sample is obtained.
        It then processes the generated samples into a format that can be displayed.
        Optionally displays the reverse diffusion process at each 1/5 of the steps"""
    
    # Set intervals for displaying intermediate steps
    display_intervals = []
    if display_steps:
        display_intervals = [0, 99, 199, 299, 399, 499, 599, 999, 1999, 2999, 3999] #max_timesteps * i // (num_intervals - 1) for i in range(num_intervals)]

    # Initialise noise for the batch
    images = torch.randn(shape, device=device)

    for step in tqdm(reversed(range(0, max_timesteps)), desc="Sampling timestep", unit="step"):

        # Create a tensor filled with the current timestep for the batch
        batch_step = torch.full((shape[0],), step, device=device, dtype=torch.long)
        
        # Predict the noise for the current batch given the current timestep and class embeddings
        model_output = model(images, batch_step, class_labels=class_labels).detach() #, slice_positions=slice_positions).detach() # Detaching prevents memory build up at each step

        # Perform the scheduler step and update images in-place
        images.copy_(noise_scheduler.step(
            model_output=model_output,
            timestep=step,
            sample=images,
            generator=None,
        )["prev_sample"].detach())

        # Save intermediate results if the current step matches a display interval
        if display_steps and step in display_intervals:
            print(f"Saving diffusion process at step {step}")
            intermediate_images = preprocess_images(images)
            save_samples(intermediate_images, os.path.join("Generated_Images", "Reverse Diffusion"), step)

            

    # Process the images
    images = preprocess_images(images)

    return images

def preprocess_images(images):
    """Processes the generated samples from the model into a format that can be displayed."""
    
    # Normalise the image pixel values
    images = (images / 2 + 0.5).clamp(0, 1) # During pre-processing, the range is from -1 to 1

    # Rearrange the dimensions of the tensor from (batch, channels, height, width) to (batch, height, width, channels) so that they can be correctly displayed
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    images = list(
        map(lambda _: Image.fromarray(_[:, :, 0]), images)
        if images.shape[3] == 1
        else map(lambda _: Image.fromarray(_, mode="RGB").convert("L"), images)
    )

    return images

def save_samples(images, folder_path, batch_name="TEST_IMAGE", save_audio=False):

    # Ensure the directory exists
    print("FOLDER PATH: ", folder_path)
    os.makedirs(folder_path, exist_ok=True)

    # Save final images
    for idx, image in enumerate(images):
        output_file = os.path.join(folder_path, f'{batch_name}_{idx}_seed_{seed}')
        output_mel_spectrogram = output_file + ".png"

        # Save the generated spectrogram image
        image.save(output_mel_spectrogram)
        print(f"Generated spectrogram {idx} saved!")

        if save_audio:
            # Load the generated spectrograms and convert to audio
            generated_spectrogram_file = output_mel_spectrogram#os.path.join(folder, "output_audio_0_100_epochs_64_x_res_640_y_res.png")
            image = Image.open(generated_spectrogram_file)
            audio = dataset.mel_spectrogram_to_audio(image)
            audio_tensor = torch.tensor(audio).unsqueeze(0) # Add channel dimension due to the mono audio output (duplicate the channel so that the audio can be played)

            # Save the audio file
            output_audio_path = generated_spectrogram_file.replace(".png", ".wav")
            torchaudio.save(output_audio_path, audio_tensor, dataset.sample_rate)
            print(f"Audio {idx} saved!")


if __name__ == '__main__':
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
    # TODO: Make sure that the hop-length, n_iter parameters are set correctly for the given pre-trained model
    dataset.hop_length = 512
    dataset.n_iter = 100


    # Setup directory to store the model parameters
    filepath = "Saved Models"
    # 1024 hop_length 32 n_iter
    # filename = 'Date_02_03_2025_100_epochs_1000_timesteps_0.95_0.0001_lr_class_conditioned_GTZAN_100_epochs.pth' #f'{run_name}_{epochs}_epochs.pth' # Test 1000 timesteps 64x64
    # filename = 'Date_03_03_2025_100_epochs_4000_timesteps_0.95_0.0001_lr_class_conditioned_GTZAN_100_epochs.pth' #f'{run_name}_{epochs}_epochs.pth' # Test 4000 timesteps 256x256 (doesn't work with current code)
    # filename = 'Date_05_03_2025_16_18_22_100_epochs_4000_timesteps_class__position_embeddings_GTZAN_100_epochs_64_x_res_position_embed.pth' # Positional embedding part 1 (shouldn't work)
    # filename = 'Date_06_03_2025_11_25_30_100_epochs_4000_timesteps_class__position_embeddings_GTZAN_100_epochs_64_x_res_position_embed.pth' # Positional embedding part 2 (shouldn't work)
    #filename = 'Date_06_03_2025_23_31_27_200_epochs_4000_timesteps_class_embeddings_GTZAN_200_epochs_64_x_res.pth' # 4000 timesteps more epochs (works with current code)

    # 256x256 512 hop_length, 100 n_iter, 4000 timesteps
    filename = 'Date_10_03_2025_01_44_19_100_epochs_4000_timesteps_class_embeddings_GTZAN_100_epochs_256_x_res.pth'


    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    fullpath = os.path.join(filepath, filename)

    # Load the state dictionary into the model
    model.load_state_dict(torch.load(fullpath))
    model.eval() # Set the model to evaluation mode



    # Sample new audio using the trained model

    # Generate labels: 1 sample for each class (len(dataset.genres) total classes)
    class_indices = [2,3]#[i for i in range(2)] # First n classes # len(dataset.genres))]  # One sample per class

    # Generate positional indices
    #slice_positions = torch.tensor([0] * len(class_indices), device=device)  # Shape: (batch_size,)

    # Batch size is equal to the number of classes
    batch_size = len(class_indices)
    shape = (batch_size, 1, resolution[0], resolution[1] * 2)  # Shape of spectrogram: (batch_size, channels, height, width/length)

    # Convert class indices into a tensor of shape (batch_size,)
    class_labels = torch.tensor(class_indices, device=device)  # Shape: (batch_size,)

    # Check the labels
    print("Class labels: ", class_labels)



    # Perform reverse diffusion

    images = generate(model, device, noise_scheduler, timesteps, shape, class_labels, display_steps=True)



    # Save the generated samples

    # Setup directory to store the generated audio mel-spectrograms
    folder1 = "Generated_Images"
    folder2 = "Class_Conditioned_Audio" #"Unconditional_Audio"
    folder = os.path.join(folder1, folder2)
    #filename = "Audio class conditioned_"+epochs+"_epochs.pth"

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    save_samples(images, folder, "Test_Batch_256", True)