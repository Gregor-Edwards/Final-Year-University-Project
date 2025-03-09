import datasets
import diffusion_models
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

# Reverse diffusion
import generate_from_pretrained_model

# Weights and Biases
import wandb
from datetime import datetime

# Testing
import tests

if __name__ == "__main__":

    # Initialise constants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0) # use seed for reproducability
    resolution = (64, 64) # (128, 128) Compromise? #(256, 256) better quality but too large to train on my current GPU #(64, 64) best for training quickly #dataset[0]["image"].height, dataset[0]["image"].width
    batch_size = 64 #4 if the resolution is lower, a higher batch_size can be used before the vram on the GPU is used up

    # Initialise forward diffusion parameters

    timesteps = 4000 #1000 # 4000
    #noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=4000,
        beta_start=0.00005,  # Adjusted for 4000 timesteps
        beta_end=0.01,       # Adjusted for 4000 timesteps
        beta_schedule="linear"
    )


    # Initialise dataset and dataloader

    dataset = datasets.GTZANDataset(resolution=resolution, root_dir="GTZAN_Genre_Collection/genres", spectrogram_dir="GTZAN_Genre_Collection/slices")

    # Save mel-spectrograms to disk if they are not present
    print("Saving mel spectrograms")

    dataset.save_mel_spectrograms()

    print("Done.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # Test the forward diffusion
    tests.test_forward_diffusion(dataset, device, timesteps, noise_scheduler)




    # Initialise model

    # sample_rate = 22050 # All files in the  dataset should have this sample_rate
    channels = 1 # Mono audio
    num_classes = len(dataset.genres)

    epochs = 200 # 100 #2000#2000 # 20 epochs or above starts to produce 'reasonable' quality images but it takes longer time
    learning_rate = 1e-4#, 1e-3] # 1e-5, too low with a learning rate scheduler, 1e-3 too high
    num_classes = len(dataset.genres)


    #model = models.SimplifiedUNet2D(resolution).to(device)
    #model = models.AudioUnet2D(resolution).to(device)
    #max_slice_position = 18 # Dependent on the resolution of the slices and the length of the audio samples
    model = diffusion_models.ClassConditionedAudioUnet2D(sample_size=resolution, num_classes=num_classes).to(device)#, max_position=max_slice_position).to(device)





    # Define the optimizer for training the model.
    # model.parameters(): Parameters of the U-Net model to optimize.
    beta_start = 0.95
    beta_end = 0.999
    weight_decay = 1e-6
    epsilon = 1e-08
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(beta_start, beta_end), weight_decay=weight_decay, eps=epsilon,) # Weight decay is L2 regularisation (a term added to the loss function being optimised)


    # Learning rate scheduler
    warmup_steps = 5000 # 500
    num_steps_per_epoch = len(dataloader)
    total_training_steps = num_steps_per_epoch * epochs

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps,) # Used to vary the learning rate at each step to better tune the training process


    # Setup Weights and Biases

    # Get current date and time
    now = datetime.now()

    # Format time as hh:mm:ss
    time_str = now.strftime("%H_%M_%S")

    # Format date as dd/mm/yyyy
    date_str = now.strftime("%d_%m_%Y")

    print("Current Time:", time_str)
    print("Current Date:", date_str)

    run_name = f'Date_{date_str}_{time_str}_{epochs}_epochs_{timesteps}_timesteps_class_embeddings_GTZAN' # MAKE SURE THIS NAME IS NOT TOO LONG!!!

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
        "Noise scheduler": "DDPM",
        "timesteps": timesteps,
        "learning_rate": learning_rate,
        "learning_rate_scheduler": True,
        "learning_rate_warmup": warmup_steps,
        "architecture": "CNN",
        "number_of_layers": 6,
        "dataset": "GTZAN",
        }
    )

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
            class_labels = batch['label'].to(device)
            #batch_positions = batch['position'].to(device)

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
            noise_prediction = model(noisy_mel_spectrograms, timestep, class_labels=class_labels) #, slice_positions=batch_positions)
            loss = F.mse_loss(noise_prediction, noise)

            # log metrics to wandb
            wandb.log({f'mse_loss': loss})
            wandb.log({f'learning_rate': lr_scheduler.get_last_lr()[0]})

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

        if upload_model:
            # Upload artifact to Weights and Biases
            at = wandb.Artifact("model", type="model", description="Class Conditioned Audio Diffusion Model.", metadata={"epoch": epochs})
            #at.add_dir(os.path.join("models", run_name))
            at.add_file(fullpath)
            wandb.log_artifact(at)

            print("Model uploaded!")

    # Setup directory to store the model parameters
    filepath = "Saved Models"
    filename = f'{run_name}_{epochs}_epochs_{resolution[0]}_x_res.pth'

    # Create the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)
    fullpath = os.path.join(filepath, filename)


    save_model(model, fullpath)

    # Load the state dictionary into the model
    model.load_state_dict(torch.load(fullpath))
    model.eval() # Set the model to evaluation mode



    # Sample new audio using the trained model

    # Generate labels: 1 sample for each class (len(dataset.genres) total classes)
    class_indices = [i for i in range(1)] # First n classes # len(dataset.genres))]  # One sample per class

    # Generate positional indices
    #slice_positions = torch.tensor([0] * len(class_indices), device=device)  # Shape: (batch_size,)

    # Batch size is equal to the number of classes
    batch_size = len(class_indices)
    shape = (batch_size, 1, resolution[0], resolution[1])  # Shape of spectrogram: (batch_size, channels, height, width/length)

    # Convert class indices into a tensor of shape (batch_size,)
    class_labels = torch.tensor(class_indices, device=device)  # Shape: (batch_size,)

    # Check the labels
    print("Class labels: ", class_labels)

    # Perform reverse diffusion
    images = generate_from_pretrained_model.generate(model, noise_scheduler, timesteps, shape, class_labels)



    # Setup directory to store the generated audio mel-spectrograms
    folder1 = "Generated_Images"
    folder2 = "Class_Conditioned_Audio" #"Unconditional_Audio"
    folder = os.path.join(folder1, folder2)
    #filename = "Audio class conditioned_"+epochs+"_epochs.pth"

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save final images
    upload_mel_spectrogram_files = []
    upload_audio_files = []
    for idx, image in enumerate(images):
        output_file = os.path.join(folder, f'{run_name}_output_audio_{idx}_{epochs}_epochs_{shape[2]}_x_res_{shape[3]}_y_res_PLEASE_WORK')
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

        upload_mel_spectrogram_files.append(output_mel_spectrogram)
        upload_audio_files.append(output_audio_path)

    # Upload files to weights and biases

    # log mel-spectrograms to wandb
    #wandb.log({f'{output_file}': wandb.Image(output_file)})
    wandb.log({"sampled_mel_spectrograms":     [wandb.Image(output_file) for output_file in upload_mel_spectrogram_files]})

    # log audio to wandb
    wandb.log({"sampled_audio":     [wandb.Audio(output_file, sample_rate=dataset.sample_rate) for output_file in upload_audio_files]})

    print("Samples uploaded!")


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

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()