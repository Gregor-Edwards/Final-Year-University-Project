import torch
from torch import nn
from einops import rearrange # needed by model blocks and training code
import math # Used within the positional embeddings



# Utility methods

def exists(x):
    """Utility function to check if a value exists (is not None)."""
    return x is not None



# Timestep schedulers

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



# Positional embeddings

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



# Model layer blocks

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
        self.time_emb_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        # # self.class_emb_mlp = (
        # #     nn.Sequential(nn.SiLU(), nn.Linear(class_emb_dim, dim_out))
        # #     if exists(class_emb_dim)
        # #     else None
        # # )

        self.block1 = Block(dim, dim_out, groups=groups) # First Block layer
        self.block2 = Block(dim_out, dim_out, groups=groups) # Second Block layer
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity() # Adjusts input if needed

    def forward(self, x, time_emb=None, class_emb=None):
        h = self.block1(x)

        # Adds time embedding if it exists
        if exists(self.time_emb_mlp) and exists(time_emb):
            time_emb = self.time_emb_mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        # # # Adds class embedding if it exists (do not use)
        # # if exists(self.class_emb_mlp) and exists(class_emb):
        # #     class_emb = self.class_emb_mlp(class_emb)
        # #     h = rearrange(class_emb, "b c -> b c 1 1") + h

        h = self.block2(h) # Second Block
        return h + self.res_conv(x) # Residual connection (function(x) + value) that allows better gradient flow through the model parameters

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



# Models

class AudioUnet2D(nn.Module):
    def __init__(self, 
                 sample_size, 
                 in_channels=1, 
                 out_channels=1, 
                 block_out_channels=(128, 128, 256, 256, 512, 512),
                 layers_per_block=2,
                 temb_channels=512):
        super().__init__()

        # Timestep embedding
        image_height = sample_size[0]
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(image_height),
            nn.Linear(image_height, temb_channels),
            nn.GELU(),
            nn.Linear(temb_channels, temb_channels),
        )

        # Initial convolution layer
        self.init_conv = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Downsampling layers
        self.down1_block1 = ResnetBlock(block_out_channels[0], block_out_channels[0], time_emb_dim=temb_channels)
        self.down1_block2 = ResnetBlock(block_out_channels[0], block_out_channels[0], time_emb_dim=temb_channels)
        self.down1_pool = nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=3, stride=2, padding=1)

        self.down2_block1 = ResnetBlock(block_out_channels[0], block_out_channels[1], time_emb_dim=temb_channels)
        self.down2_block2 = ResnetBlock(block_out_channels[1], block_out_channels[1], time_emb_dim=temb_channels)
        self.down2_pool = nn.Conv2d(block_out_channels[1], block_out_channels[1], kernel_size=3, stride=2, padding=1)

        self.down3_block1 = ResnetBlock(block_out_channels[1], block_out_channels[2], time_emb_dim=temb_channels)
        self.down3_block2 = ResnetBlock(block_out_channels[2], block_out_channels[2], time_emb_dim=temb_channels)
        self.down3_pool = nn.Conv2d(block_out_channels[2], block_out_channels[2], kernel_size=3, stride=2, padding=1)

        self.down4_block1 = ResnetBlock(block_out_channels[2], block_out_channels[3], time_emb_dim=temb_channels)
        self.down4_block2 = ResnetBlock(block_out_channels[3], block_out_channels[3], time_emb_dim=temb_channels)
        self.down4_pool = nn.Conv2d(block_out_channels[3], block_out_channels[3], kernel_size=3, stride=2, padding=1)

        self.down5_block1 = ResnetBlock(block_out_channels[3], block_out_channels[4], time_emb_dim=temb_channels)
        self.down5_block2 = ResnetBlock(block_out_channels[4], block_out_channels[4], time_emb_dim=temb_channels)
        self.down5_attention = nn.MultiheadAttention(embed_dim=block_out_channels[4], num_heads=4, batch_first=True)  # Attention Layer
        self.down5_pool = nn.Conv2d(block_out_channels[4], block_out_channels[4], kernel_size=3, stride=2, padding=1)

        self.down6_block1 = ResnetBlock(block_out_channels[4], block_out_channels[5], time_emb_dim=temb_channels)
        self.down6_block2 = ResnetBlock(block_out_channels[5], block_out_channels[5], time_emb_dim=temb_channels)

        # Bottleneck
        self.bottleneck_block1 = ResnetBlock(block_out_channels[5], block_out_channels[5], time_emb_dim=temb_channels)
        self.bottleneck_attention = nn.MultiheadAttention(embed_dim=block_out_channels[5], num_heads=4, batch_first=True)
        self.bottleneck_block2 = ResnetBlock(block_out_channels[5], block_out_channels[5], time_emb_dim=temb_channels)

        # Upsampling layers
        self.up6_upsample = nn.ConvTranspose2d(block_out_channels[5], block_out_channels[4], kernel_size=2, stride=2)
        self.up6_block1 = ResnetBlock(block_out_channels[5] + block_out_channels[4], block_out_channels[4], time_emb_dim=temb_channels)
        self.up6_attention = nn.MultiheadAttention(embed_dim=block_out_channels[4], num_heads=4, batch_first=True)  # Attention Layer
        self.up6_block2 = ResnetBlock(block_out_channels[4], block_out_channels[4], time_emb_dim=temb_channels)

        self.up5_upsample = nn.ConvTranspose2d(block_out_channels[4], block_out_channels[3], kernel_size=2, stride=2)
        self.up5_block1 = ResnetBlock(block_out_channels[4], block_out_channels[3], time_emb_dim=temb_channels)
        self.up5_block2 = ResnetBlock(block_out_channels[3], block_out_channels[3], time_emb_dim=temb_channels)

        self.up4_upsample = nn.ConvTranspose2d(block_out_channels[3], block_out_channels[2], kernel_size=2, stride=2)
        self.up4_block1 = ResnetBlock(block_out_channels[3] + block_out_channels[2], block_out_channels[2], time_emb_dim=temb_channels)
        self.up4_block2 = ResnetBlock(block_out_channels[2], block_out_channels[2], time_emb_dim=temb_channels)

        self.up3_upsample = nn.ConvTranspose2d(block_out_channels[2], block_out_channels[1], kernel_size=2, stride=2)
        self.up3_block1 = ResnetBlock(block_out_channels[2], block_out_channels[1], time_emb_dim=temb_channels)
        self.up3_block2 = ResnetBlock(block_out_channels[1], block_out_channels[1], time_emb_dim=temb_channels)

        self.up2_upsample = nn.ConvTranspose2d(block_out_channels[1], block_out_channels[0], kernel_size=2, stride=2)
        self.up2_block1 = ResnetBlock(block_out_channels[1] + block_out_channels[0], block_out_channels[0], time_emb_dim=temb_channels)
        self.up2_block2 = ResnetBlock(block_out_channels[0], block_out_channels[0], time_emb_dim=temb_channels)

        # Final convolution layer
        self.final_conv = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps):
        # Compute timestep embedding
        temb = self.time_mlp(timesteps)

        # Initial convolution
        x = self.init_conv(x)

        #print("Start: ", x.shape)
        #print("Start t_emb: ", temb.shape)

        # Downsampling
        h1 = self.down1_block1(x, temb)
        h1 = self.down1_block2(h1, temb)
        h1_downsampled = self.down1_pool(h1)

        #print("h1: ", h1_downsampled.shape)

        h2 = self.down2_block1(h1_downsampled, temb)
        h2 = self.down2_block2(h2, temb)
        h2_downsampled = self.down2_pool(h2)

        #print("h2: ", h2_downsampled.shape)

        h3 = self.down3_block1(h2_downsampled, temb)
        h3 = self.down3_block2(h3, temb)
        h3_downsampled = self.down3_pool(h3)

        #print("h3: ", h3_downsampled.shape)

        h4 = self.down4_block1(h3_downsampled, temb)
        h4 = self.down4_block2(h4, temb)
        h4_downsampled = self.down4_pool(h4)

        #print("h4: ", h4_downsampled.shape)

        h5 = self.down5_block1(h4_downsampled, temb)
        h5 = self.down5_block2(h5, temb)
        h5_shape = h5.shape
        h5 = h5.view(h5_shape[0], h5_shape[1], -1).permute(0, 2, 1)  # Reshape for attention
        h5 = self.down5_attention(h5, h5, h5)[0]
        h5 = h5.permute(0, 2, 1).view(h5_shape)  # Restore shape
        h5_downsampled = self.down5_pool(h5)

        #print("h5: ", h5_downsampled.shape)

        h6 = self.down6_block1(h5_downsampled, temb)
        h6 = self.down6_block2(h6, temb)

        #print("h6: ", h6.shape)

        # Bottleneck
        x = self.bottleneck_block1(h6, temb)
        b_shape = x.shape
        x = x.view(b_shape[0], b_shape[1], -1).permute(0, 2, 1)  # Prepare for attention
        x, _ = self.bottleneck_attention(x, x, x)
        x = x.permute(0, 2, 1).view(b_shape)  # Restore shape
        x = self.bottleneck_block2(x, temb)

        # Upsampling
        x = self.up6_upsample(x)
        x = torch.cat([x, h5], dim=1)
        x = self.up6_block1(x, temb)
        x = self.up6_block2(x, temb)

        x = self.up5_upsample(x)
        x = torch.cat([x, h4], dim=1)
        x = self.up5_block1(x, temb)
        x = self.up5_block2(x, temb)

        x = self.up4_upsample(x)
        x = torch.cat([x, h3], dim=1)
        x = self.up4_block1(x, temb)
        x = self.up4_block2(x, temb)

        x = self.up3_upsample(x)
        x = torch.cat([x, h2], dim=1)  # Skip connection
        x = self.up3_block1(x)
        x = self.up3_block2(x)

        x = self.up2_upsample(x)
        x = torch.cat([x, h1], dim=1)  # Skip connection
        x = self.up2_block1(x)
        x = self.up2_block2(x)

        prediction = self.final_conv(x)

        # Final convolution
        return prediction

class ClassConditionedAudioUnet2D(AudioUnet2D):
    def __init__(self, 
                 sample_size, 
                 in_channels=1, 
                 out_channels=1, 
                 block_out_channels=(128, 128, 256, 256, 512, 512),
                 layers_per_block=2,
                 temb_channels=512,
                 num_classes=10): # Add the number of classes for conditioning
                 #max_position=3): # Add position of the time slices from the dataset 
        super().__init__(sample_size, in_channels, out_channels, block_out_channels, layers_per_block, temb_channels)

        # Class embedding layer
        self.class_label_emb = nn.Embedding(num_classes, temb_channels)

        # Positional embedding layer for slice positions
        #self.slice_position_emb = nn.Embedding(max_position + 1, temb_channels // 4) # +1 because 0 is a valid position

        combined_size = temb_channels + temb_channels #+ (temb_channels // 4)

        # Combine timestep, class, and positional embeddings
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_size, temb_channels),  # Reduce to match embedding size
            nn.GELU(),
            nn.Dropout(p=0.2),  # Dropout for regularisation
            nn.Linear(temb_channels, temb_channels),  # Final projection
            nn.GELU()  # Stabilisation
        )

        #self.layer_norm = nn.LayerNorm(temb_channels) # Used to normalise each embedding so that one embedding does not dominate over the others

    def forward(self, x, timesteps, class_labels=None):#, slice_positions=None):
        """
        Args:
            x: Tensor of shape (batch_size, in_channels, height, width), the noisy input samples.
            timesteps: Tensor of shape (batch_size,), the timesteps corresponding to the noisy samples.
            class_labels: Tensor of shape (batch_size,), the class labels for conditioning.
        Returns:
            prediction: Tensor of shape (batch_size, out_channels, height, width), the predicted noise.
        """
        # Compute timestep embedding
        temb = self.time_mlp(timesteps)

        # Compute class embedding
        class_emb = self.class_label_emb(class_labels) if class_labels is not None else 0

        # Compute positional embedding
        #positional_emb = self.slice_position_emb(slice_positions) if slice_positions is not None else 0

        # Normalize embeddings
        #temb = self.layer_norm(temb)
        #class_emb = self.layer_norm(class_emb)
        #positional_emb = self.layer_norm(positional_emb)
        # Normalize timestep and class embeddings separately
        temb = nn.functional.normalize(temb, p=2, dim=-1)  # L2 normalization
        class_emb = nn.functional.normalize(class_emb, p=2, dim=-1)
        
        # Combine timestep, class, and positional embeddings
        combined_emb = torch.cat([temb, class_emb], dim=-1) # torch.cat([temb, class_emb, positional_emb], dim=-1)
        combined_emb = self.combined_mlp(combined_emb) 

        # Predict the noise using the combined conditional embeddings
        # This cannot reuse the parent class' forward method from here, since the timestep embeddings would be recreated!
        
        # Initial convolution
        x = self.init_conv(x)

        # Downsampling
        h1 = self.down1_block1(x, temb)#, class_emb)
        h1 = self.down1_block2(h1, temb)#, class_emb)
        h1_downsampled = self.down1_pool(h1)

        h2 = self.down2_block1(h1_downsampled, temb)#, class_emb)
        h2 = self.down2_block2(h2, temb)#, class_emb)
        h2_downsampled = self.down2_pool(h2)

        h3 = self.down3_block1(h2_downsampled, temb)#, class_emb)
        h3 = self.down3_block2(h3, temb)#, class_emb)
        h3_downsampled = self.down3_pool(h3)

        h4 = self.down4_block1(h3_downsampled, temb)#, class_emb)
        h4 = self.down4_block2(h4, temb)#, class_emb)
        h4_downsampled = self.down4_pool(h4)

        h5 = self.down5_block1(h4_downsampled, temb)#, class_emb)
        h5 = self.down5_block2(h5, temb)#, class_emb)
        h5_shape = h5.shape
        h5 = h5.view(h5_shape[0], h5_shape[1], -1).permute(0, 2, 1)  # Reshape for attention
        h5 = self.down5_attention(h5, h5, h5)[0]
        h5 = h5.permute(0, 2, 1).view(h5_shape)  # Restore shape
        h5_downsampled = self.down5_pool(h5)

        h6 = self.down6_block1(h5_downsampled, temb)#, class_emb)
        h6 = self.down6_block2(h6, temb)#, class_emb)

        # Bottleneck
        x = self.bottleneck_block1(h6, temb)#, class_emb)
        b_shape = x.shape
        x = x.view(b_shape[0], b_shape[1], -1).permute(0, 2, 1)  # Prepare for attention
        x, _ = self.bottleneck_attention(x, x, x)
        x = x.permute(0, 2, 1).view(b_shape)  # Restore shape
        x = self.bottleneck_block2(x, temb)#, class_emb)

        # Upsampling
        x = self.up6_upsample(x)
        x = torch.cat([x, h5], dim=1)
        x = self.up6_block1(x, temb)#, class_emb)
        x = self.up6_block2(x, temb)#, class_emb)

        x = self.up5_upsample(x)
        x = torch.cat([x, h4], dim=1)
        x = self.up5_block1(x, temb)#, class_emb)
        x = self.up5_block2(x, temb)#, class_emb)

        x = self.up4_upsample(x)
        x = torch.cat([x, h3], dim=1)
        x = self.up4_block1(x, temb)#, class_emb)
        x = self.up4_block2(x, temb)#, class_emb)

        x = self.up3_upsample(x)
        x = torch.cat([x, h2], dim=1)  # Skip connection
        x = self.up3_block1(x, temb)#, class_emb)
        x = self.up3_block2(x, temb)#, class_emb)

        x = self.up2_upsample(x)
        x = torch.cat([x, h1], dim=1)  # Skip connection
        x = self.up2_block1(x, temb)#, class_emb)
        x = self.up2_block2(x, temb)#, class_emb)

        prediction = self.final_conv(x)

        return prediction
