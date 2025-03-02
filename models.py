import torch
from torch import nn
from functools import partial # to create partial functions, allowing pre-filling of certain arguments
from einops import rearrange # needed by model blocks and training code
from torch import einsum  # needed by model blocks and training code
import math # Used within the positional embeddings



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
                 num_classes=10):  # Add the number of classes for conditioning
        super().__init__(sample_size, in_channels, out_channels, block_out_channels, layers_per_block, temb_channels)

        # Class embedding layer
        self.class_label_emb = nn.Embedding(num_classes, temb_channels)

    def forward(self, x, timesteps, class_labels=None):
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

        # Add class label embedding to timestep embedding
        if class_labels is not None:
            class_emb = self.class_label_emb(class_labels)  # Shape: (batch_size, temb_channels)
            temb = temb + class_emb  # Combine timestep embedding with class embedding

        # Initial convolution
        x = self.init_conv(x)

        # Downsampling
        h1 = self.down1_block1(x, temb)
        h1 = self.down1_block2(h1, temb)
        h1_downsampled = self.down1_pool(h1)

        h2 = self.down2_block1(h1_downsampled, temb)
        h2 = self.down2_block2(h2, temb)
        h2_downsampled = self.down2_pool(h2)

        h3 = self.down3_block1(h2_downsampled, temb)
        h3 = self.down3_block2(h3, temb)
        h3_downsampled = self.down3_pool(h3)

        h4 = self.down4_block1(h3_downsampled, temb)
        h4 = self.down4_block2(h4, temb)
        h4_downsampled = self.down4_pool(h4)

        h5 = self.down5_block1(h4_downsampled, temb)
        h5 = self.down5_block2(h5, temb)
        h5_shape = h5.shape
        h5 = h5.view(h5_shape[0], h5_shape[1], -1).permute(0, 2, 1)  # Reshape for attention
        h5 = self.down5_attention(h5, h5, h5)[0]
        h5 = h5.permute(0, 2, 1).view(h5_shape)  # Restore shape
        h5_downsampled = self.down5_pool(h5)

        h6 = self.down6_block1(h5_downsampled, temb)
        h6 = self.down6_block2(h6, temb)

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
        x = self.up3_block1(x, temb)
        x = self.up3_block2(x, temb)

        x = self.up2_upsample(x)
        x = torch.cat([x, h1], dim=1)  # Skip connection
        x = self.up2_block1(x, temb)
        x = self.up2_block2(x, temb)

        prediction = self.final_conv(x)

        return prediction


class SimplifiedUNet2D(nn.Module):
    def __init__(self, 
                 sample_size, 
                 in_channels=1, 
                 out_channels=1, 
                 block_out_channels=(128, 128, 256, 256, 512, 512)):
        super().__init__()

        # Timestep embedding
        image_height = sample_size[0]
        time_dim = image_height * 4 # The larger dimensionality of the time dimension allows the model to encode more information about the timestep
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(image_height), # This version of positional embeddings is inspired by the transformer paper "attention is all you need"
                nn.Linear(image_height, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        
        # Initial convolution layer
        self.init_conv = nn.Conv2d(in_channels + time_dim, block_out_channels[0], kernel_size=3, padding=1)
        
        # Timestep embedding
        # # time_dim = block_out_channels[0]  # Match the number of channels in the first block
        # # self.time_mlp = nn.Sequential(
        # #     SinusoidalPositionEmbeddings(image_height),  # Output shape: [batch_size, 1]
        # #     nn.Linear(image_height, time_dim),           # Map scalar timestep to time_dim
        # #     nn.GELU(),
        # #     nn.Linear(time_dim, time_dim * sample_size[0] * sample_size[1]),  # Expand to match full spatial dimensions
        # #     nn.GELU()
        # # )

        # Downsampling layers
        self.down1_block1 = nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=3, padding=1)
        self.down1_block2 = nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=3, padding=1)
        self.down1_pool = nn.MaxPool2d(kernel_size=2)

        self.down2_block1 = nn.Conv2d(block_out_channels[0], block_out_channels[1], kernel_size=3, padding=1)
        self.down2_block2 = nn.Conv2d(block_out_channels[1], block_out_channels[1], kernel_size=3, padding=1)
        self.down2_pool = nn.MaxPool2d(kernel_size=2)

        self.down3_block1 = nn.Conv2d(block_out_channels[1], block_out_channels[2], kernel_size=3, padding=1)
        self.down3_block2 = nn.Conv2d(block_out_channels[2], block_out_channels[2], kernel_size=3, padding=1)
        self.down3_pool = nn.MaxPool2d(kernel_size=2)

        self.down4_block1 = nn.Conv2d(block_out_channels[2], block_out_channels[3], kernel_size=3, padding=1)
        self.down4_block2 = nn.Conv2d(block_out_channels[3], block_out_channels[3], kernel_size=3, padding=1)
        self.down4_pool = nn.MaxPool2d(kernel_size=2)

        self.down5_block1 = nn.Conv2d(block_out_channels[3], block_out_channels[4], kernel_size=3, padding=1)
        self.down5_block2 = nn.Conv2d(block_out_channels[4], block_out_channels[4], kernel_size=3, padding=1)
        self.down5_pool = nn.MaxPool2d(kernel_size=2)

        self.down6_block1 = nn.Conv2d(block_out_channels[4], block_out_channels[5], kernel_size=3, padding=1)
        self.down6_block2 = nn.Conv2d(block_out_channels[5], block_out_channels[5], kernel_size=3, padding=1)

        # Bottleneck
        self.bottleneck_block1 = nn.Conv2d(block_out_channels[5], block_out_channels[5], kernel_size=3, padding=1)
        self.bottleneck_attention = nn.MultiheadAttention(embed_dim=block_out_channels[5], num_heads=1, batch_first=True)
        self.bottleneck_block2 = nn.Conv2d(block_out_channels[5], block_out_channels[5], kernel_size=3, padding=1)

        # Upsampling layers
        self.up6_upsample = nn.ConvTranspose2d(block_out_channels[5], block_out_channels[4], kernel_size=2, stride=2)
        self.up6_block1 = nn.Conv2d(block_out_channels[5] + block_out_channels[4], block_out_channels[4], kernel_size=3, padding=1) # The  + block_out_channels[4] is due to the appended skip connections
        self.up6_block2 = nn.Conv2d(block_out_channels[4], block_out_channels[4], kernel_size=3, padding=1)

        self.up5_upsample = nn.ConvTranspose2d(block_out_channels[4], block_out_channels[3], kernel_size=2, stride=2)
        self.up5_block1 = nn.Conv2d(block_out_channels[4], block_out_channels[3], kernel_size=3, padding=1)
        self.up5_block2 = nn.Conv2d(block_out_channels[3], block_out_channels[3], kernel_size=3, padding=1)

        self.up4_upsample = nn.ConvTranspose2d(block_out_channels[3], block_out_channels[2], kernel_size=2, stride=2)
        self.up4_block1 = nn.Conv2d(block_out_channels[3] + block_out_channels[2], block_out_channels[2], kernel_size=3, padding=1)
        self.up4_block2 = nn.Conv2d(block_out_channels[2], block_out_channels[2], kernel_size=3, padding=1)

        self.up3_upsample = nn.ConvTranspose2d(block_out_channels[2], block_out_channels[1], kernel_size=2, stride=2)
        self.up3_block1 = nn.Conv2d(block_out_channels[2], block_out_channels[1], kernel_size=3, padding=1)
        self.up3_block2 = nn.Conv2d(block_out_channels[1], block_out_channels[1], kernel_size=3, padding=1)

        self.up2_upsample = nn.ConvTranspose2d(block_out_channels[1], block_out_channels[0], kernel_size=2, stride=2)
        self.up2_block1 = nn.Conv2d(block_out_channels[1] + block_out_channels[0], block_out_channels[0], kernel_size=3, padding=1)
        self.up2_block2 = nn.Conv2d(block_out_channels[0], block_out_channels[0], kernel_size=3, padding=1)

        # Final convolution layer
        self.final_conv = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps):
        """
        Forward pass to predict the noise from the noisy samples.

        Args:
            x: Tensor of shape (batch_size, in_channels, height, width), the noisy input samples.
            timesteps: Tensor of shape (batch_size,), the timesteps corresponding to the noisy samples.

        Returns:
            Tensor of shape (batch_size, out_channels, height, width), the predicted noise.
        """
        #print("x_dim: ", x.shape)

        # Compute timestep embedding
        t_embedding = self.time_mlp(timesteps)
        #print("t_emb_dim: ", t_embedding.shape)

        # Reshape timestep embedding to match the spatial dimensions of x
        t_embedding = t_embedding[:, :, None, None]  # Shape: [batch_size, time_dim, 1, 1]
        t_embedding = t_embedding.expand(-1, -1, x.shape[2], x.shape[3]) # Shape: [batch_size, time_dim, height, width]

        # Concatenate along the channel dimension
        x = torch.cat((x, t_embedding), dim=1)

        # Initial convolution
        x = self.init_conv(x)

        # Downsampling (adding timestep embedding to each down block)

        h1 = self.down1_block1(x)
        h1 = self.down1_block2(h1)
        h1_downsampled = self.down1_pool(h1)

        h2 = self.down2_block1(h1_downsampled)
        h2 = self.down2_block2(h2)
        h2_downsampled = self.down2_pool(h2)

        h3 = self.down3_block1(h2_downsampled)
        h3 = self.down3_block2(h3)
        h3_downsampled = self.down3_pool(h3)

        h4 = self.down4_block1(h3_downsampled)
        h4 = self.down4_block2(h4)
        h4_downsampled = self.down4_pool(h4)

        h5 = self.down5_block1(h4_downsampled)
        h5 = self.down5_block2(h5)
        h5_downsampled = self.down5_pool(h5)

        h6 = self.down6_block1(h5_downsampled)
        h6 = self.down6_block2(h6)

        # Bottleneck with attention
        bottleneck = self.bottleneck_block1(h6)
        b_shape = bottleneck.shape
        bottleneck = bottleneck.view(b_shape[0], b_shape[1], -1).permute(0, 2, 1)  # Flatten spatial dimensions
        bottleneck, _ = self.bottleneck_attention(bottleneck, bottleneck, bottleneck)  # Apply attention
        bottleneck = bottleneck.permute(0, 2, 1).view(b_shape)  # Reshape back to original
        bottleneck = self.bottleneck_block2(bottleneck)

        # Upsampling (adding timestep embedding to each up block)
        x = self.up6_upsample(bottleneck)
        x = torch.cat([x, h5], dim=1)  # Skip connection
        x = self.up6_block1(x)
        x = self.up6_block2(x)

        x = self.up5_upsample(x)
        x = torch.cat([x, h4], dim=1)  # Skip connection
        #print("Test: ", x.shape)
        x = self.up5_block1(x)
        x = self.up5_block2(x)

        x = self.up4_upsample(x)
        x = torch.cat([x, h3], dim=1)  # Skip connection
        x = self.up4_block1(x)
        x = self.up4_block2(x)

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
            label_emb = label_emb.unsqueeze(-1).unsqueeze(-1)  # Reshape to [batch_size, emb_dim, 1, 1]
            label_emb = label_emb.expand(-1, -1, x.shape[2], x.shape[3])  # Now label_emb should be [12, emb_dim, 128, 216]

            # Check the batch sizes and dimensions for concatenation
            #print(f"x shape: {x.shape}, label_emb shape: {label_emb.shape}")
            assert x.shape[0] == label_emb.shape[0], f"The batch sizes should be the same, but are actually: {x.shape[0]} and {label_emb.shape[0]}!"
            
            # Concatenate along the channel dimension
            x = torch.cat((x, label_emb), dim=1)



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



# TODO: Test forward diffusion