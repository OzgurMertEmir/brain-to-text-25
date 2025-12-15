import torch
import torch.nn.functional as F

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,  padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    This version creates the kernel directly on the target device to avoid CPU-GPU data transfers within the training loop.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        smooth_kernel_std (float): Standard deviation of the Gaussian smoothing kernel.
        smooth_kernel_size (int): The size of the Gaussian kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (torch.device): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Create 1D Gaussian kernel directly on the target device
    x_cord = torch.arange(smooth_kernel_size, device=device, dtype=torch.float32)
    mean = (smooth_kernel_size - 1) / 2.
    variance = smooth_kernel_std**2.
    
    # Calculate the 1D gaussian kernel, mirroring the original scipy implementation
    gaussian_kernel = torch.exp(-((x_cord - mean)**2) / (2 * variance))
    gaussian_kernel[gaussian_kernel < 0.01] = 0
    
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape for convolution
    gaussKernel = gaussian_kernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]

def random_time_mask(inputs, max_mask_frac=0.1, num_masks=1):
    """
    Zero out random contiguous time spans (SpecAugment-style).

    Args:
        inputs:  (B, T, C) tensor on any device.
        max_mask_frac: maximum fraction of T for a single mask.
        num_masks: number of masks per example.

    Returns:
        Tensor of same shape, with some time windows set to zero.
    """
    if max_mask_frac <= 0 or num_masks <= 0:
        return inputs

    B, T, C = inputs.shape
    max_len = int(T * max_mask_frac)
    if max_len < 1:
        return inputs

    device = inputs.device
    out = inputs.clone()

    for b in range(B):
        for _ in range(num_masks):
            L = torch.randint(1, max_len + 1, (1,), device=device).item()
            start = torch.randint(0, max(1, T - L + 1), (1,), device=device).item()
            out[b, start:start + L, :] = 0.0

    return out


def random_channel_dropout(inputs, drop_prob=0.1):
    """
    Drop whole channels for each trial (simulates lost / very noisy electrodes).

    Args:
        inputs: (B, T, C) tensor.
        drop_prob: probability of dropping each channel independently.

    Returns:
        Tensor of same shape with some channels zeroed.
    """
    if drop_prob <= 0.0:
        return inputs

    B, T, C = inputs.shape
    device = inputs.device

    # (B, C) mask: True = keep, False = drop
    mask = (torch.rand(B, C, device=device) > drop_prob).float()
    mask = mask.view(B, 1, C)  # broadcast over time

    return inputs * mask


def random_time_shift(inputs, max_shift=5):
    """
    Apply a small random integer time shift per trial, zero-padding exposed edges.

    Args:
        inputs: (B, T, C) tensor.
        max_shift: maximum shift in time steps (both positive and negative).

    Returns:
        Shifted tensor of same shape.
    """
    if max_shift <= 0:
        return inputs

    B, T, C = inputs.shape
    device = inputs.device
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=device)

    out = torch.zeros_like(inputs)
    for b in range(B):
        s = int(shifts[b].item())
        if s > 0:
            # shift to the right
            out[b, s:, :] = inputs[b, :T - s, :]
        elif s < 0:
            # shift to the left
            s = -s
            out[b, :T - s, :] = inputs[b, s:, :]
        else:
            out[b] = inputs[b]

    return out