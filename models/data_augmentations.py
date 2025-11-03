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