import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cp_ndimage
from functools import lru_cache


# Cache for disk kernels to avoid recreation
@lru_cache(maxsize=32)
def create_disk_kernel_gpu_cached(radius):
    """
    Create a disk-shaped kernel on GPU with caching.
    """
    if radius <= 0:
        return cp.array([[True]], dtype=cp.bool_)
    
    size = 2 * radius + 1
    y, x = cp.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = mask.astype(cp.bool_)
    
    return kernel

def calculate_local_color_variance(img_gpu, window_size):

    # Pre-allocate arrays
    h, w, c = img_gpu.shape
    kernel_size = window_size * window_size
    kernel_gpu = cp.ones((window_size, window_size), dtype=cp.float32) / kernel_size
    
    # Calculate grayscale variance 
    gray_gpu = cp.mean(img_gpu, axis=2)
    
    
    if window_size > 5:
        # Create 1D kernel
        kernel_1d = cp.ones(window_size, dtype=cp.float32) / window_size
        
        # Apply separable convolution for mean
        temp = cp_ndimage.convolve1d(gray_gpu, kernel_1d, axis=0, mode='reflect')
        local_mean_gpu = cp_ndimage.convolve1d(temp, kernel_1d, axis=1, mode='reflect')
        
        # Apply separable convolution for squared values
        gray_squared = gray_gpu ** 2
        temp = cp_ndimage.convolve1d(gray_squared, kernel_1d, axis=0, mode='reflect')
        local_mean_squared = cp_ndimage.convolve1d(temp, kernel_1d, axis=1, mode='reflect')
        
        # Variance = E[X^2] - E[X]^2
        local_variance_gpu = local_mean_squared - local_mean_gpu ** 2
    else:
        # For small kernels, use regular convolution
        local_mean_gpu = cp_ndimage.convolve(gray_gpu, kernel_gpu, mode='reflect')
        gray_squared = gray_gpu ** 2
        local_mean_squared = cp_ndimage.convolve(gray_squared, kernel_gpu, mode='reflect')
        local_variance_gpu = local_mean_squared - local_mean_gpu ** 2
    
    # RGB variance 
    rgb_variance_gpu = cp.zeros_like(gray_gpu)
    
    # Pre-allocate temporary arrays
    if window_size > 5:
        kernel_1d = cp.ones(window_size, dtype=cp.float32) / window_size
        temp = cp.empty_like(gray_gpu)
        
        for c in range(3):
            channel_gpu = img_gpu[:, :, c]
            
            # Separable convolution for mean
            cp_ndimage.convolve1d(channel_gpu, kernel_1d, axis=0, mode='reflect', output=temp)
            channel_mean = cp_ndimage.convolve1d(temp, kernel_1d, axis=1, mode='reflect')
            
            # Separable convolution for squared values
            channel_squared = channel_gpu ** 2
            cp_ndimage.convolve1d(channel_squared, kernel_1d, axis=0, mode='reflect', output=temp)
            channel_mean_squared = cp_ndimage.convolve1d(temp, kernel_1d, axis=1, mode='reflect')
            
            rgb_variance_gpu += (channel_mean_squared - channel_mean ** 2)
    else:
        for c in range(3):
            channel_gpu = img_gpu[:, :, c]
            channel_mean = cp_ndimage.convolve(channel_gpu, kernel_gpu, mode='reflect')
            channel_squared = channel_gpu ** 2
            channel_mean_squared = cp_ndimage.convolve(channel_squared, kernel_gpu, mode='reflect')
            rgb_variance_gpu += (channel_mean_squared - channel_mean ** 2)
    
    rgb_variance_gpu /= 3
    
    # Combine both measures using in-place operations
    combined_variance_gpu = 0.7 * local_variance_gpu
    combined_variance_gpu += 0.3 * rgb_variance_gpu
    
    return combined_variance_gpu

def automatic_background_detection(img_rgb, params={ 'variance_window': 100,
                                                         'variance_threshold': 0.0001,
                                                         'morph_kernel_size': 0,
                                                         'min_background_fraction': 0.3 }):

    
    variance_window = params['variance_window']
    variance_threshold = params['variance_threshold']
    morph_kernel_size = params['morph_kernel_size']
    
    # Transfer image to GPU 
    img_gpu = cp.asarray(img_rgb, dtype=cp.float32, order='C')
    
    # Calculate local color variance 
    variance_map_gpu = calculate_local_color_variance(img_gpu, variance_window)
    
    # Initial background detection based on low variance
    background_mask_gpu = variance_map_gpu < variance_threshold
    
    # Dilation and erosion operations to remove small objects and fill small holes
    if morph_kernel_size > 0:
        # Get cached kernel
        kernel_gpu = create_disk_kernel_gpu_cached(morph_kernel_size)
        
        temp_mask = cp.empty_like(background_mask_gpu)
        
        # Binary closing (dilation followed by erosion)
        cp_ndimage.binary_dilation(background_mask_gpu, kernel_gpu, output=temp_mask)
        cp_ndimage.binary_erosion(temp_mask, kernel_gpu, output=background_mask_gpu)
        
        # Binary opening (erosion followed by dilation)
        cp_ndimage.binary_erosion(background_mask_gpu, kernel_gpu, output=temp_mask)
        cp_ndimage.binary_dilation(temp_mask, kernel_gpu, output=background_mask_gpu)
    
    # Convert to mask format (0 = background, 1 = foreground)
    mask_gpu = (~background_mask_gpu).astype(cp.uint8)
    
    # Transfer back to CPU
    mask = cp.asnumpy(mask_gpu)
    
    return mask
