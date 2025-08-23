import os
import cv2
import numpy as np
import cupy as cp
import time
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from .data_management import load_batch_data, review_and_edit_results, save_clustering_results, create_sample_batch_file, save_analysis_data, npz2dict
from .background_detection import automatic_background_detection
from .visualization import inference_visualization, generate_cluster_colors, training_visualizations


##############################################
# FUNCTIONS SHARED BETWEEN TRAINING AND TESTING ARE BELOW
##############################################

def line2d(x, y, coeffs=[1]*3, return_coeff=False):
    """Returns the result of a plane, or returns the coefficients"""
    a0 = (x*0+1)*coeffs[0]
    a1 = x*coeffs[1]
    a2 = y*coeffs[2]
    if return_coeff:
        return a0, a1, a2
    else:
        return a0+a1+a2



##############################################
# TESTING FUNCTIONS ARE BELOW
##############################################

def multivar_gauss_testing(x, mean, cov, n_clusters):
    """
    Multivariate Gaussian probability density function for testing step
    
    Parameters:
    -----------
    x : cupy.ndarray
        Input data of shape (n_samples, n_features)
    mean : cupy.ndarray
        Mean vector of shape (n_features,)
    cov : cupy.ndarray
        Covariance matrix of shape (n_features, n_features)
    n_clusters : int
        Number of clusters (for normalization)
    
    Returns:
    --------
    prob : cupy.ndarray
        Probability density values of shape (n_samples,)
    """
    n_features = x.shape[1]
    
    # Calculate determinant and inverse of covariance matrix
    det_cov = cp.linalg.det(cov)
    
    # Handle singular covariance matrix
    if det_cov <= 0:
        # Add small regularization
        cov_reg = cov + cp.eye(n_features) * 1e-6
        det_cov = cp.linalg.det(cov_reg)
        inv_cov = cp.linalg.inv(cov_reg)
    else:
        inv_cov = cp.linalg.inv(cov)
    
    # Compute (x - mean)
    diff = x - mean
    
    # Compute quadratic form: (x-mean)^T * inv_cov * (x-mean)
    # Using einsum for efficient batch computation
    quad_form = cp.einsum('ni,ij,nj->n', diff, inv_cov, diff)
    
    # Compute normalization factor
    norm_factor = 1.0 / cp.sqrt((2 * cp.pi) ** n_features * det_cov)
    
    # Compute probability density
    prob = norm_factor * cp.exp(-0.5 * quad_form)
    
    return prob

def calculate_cluster_probabilities(pixels_gpu, master_weights, master_means, master_covs, clusters):
    """
    Calculation of cluster probabilities for all pixels.
    
    Parameters:
    -----------
    pixels_gpu : cupy.ndarray
        Pixel data of shape (n_pixels, 3) in BGR order
    master_weights : dict
        Cluster weights
    master_means : dict
        Cluster means for each color channel
    master_covs : dict
        Covariance matrices for each cluster
    clusters : list
        List of cluster indices
    
    Returns:
    --------
    cluster_prob : numpy.ndarray
        Probability matrix of shape (n_pixels, n_clusters)
    """
    n_pixels = pixels_gpu.shape[0]
    n_clusters = len(clusters)
    
    # Pre-allocate probability matrix on GPU
    cluster_prob_gpu = cp.zeros((n_pixels, n_clusters), dtype=cp.float32)
    
    # Pre-compute all cluster parameters on GPU
    weights_gpu = cp.array([master_weights[tt] for tt in clusters], dtype=cp.float32)
    means_gpu = []
    covs_gpu = []
    
    for tt in clusters:
        # Stack means in BGR order to match pixel data
        mean = cp.array([master_means['blue'][tt], master_means['green'][tt], 
                        master_means['red'][tt]], dtype=cp.float32)
        means_gpu.append(mean)
        
        # Ensure covariance matrix is float32
        cov = cp.asarray(master_covs[tt], dtype=cp.float32)
        covs_gpu.append(cov)
    
    # Calculate probabilities for each cluster
    for i, tt in enumerate(clusters):
        # Calculate multivariate Gaussian probability for all pixels at once
        prob = multivar_gauss_testing(pixels_gpu, means_gpu[i], covs_gpu[i], n_clusters)
        cluster_prob_gpu[:, i] = weights_gpu[i] * prob
    
    # Calculate normalization (sum across clusters for each pixel)
    prob_sum = cp.sum(cluster_prob_gpu, axis=1, keepdims=True)
    
    # Handle zero denominators
    prob_sum = cp.where(prob_sum == 0, 1.0 / n_clusters, prob_sum)
    
    # Normalize probabilities
    cluster_prob_gpu = cluster_prob_gpu / prob_sum
    
    # Handle NaN values (set to uniform probability)
    nan_mask = cp.isnan(cluster_prob_gpu)
    cluster_prob_gpu = cp.where(nan_mask, 1.0 / n_clusters, cluster_prob_gpu)
    
    # Transfer back to CPU
    cluster_prob = cp.asnumpy(cluster_prob_gpu)
    
    return cluster_prob

def testing(img_file, master_cat_file=None, cluster_count=None, 
            comp_rate=100, show_plot=True):
    """
    Main testing function for image classification

    Parameters
    ----------
    img_file : str
        Location of sample image file
    master_cat_file : str
        Location of master catalog npz file
    cluster_count : int
        Number of layers to fit up to (including residue if present)
    comp_rate : int [default: 100]
        Compression factor: comp = sqrt(pixels)/comp_rate
    show_plot : bool [default: True]
        Whether to display plots during processing
    """
    
    tic = time.perf_counter()
    
    # Extract filename without extension for naming outputs
    flake_name = os.path.splitext(os.path.basename(img_file))[0]
    print(f"Processing image: {flake_name}")

    # Load image
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    original_image = img.copy()
    print(f"Original image shape: {img.shape}")
    
    # Convert to float and normalize
    img_bl = img.astype(np.float32) / 255.0
    
    # First bilateral filtering
    print("Applying bilateral filtering...")
    for _ in range(1):
        img_bl = cv2.bilateralFilter(img_bl, 2, 1, 1)
    
    # Conduct automatic background detection
    print("Detecting background regions automatically...")

    mask = automatic_background_detection(img_bl)
        
    # Check background fraction
    background_fraction = np.sum(mask == 0) / mask.size
    
    # Background fitting and subtraction
    print("Fitting and subtracting background...")
    y_dim, x_dim, _ = img_bl.shape
    R = img_bl[:,:,0].flatten()
    G = img_bl[:,:,1].flatten()
    B = img_bl[:,:,2].flatten()
    X_, Y_ = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    X = X_.flatten()
    Y = Y_.flatten()
    
    # Find substrate locations (background regions)
    sub_loc = ((mask.flatten()) == 0).nonzero()[0]
    if len(sub_loc) > 0:
        Rsub = R[sub_loc]
        Gsub = G[sub_loc]
        Bsub = B[sub_loc]
        Xsub = X[sub_loc]
        Ysub = Y[sub_loc]
        
        # Fit polynomial background
        Asub = np.array([*line2d(Xsub, Ysub, return_coeff=True)]).T
        
        Rcop, _, _, _ = np.linalg.lstsq(Asub, Rsub, rcond=None)
        Gcop, _, _, _ = np.linalg.lstsq(Asub, Gsub, rcond=None)
        Bcop, _, _, _ = np.linalg.lstsq(Asub, Bsub, rcond=None)
        
        # Apply background correction
        Rfitp = line2d(X, Y, coeffs=[*Rcop])
        Gfitp = line2d(X, Y, coeffs=[*Gcop])
        Bfitp = line2d(X, Y, coeffs=[*Bcop])
        
        img_poly = np.dstack([(R-Rfitp+1).reshape(y_dim,x_dim)/2,
                              (G-Gfitp+1).reshape(y_dim,x_dim)/2,
                              (B-Bfitp+1).reshape(y_dim,x_dim)/2])
    else:
        print("No background regions detected, skipping background subtraction")
        img_poly = img_bl
    
    # Show background reduction if requested
    if show_plot:
        print('Manually inspect background reduction, then close figures.')
        plt.figure()
        plt.imshow(img_poly)
        plt.title('Background-corrected image')
        plt.figure()
        plt.imshow(mask)
        plt.title('Mask (black = background regions)')
        plt.show()
    
    # Second round of bilateral filtering
    print("Applying second round of bilateral filtering...")
    img_bl2 = img_poly.astype(np.float32)
    for _ in range(3):
        img_bl2 = cv2.bilateralFilter(img_bl2, 2, 0.5, 1)
    
    if comp_rate is None:
        img_size = img_bl2.shape
        img_proc = img_bl2
        img_proc_display = original_image
        comp = 1
    else:
        # Optional Image compression 
        img_size = img_bl2.shape
        comp = int(((img_size[0] * img_size[1]) ** 0.5) / comp_rate)
        comp = max(1, comp)
        img_proc = img_bl2[0::comp, 0::comp]
        img_proc_display = original_image[0::comp, 0::comp]
        print(f"Compressed image shape: {img_proc.shape}, compression factor: {comp}")
    
    y_size, x_size, _ = img_proc.shape

    # Extract RGB channels
    R = img_proc[:,:,0].flatten()
    G = img_proc[:,:,1].flatten()
    B = img_proc[:,:,2].flatten()
    
    # Filter valid pixels
    valid_pixels = (R > 0.01) & (G > 0.01) & (B > 0.01) & (R < 0.99) & (G < 0.99) & (B < 0.99)
    R_valid = R[valid_pixels]
    G_valid = G[valid_pixels]
    B_valid = B[valid_pixels]
    
    print(f"Processing {len(R_valid)} valid pixels out of {len(R)} total pixels")

    # Import the master catalog
    in_file_dict = npz2dict(master_cat_file)
    
    # Extract available layers and optional residue
    available_layers = []
    has_residue = False
    
    for key in in_file_dict.keys():
        if key.startswith('weights-'):
            if key.endswith('layers'):
                layer_num = int(key.split('-')[1].replace('layers', ''))
                available_layers.append(layer_num)
            elif key == 'weights-residue':
                has_residue = True
    
    available_layers = sorted(available_layers)
    print(f"Available layers in master catalog: {available_layers}")
    if has_residue:
        print("Residue data is also available in master catalog")
    
    # Limit cluster_count to available data
    total_available = len(available_layers) + (1 if has_residue else 0)
    if cluster_count is None:
        cluster_count = total_available
    max_clusters = min(cluster_count, total_available)
    if max_clusters < cluster_count:
        print(f"Requested {cluster_count} clusters, but only {max_clusters} available in catalog")
    
    # Load master catalog data
    master_weights = {}
    master_red_mean = {}
    master_green_mean = {}
    master_blue_mean = {}
    master_cov = {}

    clusters = []
    layer_mapping = {}  # Map cluster index to actual layer number or 'residue'
    
    # Load layer data
    for i, layer_num in enumerate(available_layers):
        if i >= max_clusters:
            break
        try:
            master_weights[i] = in_file_dict[f'weights-{layer_num}layers']
            master_red_mean[i] = in_file_dict[f'red mean-{layer_num}layers']
            master_green_mean[i] = in_file_dict[f'green mean-{layer_num}layers']
            master_blue_mean[i] = in_file_dict[f'blue mean-{layer_num}layers']
            master_cov[i] = in_file_dict[f'covariance-{layer_num}layers']
            clusters.append(i)
            layer_mapping[i] = layer_num  # Store actual layer number
            print(f'Loaded data for {layer_num} layers (cluster index {i})')
        except KeyError as e:
            print(f'Missing data for {layer_num} layers: {e}')
    
    # Add residue if present and we haven't reached max_clusters
    if has_residue and len(clusters) < max_clusters:
        residue_idx = len(clusters)
        try:
            master_weights[residue_idx] = in_file_dict['weights-residue']
            master_red_mean[residue_idx] = in_file_dict['red mean-residue']
            master_green_mean[residue_idx] = in_file_dict['green mean-residue']
            master_blue_mean[residue_idx] = in_file_dict['blue mean-residue']
            master_cov[residue_idx] = in_file_dict['covariance-residue']
            clusters.append(residue_idx)
            layer_mapping[residue_idx] = 'residue'
            print(f'Loaded residue data (cluster index {residue_idx})')
        except KeyError as e:
            print(f'Missing residue data: {e}')

    if len(clusters) == 0:
        raise ValueError("No valid cluster data found in master catalog")

    pixel_count = len(R_valid)
    print(f"Calculating probabilities for {pixel_count} valid pixels using GPU...")

    # Prepare data for GPU processing
    # Stack pixels in BGR order to match covariance matrix ordering
    pixels_bgr = np.column_stack([B_valid, G_valid, R_valid])
    
    # Transfer to GPU
    pixels_gpu = cp.asarray(pixels_bgr, dtype=cp.float32)
    
    # Organize master data for GPU processing
    master_means = {
        'red': master_red_mean,
        'green': master_green_mean,
        'blue': master_blue_mean
    }
    
    # Calculate probabilities on GPU
    tic_gpu = time.perf_counter()
    cluster_prob = calculate_cluster_probabilities(
        pixels_gpu, master_weights, master_means, master_cov, clusters
    )
    toc_gpu = time.perf_counter()
    print(f'GPU probability calculation completed in {toc_gpu-tic_gpu:.2f} seconds')

    # Assign each pixel to most probable cluster
    nearest_cluster_valid = np.argmax(cluster_prob, axis=1)
    nearest_cluster_valid = np.array([clusters[idx] for idx in nearest_cluster_valid])

    # Create full layer image
    nearest_cluster_full = np.full(len(R), -1, dtype=float)
    nearest_cluster_full[valid_pixels] = nearest_cluster_valid
    layer_image = nearest_cluster_full.reshape(y_size, x_size)

    toc = time.perf_counter()
    print(f'Time elapsed: {toc-tic:.2f} seconds')

    # Generate cluster colors based on available layers
    cluster_colors = generate_cluster_colors(len(clusters))
    
    # Calculate pixel counts and percentages for each cluster
    pixel_counts = {}
    for i in clusters:
        pixel_counts[i] = np.sum(nearest_cluster_valid == i)
    
    # Calculate total image statistics
    total_pixels = layer_image.size
    valid_pixel_count = len(nearest_cluster_valid)
    invalid_pixel_count = np.sum(layer_image.flatten() == -1)
    background_pixel_count = np.sum(mask == 0)
    
    # Create visualization of classification results and save as image if requested
    inference_visualization(
        layer_image=layer_image,
        original_image=original_image,
        processed_image=img_proc,
        img_proc_display=img_proc_display,
        clusters=clusters,
        pixel_counts=pixel_counts,
        total_pixels=total_pixels,
        valid_pixel_count=valid_pixel_count,
        invalid_pixel_count=invalid_pixel_count,
        background_pixel_count=background_pixel_count,
        background_fraction=background_fraction,
        mask=mask,
        flake_name=flake_name,
        show_plot=show_plot
    )
    
    # Save analysis data to json and text file
    save_analysis_data(
        flake_name=flake_name,
        clusters=clusters,
        pixel_counts=pixel_counts,
        total_pixels=total_pixels,
        valid_pixel_count=valid_pixel_count,
        invalid_pixel_count=invalid_pixel_count,
        background_pixel_count=background_pixel_count,
        background_fraction=background_fraction,
        compression_factor=comp,
        runtime=toc - tic,
        layer_mapping=layer_mapping  # Pass the mapping
    )
    
    print(f"Results saved to ./outputs/classification_results/")
    
    # Return results with visualization data
    results = {
        'layer_image': layer_image,
        'cluster_probabilities': cluster_prob,
        'nearest_cluster': nearest_cluster_valid,
        'valid_pixels': valid_pixels,
        'available_layers': available_layers,
        'has_residue': has_residue,
        'layer_mapping': layer_mapping,  # Include mapping in results
        'processed_image': img_proc,
        'original_cropped': img_proc_display,
        'compression_factor': comp,
        'total_runtime': toc - tic,
        'background_fraction': background_fraction,
        'mask': mask,
        'R': R_valid,
        'G': G_valid,
        'B': B_valid,
        'cluster_colors': cluster_colors,
        'pixel_counts': pixel_counts,
        'master_weights': master_weights,
        'cluster_count': len(clusters)
    }
    
    return results



##############################################
# TRAINING FUNCTIONS ARE BELOW
##############################################

def multivariate_gaussian_training(pixels, means, covariances, weights):
    """Fully vectorized multivariate Gaussian computation on GPU"""
    n_pixels, n_dims = pixels.shape
    n_clusters = means.shape[0]
    
    # Expand dimensions for broadcasting
    pixels_expanded = pixels[:, cp.newaxis, :]  # (n_pixels, 1, n_dims)
    means_expanded = means[cp.newaxis, :, :]    # (1, n_clusters, n_dims)
    
    # Compute differences
    diff = pixels_expanded - means_expanded  # (n_pixels, n_clusters, n_dims)
    
    # Compute log probabilities for numerical stability
    log_probs = cp.zeros((n_pixels, n_clusters))
    
    for k in range(n_clusters):
        try:
            
            reg_cov = covariances[k] + cp.eye(n_dims) * 1e-6
            
            # Solve covariance system 
            inv_cov_diff = cp.linalg.solve(reg_cov, diff[:, k, :].T).T
            mahalanobis_dist = cp.sum(diff[:, k, :] * inv_cov_diff, axis=1)
            
            log_det = cp.linalg.slogdet(reg_cov)[1]
            log_probs[:, k] = -0.5 * (mahalanobis_dist + log_det + n_dims * cp.log(2 * cp.pi))
            log_probs[:, k] += cp.log(cp.maximum(weights[k], 1e-10))  # Avoid log(0)
        except cp.linalg.LinAlgError:
            # Handle singular matrices
            log_probs[:, k] = -cp.inf
    
    # Convert to probabilities 
    max_log_prob = cp.max(log_probs, axis=1, keepdims=True)
    exp_probs = cp.exp(log_probs - max_log_prob)
    prob_sums = cp.sum(exp_probs, axis=1, keepdims=True)
    
    # Avoid division by zero
    prob_sums = cp.maximum(prob_sums, 1e-10)
    
    return exp_probs / prob_sums

def confellipsoid(weight, meanB, meanG, meanR, covariance):
    """
    
    Parameters:
    -----------
    weight : float
        Cluster weight (not used in current implementation).
    meanB : float
        Blue channel mean.
    meanG : float
        Green channel mean.
    meanR : float
        Red channel mean.
    covariance : array_like
        3x3 covariance matrix.
        
    Returns:
    --------
    numpy.ndarray
        Ellipsoid surface coordinates (shape: [50, 50, 3]).
    """
    # Convert inputs to numpy if they're CuPy arrays
    if hasattr(meanB, 'get'):
        meanB = meanB.get()
    if hasattr(meanG, 'get'):
        meanG = meanG.get()
    if hasattr(meanR, 'get'):
        meanR = meanR.get()
    if hasattr(covariance, 'get'):
        covariance = covariance.get()
    
    # Create a grid for the ellipsoid surface
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    # Unit sphere coordinates
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Stack coordinates
    sphere_coords = np.stack([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()], axis=0)
    
    # Eigenvalue decomposition for ellipsoid transformation
    try:
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        # Ensure positive eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-8)
        
        # 95% confidence interval scaling (chi-squared with 3 DOF)
        scale_factor = np.sqrt(7.815)  # chi2(0.95, 3)
        
        # Transform unit sphere to ellipsoid
        scaling_matrix = eigenvecs @ np.diag(np.sqrt(eigenvals)) * scale_factor
        ellipsoid_coords = scaling_matrix @ sphere_coords
        
        # Translate to cluster mean
        center = np.array([meanB, meanG, meanR]).reshape(3, 1)
        ellipsoid_coords += center
        
        # Reshape back to grid format
        x_ellipsoid = ellipsoid_coords[0].reshape(x_sphere.shape)
        y_ellipsoid = ellipsoid_coords[1].reshape(y_sphere.shape)
        z_ellipsoid = ellipsoid_coords[2].reshape(z_sphere.shape)
        
        return np.stack([x_ellipsoid, y_ellipsoid, z_ellipsoid], axis=2)
        
    except np.linalg.LinAlgError:
        # Fallback to simple sphere if covariance is singular
        radius = scale_factor * np.mean(np.sqrt(np.diag(covariance)))
        x_ellipsoid = x_sphere * radius + meanB
        y_ellipsoid = y_sphere * radius + meanG
        z_ellipsoid = z_sphere * radius + meanR
        
        return np.stack([x_ellipsoid, y_ellipsoid, z_ellipsoid], axis=2)

        
def covariance_update_training(pixels, cluster_probs, means):

    n_pixels, n_dims = pixels.shape
    n_clusters = means.shape[0]
    covariances = cp.zeros((n_clusters, n_dims, n_dims))
    
    for k in range(n_clusters):
        # Weighted differences
        diff = pixels - means[k]  # (n_pixels, n_dims)
        weights = cluster_probs[:, k]  # (n_pixels,)
        weight_sum = cp.sum(weights)
        
        if weight_sum > 1e-10:
            # Vectorized outer product computation
            weighted_diff = diff * cp.sqrt(weights[:, cp.newaxis])
            covariances[k] = cp.dot(weighted_diff.T, weighted_diff) / weight_sum
            
            # Add regularization to prevent singular matrices
            covariances[k] += cp.eye(n_dims) * 1e-6
        else:
            # Initialize with identity if no points assigned
            covariances[k] = cp.eye(n_dims)
    
    return covariances


def process_pixels_in_batches(pixels, means, covariances, weights, batch_size=50000):
    """Process pixels in batches to manage GPU memory"""
    n_pixels = pixels.shape[0]
    n_batches = (n_pixels + batch_size - 1) // batch_size
    
    results = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_pixels)
        batch = pixels[start_idx:end_idx]
        
        # Process batch
        batch_result = multivariate_gaussian_training(batch, means, covariances, weights)
        results.append(batch_result)
    
    return cp.vstack(results)

def meanshift_initialization(pixels, density=8, max_cycles=500, convergence_threshold=1e-6, outlier_fraction=175):
    """
    
    Parameters:
    -----------
    pixels : cupy.ndarray
        Pixel data in RGB format (n_pixels, 3)
    density : int
        density^3 gives the number of initial mean points
    max_cycles : int
        Maximum number of mean-shift iterations
    convergence_threshold : float
        Threshold for determining convergence
    outlier_fraction : int
        Divisor for pixel threshold (pixels.shape[0] / outlier_fraction)
        Default 175 matches original code
        
    Returns:
    --------
    means : cupy.ndarray
        Final mean-shift centroids
    """
    B, G, R = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    
    # Get data bounds
    BGRmin = cp.array([cp.min(B), cp.min(G), cp.min(R)])
    BGRmax = cp.array([cp.max(B), cp.max(G), cp.max(R)])
    
    # Initialize array of starting points (regularly-spaced points in BGR space)
    axes_B = cp.linspace(BGRmin[0], BGRmax[0], density)
    axes_G = cp.linspace(BGRmin[1], BGRmax[1], density)
    axes_R = cp.linspace(BGRmin[2], BGRmax[2], density)
    
    # Create mesh grid
    mesh_B, mesh_G, mesh_R = cp.meshgrid(axes_B, axes_G, axes_R, indexing='ij')
    
    # Flatten to get initial mean points
    meanB = mesh_B.flatten()
    meanG = mesh_G.flatten()
    meanR = mesh_R.flatten()
    means = cp.vstack([meanB, meanG, meanR]).T  # Shape: (density^3, 3)
    
    # Calculate epsilon (radius of each point's sphere of possession)
    diag = cp.sqrt(cp.sum((BGRmax - BGRmin)**2)) / (density - 1)
    epsilon = diag / 3
    
    # Track which mean points are still active
    mp_life = cp.ones(len(means), dtype=bool)
    mean_count = cp.zeros(len(means))
    
    print(f"Starting mean-shift with {len(means)} initial points, epsilon={epsilon:.4f}")
    
    with tqdm(total=max_cycles, desc="Mean-shift algorithm", unit="cycle") as pbar:
        for cycle in range(max_cycles):
            if not cp.any(mp_life):
                print(f"Optimization achieved after {cycle} cycles.")
                break
            
            # Process only active mean points
            active_indices = cp.where(mp_life)[0]
            kill_indices = []
            
            for idx in active_indices:
                # Find all pixels within epsilon distance
                distances = cp.sqrt(cp.sum((pixels - means[idx])**2, axis=1))
                within_sphere = distances < epsilon
                pixel_count = cp.sum(within_sphere)
                
                if pixel_count == 0:
                    # Mark for removal
                    kill_indices.append(idx)
                else:
                    # Calculate new mean position
                    new_mean = cp.mean(pixels[within_sphere], axis=0)
                    
                    # Check if position changed
                    if cp.allclose(means[idx], new_mean):
                        mp_life[idx] = False  # Converged
                    elif pixel_count >= mean_count[idx]:
                        # Update only if pixel count increased or stayed same
                        means[idx] = new_mean
                        mean_count[idx] = pixel_count
                    else:
                        # Pixel count decreased, deactivate
                        mp_life[idx] = False
            
            # Remove empty mean points
            if kill_indices:
                mask = cp.ones(len(means), dtype=bool)
                mask[kill_indices] = False
                means = means[mask]
                mean_count = mean_count[mask]
                mp_life = mask[mask]  # Reindex mp_life
            
            pbar.update(1)
    
    # Remove outliers 
    pixel_threshold = pixels.shape[0] / outlier_fraction  
    sufficient_pixels = mean_count >= pixel_threshold
    final_means = means[sufficient_pixels]
    final_counts = mean_count[sufficient_pixels]
    
    print(f"Mean-shift completed: {len(final_means)} mean points after outlier removal")
    print(f"(Removed {len(means) - len(final_means)} outliers with < {pixel_threshold:.0f} pixels)")
    
    return final_means

def dbscan_consolidation(means, original_epsilon=None, eps_factor=0.5):
    """
    DBSCAN clustering to consolidate mean-shift results.
    """
    if len(means) == 0:
        return means
    
    n_means = len(means)
    
    # If we know the original epsilon from mean-shift, use it
    # Otherwise calculate from data range
    if original_epsilon is not None:
        eps2 = original_epsilon * eps_factor
    else:
        # Calculate epsilon for DBSCAN (based on mean-shift epsilon)
        data_range = cp.max(means, axis=0) - cp.min(means, axis=0)
        diag = cp.sqrt(cp.sum(data_range**2))
        density_estimate = int((n_means ** (1/3)) + 0.5)  # Estimate original density
        eps2 = (diag / max(density_estimate - 1, 1)) / 3 * eps_factor
    
    # DBSCAN algorithm
    mp_visit = cp.zeros(n_means, dtype=bool)
    mp_grouping = cp.full(n_means, -1, dtype=int)
    group_number = 0
    
    print(f"Starting DBSCAN consolidation with eps={eps2:.4f}")
    
    while not cp.all(mp_visit):
        # If no members in current group, start with first unvisited
        if group_number not in mp_grouping:
            unvisited = cp.where(~mp_visit)[0]
            if len(unvisited) == 0:
                break
            kk = unvisited[0]
            mp_visit[kk] = True
            mp_grouping[kk] = group_number
        
        # Process current group
        changed = True
        while changed:
            changed = False
            current_group = cp.where(mp_grouping == group_number)[0]
            
            for member in current_group:
                if mp_visit[member]:
                    continue
                    
                mp_visit[member] = True
                
                # Find all neighbors
                distances = cp.sqrt(cp.sum((means - means[member])**2, axis=1))
                neighbors = cp.where((distances < eps2) & (mp_grouping == -1))[0]
                
                if len(neighbors) > 0:
                    mp_grouping[neighbors] = group_number
                    changed = True
        
        # Move to next group
        if cp.any(mp_grouping == -1):
            group_number += 1
        else:
            break
    
    # Consolidate groups by averaging
    consolidated_means = []
    for group in range(group_number + 1):
        group_indices = cp.where(mp_grouping == group)[0]
        if len(group_indices) > 0:
            group_mean = cp.mean(means[group_indices], axis=0)
            consolidated_means.append(group_mean)
    
    if len(consolidated_means) > 0:
        consolidated_means = cp.vstack(consolidated_means)
    else:
        consolidated_means = cp.array([])
    
    print(f"DBSCAN completed: {len(consolidated_means)} groups from {n_means} mean points")
    
    return consolidated_means


### K means function below is unused and deprecated 

def kmeans_initialization(pixels, mean_number=100, max_cycles=100):
    """
    
    Parameters:
    -----------
    pixels : cupy.ndarray
        Pixel data in RGB format (n_pixels, 3)
    mean_number : int
        Number of initial means for k-means
    max_cycles : int
        Maximum number of k-means iterations
        
    Returns:
    --------
    means : cupy.ndarray
        Final k-means centroids
    """
    B, G, R = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    
    # Initialize means randomly
    meanB = cp.random.random(mean_number)
    meanG = cp.random.random(mean_number)
    meanR = cp.random.random(mean_number)
    
    # Normalize means to data range
    B_min, B_max = cp.min(B), cp.max(B)
    G_min, G_max = cp.min(G), cp.max(G)
    R_min, R_max = cp.min(R), cp.max(R)
    
    meanB = meanB * (B_max - B_min) + B_min
    meanG = meanG * (G_max - G_min) + G_min
    meanR = meanR * (R_max - R_min) + R_min
    
    # K-means algorithm - vectorized implementation
    with tqdm(total=max_cycles, desc="K-means clustering", unit="cycle") as pbar:
        for cycle in range(max_cycles):
            # Vectorized distance computation
            means = cp.vstack((meanB, meanG, meanR)).T  # shape (mean_number, 3)
            
            # Compute distances between all pixels and all means using broadcasting
            distances = cp.sqrt(cp.sum((pixels[:, cp.newaxis, :] - means[cp.newaxis, :, :])**2, axis=2))
            
            # Find nearest mean for each pixel
            nearest_mean = cp.argmin(distances, axis=1)
            
            # Update means vectorized
            for mm in range(mean_number):
                mask = nearest_mean == mm
                if cp.sum(mask) > 0:
                    meanB[mm] = cp.mean(B[mask])
                    meanG[mm] = cp.mean(G[mask])
                    meanR[mm] = cp.mean(R[mask])
            
            pbar.update(1)
    
    return cp.vstack([meanB, meanG, meanR]).T


def group_similar_means(means, distance_threshold=10):
    """
    Group k-means results that are close to each other.
    
    Parameters:
    -----------
    means : cupy.ndarray
        K-means centroids
    distance_threshold : float
        Distance threshold for grouping means
        
    Returns:
    --------
    grouped_means : cupy.ndarray
        Grouped mean centroids
    """
    mean_number = means.shape[0]
    
    # Vectorized distance calculation between means
    mean_distances = cp.sqrt(cp.sum((means[:, cp.newaxis, :] - means[cp.newaxis, :, :])**2, axis=2))

    # Group means based on distances
    group_number = -1
    mp_grouping = cp.full(mean_number, -1)
    
    with tqdm(total=mean_number, desc="Grouping means", unit="mean") as pbar:
        for mm in range(mean_number):
            if mp_grouping[mm] == -1:
                group_number += 1
                mp_grouping[mm] = group_number
                
                close_means = cp.where((mean_distances[mm] < distance_threshold) & (mp_grouping == -1))[0]
                mp_grouping[close_means] = group_number
            pbar.update(1)
    
    # Calculate grouped means
    grouped_means = []
    for nn in range(group_number + 1):
        indices = cp.where(mp_grouping == nn)[0]
        if len(indices) > 0:
            group_mean = cp.mean(means[indices], axis=0)
            grouped_means.append(group_mean)
    
    return cp.array(grouped_means)


def adjust_cluster_count(means, min_clusters, max_clusters, data_range):
    """
    Adjust the number of clusters to be within specified bounds.
    
    Parameters:
    -----------
    means : cupy.ndarray
        Current cluster means
    min_clusters : int
        Minimum number of clusters
    max_clusters : int
        Maximum number of clusters
    data_range : dict
        Dictionary with 'B_min', 'B_max', 'G_min', 'G_max', 'R_min', 'R_max'
        
    Returns:
    --------
    adjusted_means : cupy.ndarray
        Means adjusted to meet cluster count requirements
    """
    cluster_count = len(means)
    
    if cluster_count < min_clusters:
        additional_means = min_clusters - cluster_count
        extra_means = cp.column_stack([
            cp.random.uniform(data_range['B_min'], data_range['B_max'], additional_means),
            cp.random.uniform(data_range['G_min'], data_range['G_max'], additional_means),
            cp.random.uniform(data_range['R_min'], data_range['R_max'], additional_means)
        ])
        means = cp.vstack([means, extra_means])
        
    elif cluster_count > max_clusters:
        cluster_diff = cluster_count - max_clusters
        means = means[:-cluster_diff]
    
    return means


def training(img_file, flake_name, crop=None, masking=None,
             min_clusters=5, max_clusters=10, convergence_param=1e-6, 
             batch_size=50000, comp_rate=None, show_plot=True, 
             auto_background_params=None, 
             initialization_method='meanshift',
             density=8,
             dbscan_eps_factor=0.5,  
             outlier_fraction=175):   
    """
    Main training function for generating the master catalog
    
    Parameters:
    -----------
    img_file : str
        Path to the image file.
    flake_name : str
        Name of the flake.
    crop : list, optional
        Crop coordinates [y_min, y_max, x_min, x_max].
    masking : list, optional
        List of masking coordinates [[y_min, y_max, x_min, x_max], ...].
        If None and auto_background_params is provided, will use automatic background detection.
    min_clusters : int, optional
        Minimum number of clusters.
    max_clusters : int, optional
        Maximum number of clusters.
    convergence_param : float, optional
        Convergence parameter for the GMM algorithm.
    batch_size : int, optional
        Batch size for processing pixels to manage GPU memory.
    comp_rate : int, optional
        Factor in calculating the compression of the normalized image.
        Compression is comp = sqrt(pixels in cropped image)/comp_rate
    show_plot : bool, optional
        Whether to display plots. Set to False for batch processing.
    auto_background_params : dict, optional
        Parameters for automatic background detection (only used if masking is None):
        {
            'variance_window': int (default 50),
            'variance_threshold': float (default 0.001),
            'morph_kernel_size': int (default 20),
            'min_background_fraction': float (default 0.3)
        }
    initialization_method : str, optional
        Method for initializing clusters: 'meanshift' or 'kmeans'
    density : int, optional
        For mean-shift: density^3 gives the number of initial mean points
        
    Returns:
    --------
    results_dict : dict
        Dictionary containing the clustering results for the specified flake.
    """
    
    tic = time.perf_counter()
    
    # Load image using OpenCV 
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    original_image = img.copy()
    print(f"Original image shape: {img.shape}")
    
    # Apply cropping from user input
    if crop is not None:
        img_cropped = img[crop[0]:crop[1], crop[2]:crop[3]]
    else:
        img_cropped = img
        crop = [0, img.shape[0], 0, img.shape[1]]  # Full image bounds
    
    print(f"Cropped image shape: {img_cropped.shape}")
    
    # Convert to float and normalize for bilateral filtering
    img_bl = img_cropped.astype(np.float32) / 255.0
    
    # Apply bilateral filtering 
    print("Applying bilateral filtering...")
    for ii in range(1):
        img_bl = cv2.bilateralFilter(img_bl, 2, 1, 1)


    # Create mask for background fitting
    if masking is not None:
        # Use manual masking
        mask = np.ones(img_bl[:,:,0].shape)
        for reg in masking:
            mask[reg[0]:reg[1], reg[2]:reg[3]] = 0
    elif auto_background_params is not None:
    # automatic background detection [NOT TYPICALLY USED IN TRAINING TO ENSURE RELIABILITY OF TRAINING INPUTS]
        print("Detecting background regions automatically...")
        mask = automatic_background_detection(img_bl, auto_background_params)
        # Check if enough background was detected
        background_fraction = np.sum(mask == 0) / mask.size
        if background_fraction < auto_background_params.get('min_background_fraction', 0.3):
            print(f"Warning: Only {background_fraction:.1%} of image detected as background.")
            print("Consider adjusting auto_background_params if results are poor.")
        else:
            print(f"Detected {background_fraction:.1%} of image as background.")
    else:
        # No background correction
        mask = np.ones(img_bl[:,:,0].shape)


    
    # Background fitting and subtraction
    print("Fitting and subtracting background...")
    y_dim, x_dim, _ = img_bl.shape
    R = img_bl[:,:,0].flatten()
    G = img_bl[:,:,1].flatten()
    B = img_bl[:,:,2].flatten()
    X_, Y_ = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
    X = X_.flatten()
    Y = Y_.flatten()
    
    # Find substrate locations (masked regions)
    sub_loc = ((mask.flatten()) == 0).nonzero()[0]
    if len(sub_loc) > 0:
        Rsub = R[sub_loc]
        Gsub = G[sub_loc]
        Bsub = B[sub_loc]
        Xsub = X[sub_loc]
        Ysub = Y[sub_loc]
        
        # Fit polynomial background
        Asub = np.array([*line2d(Xsub, Ysub, return_coeff=True)]).T
        
        Rcop, _, _, _ = np.linalg.lstsq(Asub, Rsub, rcond=None)
        Gcop, _, _, _ = np.linalg.lstsq(Asub, Gsub, rcond=None)
        Bcop, _, _, _ = np.linalg.lstsq(Asub, Bsub, rcond=None)
        
        # Apply background correction
        Rfitp = line2d(X, Y, coeffs=[*Rcop])
        Gfitp = line2d(X, Y, coeffs=[*Gcop])
        Bfitp = line2d(X, Y, coeffs=[*Bcop])
        
        img_poly = np.dstack([(R-Rfitp+1).reshape(y_dim,x_dim)/2,
                              (G-Gfitp+1).reshape(y_dim,x_dim)/2,
                              (B-Bfitp+1).reshape(y_dim,x_dim)/2])
    else:
        print("No background detected, skipping background subtraction")
        img_poly = img_bl
    
    # Show background reduction if plotting enabled
    if show_plot:
        print('Manually inspect background reduction, then close figures.')
        plt.figure()
        plt.imshow(img_poly)
        plt.title('Background-corrected image')
        plt.figure()
        plt.imshow(mask)
        plt.title('Mask (black = background regions)')
        plt.show()
    
    # Second round of bilateral filtering
    print("Applying second round of bilateral filtering...")
    img_bl2 = img_poly.astype(np.float32)
    for ii in range(3):
        img_bl2 = cv2.bilateralFilter(img_bl2, 2, 0.5, 1)
    
    # Optional Image compression [NOT TYPICALLY USED IN TRAINING TO ENSURE RELIABILITY OF TRAINING INPUTS]
    if comp_rate is not None:
        img_size = img_bl2.shape
        comp = int(((img_size[0] * img_size[1]) ** 0.5) / comp_rate)
        comp = max(1, comp)
        img_proc = img_bl2[0::comp, 0::comp]
        print(f"Compressed image shape: {img_proc.shape}, compression factor: {comp}")
    else:
        img_proc = img_bl2  # No compression
        comp = 1
        print("No compression applied - using full resolution")
    
    # Convert to CuPy for GPU processing
    image_np = img_proc
    image_size = img_proc.shape
    image = cp.array(img_proc)
    
    # Extract RGB channels and remove black/invalid pixels
    R = image[:,:,0].flatten()
    G = image[:,:,1].flatten()
    B = image[:,:,2].flatten()
    
    # Remove pixels that are too dark or too bright (likely artifacts)
    valid_pixels = (R > 0.01) & (G > 0.01) & (B > 0.01) & (R < 0.99) & (G < 0.99) & (B < 0.99)
    R, G, B = R[valid_pixels], G[valid_pixels], B[valid_pixels]
    
    pixels = cp.vstack([B, G, R]).T
    pixel_count = pixels.shape[0]
    print(f"Processing {pixel_count} valid pixels")
    
    if pixel_count == 0:
        raise ValueError("No valid pixels found after preprocessing. Check crop and masking parameters.")
    


    if initialization_method == 'meanshift':
        print("Initializing mean-shift clustering...")
        
        # Calculate epsilon for mean-shift
        data_range = {
            'B_min': cp.min(B), 'B_max': cp.max(B),
            'G_min': cp.min(G), 'G_max': cp.max(G),
            'R_min': cp.min(R), 'R_max': cp.max(R)
        }
        
        BGRmin = cp.array([data_range['B_min'], data_range['G_min'], data_range['R_min']])
        BGRmax = cp.array([data_range['B_max'], data_range['G_max'], data_range['R_max']])
        diag = cp.sqrt(cp.sum((BGRmax - BGRmin)**2)) / (density - 1)
        epsilon = diag / 3
        
        # Run mean-shift with proper outlier removal
        meanshift_means = meanshift_initialization(
            pixels, 
            density=density,
            outlier_fraction=outlier_fraction
        )
        
        print(f"Mean-shift found {len(meanshift_means)} mean points")
        
        if len(meanshift_means) > max_clusters:
            print(f"Too many mean points ({len(meanshift_means)}), applying DBSCAN consolidation...")
            means = dbscan_consolidation(
                meanshift_means, 
                original_epsilon=epsilon,
                eps_factor=dbscan_eps_factor
            )
        else:
            means = meanshift_means
        
        # If still too many, use more aggressive consolidation
        if len(means) > max_clusters * 1.5:
            print(f"Still too many clusters ({len(means)}), applying aggressive consolidation...")
            means = dbscan_consolidation(
                means,
                original_epsilon=epsilon,
                eps_factor=dbscan_eps_factor * 2  # Double the epsilon
            )
    else:  # DEPRECTATED - option to use kmeans clustering to test against mean-shift clustering
        print("Initializing K-means clustering...")
        kmeans_means = kmeans_initialization(pixels)
        print("Grouping similar means...")
        means = group_similar_means(kmeans_means)
    
    # Adjust cluster count 
    data_range = {
        'B_min': cp.min(B), 'B_max': cp.max(B),
        'G_min': cp.min(G), 'G_max': cp.max(G),
        'R_min': cp.min(R), 'R_max': cp.max(R)
    }
    means = adjust_cluster_count(means, min_clusters, max_clusters, data_range)
    cluster_count = len(means)
    print(f"Fitting to {cluster_count} clusters")
    
    # Initialize covariance matrices 
    print("Initializing covariance matrices...")
    covariances = cp.zeros((cluster_count, 3, 3))
    
    with tqdm(total=cluster_count, desc="Creating covariance matrices", unit="cluster") as pbar:
        for kk in range(cluster_count):
            diff_vectors = pixels - means[kk]
            cov_matrix = cp.dot(diff_vectors.T, diff_vectors) / pixel_count
            covariances[kk] = cov_matrix + cp.eye(3) * 1e-6
            pbar.update(1)
    
    # Initialize weights 
    weights = cp.full(cluster_count, 1/cluster_count)
    weight_differ = weights.copy()
    
    # Expectation Maximization algorithm 
    print("Starting GMM Expectation-Maximization...")
    max_cycles = 1001
    
    with tqdm(total=max_cycles, desc="EM algorithm", unit="cycle") as pbar:
        for cycle in range(max_cycles):
            # Expectation step
            if pixel_count > batch_size:
                cluster_probs = process_pixels_in_batches(pixels, means, covariances, weights, batch_size)
            else:
                cluster_probs = multivariate_gaussian_training(pixels, means, covariances, weights)
            
            if cycle == max_cycles - 1:
                pbar.update(1)
                break
            
            # Maximization step
            weights = cp.mean(cluster_probs, axis=0)
            weights = cp.maximum(weights, 1e-10)
            weights = weights / cp.sum(weights)
            
            # Update means
            for kk in range(cluster_count):
                weight_sum = cp.sum(cluster_probs[:, kk])
                if weight_sum > 1e-10:
                    means[kk] = cp.sum(cluster_probs[:, kk:kk+1] * pixels, axis=0) / weight_sum
            
            # Update covariances
            covariances = covariance_update_training(pixels, cluster_probs, means)
            
            pbar.update(1)
            
            # Check for convergence
            if cycle % 100 == 0:
                if cp.all(cp.abs(weights - weight_differ) < convergence_param):
                    pbar.set_description(f"EM algorithm (converged at cycle {cycle})")
                    pbar.close()
                    print(f"Optimization achieved after {cycle} cycles")
                    break
            
            # Check for timeout
            tac = time.perf_counter()
            if (tac - tic) > 5*60*60:
                pbar.set_description("EM algorithm (timeout)")
                pbar.close()
                print("TIMEOUT: Longer than 5 hours.")
                break
            
            weight_differ = weights.copy()
    
    print("GMM training completed.")
    
    # Final cluster assignments 
    print("Assigning pixels to clusters...")
    nearest_cluster = cp.argmax(cluster_probs, axis=1)
    
    # Convert results to CPU 
    print("Converting results to CPU...")
    results_cpu = {
        'nearest_cluster': cp.asnumpy(nearest_cluster),
        'B': cp.asnumpy(B),
        'G': cp.asnumpy(G),
        'R': cp.asnumpy(R),
        'weights': cp.asnumpy(weights),
        'means': cp.asnumpy(means),
        'covariances': cp.asnumpy(covariances),
        'valid_pixels': cp.asnumpy(valid_pixels)
    }
    
    # Create layer image (adapted for compressed/processed image)
    full_nearest_cluster = np.full(image_size[0] * image_size[1], -1, dtype=int)  # -1 for invalid pixels
    full_nearest_cluster[results_cpu['valid_pixels']] = results_cpu['nearest_cluster']
    layer_image = full_nearest_cluster.reshape(image_size[0], image_size[1])
    
    # Create ellipsoids 
    print("Creating confidence ellipsoids...")
    ellipsoids = {}
    with tqdm(total=cluster_count, desc="Creating ellipsoids", unit="cluster") as pbar:
        for kk in range(cluster_count):
            ellipsoids[f'{kk}'] = confellipsoid(
                results_cpu['weights'][kk],
                results_cpu['means'][kk, 0],
                results_cpu['means'][kk, 1],
                results_cpu['means'][kk, 2],
                results_cpu['covariances'][kk]
            )
            pbar.update(1)
    
    # Generate colors 
    cmap = mpl.colormaps.get_cmap('viridis')
    cluster_colors = [cmap(ii/cluster_count) for ii in range(cluster_count)]
    elli_color = [cmap(ii/cluster_count, alpha=0.25) for ii in range(cluster_count)]
    
    # Create results dictionary with preprocessing info
    results_dict = {
        flake_name + ' weights': results_cpu['weights'],
        flake_name + ' blue means': results_cpu['means'][:, 0],
        flake_name + ' green means': results_cpu['means'][:, 1],
        flake_name + ' red means': results_cpu['means'][:, 2],
        flake_name + ' covariance': results_cpu['covariances'],
        flake_name + ' layer_image': layer_image,
        flake_name + ' B': results_cpu['B'],
        flake_name + ' G': results_cpu['G'],
        flake_name + ' R': results_cpu['R'],
        flake_name + ' nearest_cluster': results_cpu['nearest_cluster'],
        flake_name + ' image_size': image_size,
        flake_name + ' ellipsoids': ellipsoids,
        flake_name + ' elli_color': elli_color,
        flake_name + ' cluster_colors': cluster_colors,
        flake_name + ' original_image': original_image,
        flake_name + ' processed_image': image_np,
        flake_name + ' crop': crop,
        flake_name + ' compression_factor': comp,
        flake_name + ' mask': mask if masking is not None else None
    }
    
    toc = time.perf_counter()
    print(f"Total runtime: {toc-tic:.2f} seconds")
    
    # Create visualizations 
    if show_plot:
        print("Creating visualizations...")
        training_visualizations(results_dict, flake_name, show_plot=True)
    else:
        print("Saving visualization to file...")
        training_visualizations(results_dict, flake_name, show_plot=False)
        
    # Add initialization method info to results
    results_dict[flake_name + ' initialization_method'] = initialization_method
    if initialization_method == 'meanshift':
        results_dict[flake_name + ' density'] = density

    return results_dict


def process_single_image(img_config, base_params, show_plot=True):
    """
    Process a single image 
    
    Batch processing calls this function multiple times
    
    Parameters:
    -----------
    img_config : dict
        Dictionary containing image-specific parameters:
        - img_file: str - Path to the image file
        - flake_name: str - Name identifier for the flake
        - crop: list, optional - Crop region [y1, y2, x1, x2]
        - masking: list of lists, optional - Background regions [[y1, y2, x1, x2], ...]
    base_params : dict
        Dictionary containing base processing parameters:
        - min_clusters: int - Minimum number of clusters (default: 8)
        - max_clusters: int - Maximum number of clusters (default: 8)
        - convergence_param: float - Convergence threshold (default: 1e-6)
        - comp_rate: int, optional - Compression factor
        - batch_size: int - Batch size for GPU processing (default: 50000)
        - initialization_method: str - 'meanshift' or 'kmeans' (default: 'meanshift')
        - density: int - For mean-shift density^3 gives initial mean points (default: 8)
    show_plot : bool
        Whether to display interactive plots.
        
    Returns:
    --------
    dict
        Clustering results dictionary containing all training outputs.
    """
    # Extract parameters for this image
    img_file = img_config['img_file']
    flake_name = img_config['flake_name']
    crop_region = img_config.get('crop', None)
    masking_regions = img_config.get('masking', None)
    
    # Combine base params with image-specific settings
    params = base_params.copy()
    params.update({
        'img_file': img_file,
        'flake_name': flake_name,
        'crop': crop_region,
        'masking': masking_regions,
        'show_plot': show_plot
    })
    
    # Set default values and filter parameters for the training function
    training_params = {
        'img_file': params.get('img_file'),
        'flake_name': params.get('flake_name'),
        'crop': params.get('crop'),
        'masking': params.get('masking'),
        'min_clusters': params.get('min_clusters', 8),
        'max_clusters': params.get('max_clusters', 8),
        'convergence_param': params.get('convergence_param', 1e-6),
        'comp_rate': params.get('comp_rate'),
        'show_plot': params.get('show_plot', True),
        'batch_size': params.get('batch_size', 50000),
        'initialization_method': params.get('initialization_method', 'meanshift'),
        'density': params.get('density', 8)
    }
    
    
    # Print processing information
    print(f"Processing: {flake_name}")
    print(f"Image file: {img_file}")
    print(f"Crop region: {crop_region}")
    print(f"Masking regions: {masking_regions}")
    print(f"Compression rate: {training_params['comp_rate']}")
    print(f"Initialization method: {training_params['initialization_method']}")
    print(f"Using {training_params['initialization_method']} initialization method")
    
    try:
        
        # Run clustering
        print("Running enhanced clustering...")
        results = training(**training_params)
        
        print(f"✓ Clustering completed for {flake_name}")
        return results
        
    except Exception as e:
        print(f"✗ Error processing {flake_name}: {e}")
        raise e


def batch_processing():
    """
    Batch processing from json file
    """
    ## Load batch data from JSON file
    try:
        image_list, base_params = load_batch_data("batch_data.json")
        

        # Below is a fail state for JSON file issues
        # this will offer to create a sample JSON
        if not image_list or len(image_list) == 0:
            print("\n❌ No images found in batch_data.json!")
            print("\nTo use batch processing, you need to create a 'batch_data.json' file with your image configurations.")
            print("\nExample batch_data.json structure:")
            print("""
{
    "base_parameters": {
        "min_clusters": 8,
        "max_clusters": 8,
        "convergence_param": 1e-6,
        "comp_rate": 200,
        "batch_size": 50000,
        "initialization_method": "meanshift",
        "density": 8
    },
    "images": [
        {
            "img_file": "./image_data/training/Fig_1a.jpg",
            "flake_name": "Sample_1",
            "crop": [1100, 1850, 2300, 3300],
            "masking": [[0, 300, 0, 200], [650, -1, 400, -1]]
        },
        {
            "img_file": "./image_data/training/Fig_1b.jpg",
            "flake_name": "Sample_2",
            "crop": [800, 1600, 1000, 2000],
            "masking": [[0, 200, 0, 300]]
        }
    ]
}""")
            print("\nWould you like to:")
            print("1. Create batch_data.json with single image (current main() parameters)")
            print("2. Return to main menu", flush=True)
            
            print("Enter your choice (1-2): ", end="", flush=True)
            choice = input().strip()
            
            if choice == "1":
                create_sample_batch_file()
                print("\n✓ Created sample batch_data.json file!")
                print("You can now edit this file with your image parameters and run batch processing again.")
                return
            else:
                print("Returning to main menu...")
                return
        

        if 'comp_rate' not in base_params:
            base_params['comp_rate'] = 100  # Default compression rate
        
        print("Successfully loaded batch data!")
        print(f"Images to process: {[img['flake_name'] for img in image_list]}")
        print(f"Compression rate: {base_params.get('comp_rate', 'None')}")
        
    except FileNotFoundError:
        print("\n❌ batch_data.json file not found!")
        print("\nTo use batch processing, you need to create a 'batch_data.json' file.")
        print("Would you like to create a sample batch_data.json file? (y/n): ", end="", flush=True)
        
        create_file = input().strip().lower()
        if create_file in ['y', 'yes']:
            create_sample_batch_file()
            print("\n✓ Created sample batch_data.json file!")
            print("Please edit this file with your image parameters and run batch processing again.")
        else:
            print("Returning to main menu...")
        return
        
    except Exception as e:
        print(f"\n❌ Error loading batch data: {e}")
        print("Please check your batch_data.json file format.")
        return
    
    ## PHASE 1: Automatic processing of all images
    print("\n" + "="*60)
    print("PHASE 1: AUTOMATIC CLUSTERING OF ALL IMAGES")
    print("="*60)
    
    all_results = {}  # Store results for each flake
    
    for i, img_config in enumerate(image_list):
        print(f"\n--- Processing Image {i+1}/{len(image_list)}: {img_config['flake_name']} ---")
        
        try:
            # Process using the unified method
            results = process_single_image(
                img_config=img_config,
                base_params=base_params,
                show_plot=False  # Don't show interactive plots during batch processing
            )
            
            # Store results for later review
            all_results[img_config['flake_name']] = results
            
        except Exception as e:
            print(f"✗ Error processing {img_config['flake_name']}: {e}")
            continue
    
    print(f"\n✓ Automatic processing complete! Successfully processed {len(all_results)} images.")
    
    # PHASE 2: Manual review and thickness assignment
    print("\n" + "="*60)
    print("PHASE 2: MANUAL REVIEW AND THICKNESS ASSIGNMENT")
    print("="*60)
    print("Now you can review each image's clustering results and assign thickness labels.", flush=True)
    
    # Keep track of which flakes have been reviewed
    reviewed_flakes = set()
    
    while len(reviewed_flakes) < len(all_results):
        # Show available flakes for review 
        print(f"\nAvailable flakes for review:")
        unreviewed_flakes = [name for name in all_results.keys() 
                           if name not in reviewed_flakes]
        
        # Refresh numbering each time menu is shown
        for i, flake_name in enumerate(unreviewed_flakes):
            print(f"  {i+1}. {flake_name}")
        
        # Show summary of reviewed flakes
        if reviewed_flakes:
            print(f"\nAlready reviewed ({len(reviewed_flakes)} flakes):")
            for flake_name in sorted(reviewed_flakes):
                print(f"  ✓ {flake_name}")
        
        # Get user choice
        print("\nOptions:")
        print(f"  - Enter number (1-{len(unreviewed_flakes)}) to review a specific flake")
        print("  - Enter 'auto' to review all remaining flakes in order")
        print("  - Enter 'skip' to skip remaining reviews and save current results")
        print("  - Enter 'quit' to exit without saving", flush=True)
        
        print("Your choice: ", end="", flush=True)
        choice = input().strip().lower()
        
        if choice == 'quit':
            print("Exiting without saving remaining results.")
            break
            
        elif choice == 'skip':
            print("Skipping remaining reviews. Saving current results...")
            break
            
        elif choice == 'auto':
            # Will automatically prompt user to review each flake sequentially
            for flake_name in unreviewed_flakes:
                print(f"\n{'='*50}")
                print(f"Auto-reviewing: {flake_name}")
                print('='*50)
                
                try:
                    # Review and edit results
                    edited_results = review_and_edit_results(all_results[flake_name], flake_name)
                    
                    # Save the results
                    out_file = f'./outputs/training/catalogs/{flake_name}.npz'
                    final_data = save_clustering_results(edited_results, out_file)
                    
                    reviewed_flakes.add(flake_name)
                    print(f"✓ {flake_name} reviewed and saved to {out_file}")
                    
                except Exception as e:
                    print(f"✗ Error reviewing {flake_name}: {e}")
                    continue
            
        else:
            # Try to parse as a number for specific flake selection
            try:
                flake_index = int(choice) - 1
                if 0 <= flake_index < len(unreviewed_flakes):
                    flake_name = unreviewed_flakes[flake_index]  # Use the current unreviewed list
                    
                    print(f"\n{'='*50}")
                    print(f"Reviewing: {flake_name}")
                    print('='*50)
                    
                    try:
                        # Review and edit results
                        edited_results = review_and_edit_results(all_results[flake_name], flake_name)
                        
                        # Save the results  
                        out_file = f'./outputs/training/catalogs/{flake_name}.npz'
                        final_data = save_clustering_results(edited_results, out_file)
                        
                        reviewed_flakes.add(flake_name)
                        print(f"✓ {flake_name} reviewed and saved to {out_file}")
                        
                    except Exception as e:
                        print(f"✗ Error reviewing {flake_name}: {e}")
                        
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(unreviewed_flakes)}.")
                    
            except ValueError:
                print("Invalid input. Please enter a number, 'auto', 'skip', or 'quit'.")
    
    # Print final summary of batch processing
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print('='*60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Images reviewed: {len(reviewed_flakes)}")
    
    if reviewed_flakes:
        print("Reviewed flakes:")
        for flake_name in reviewed_flakes:
            print(f"  ✓ {flake_name}")
    
    unreviewed = set(all_results.keys()) - reviewed_flakes
    if unreviewed:
        print("Unreviewed flakes (results saved without thickness assignment):")
        for flake_name in unreviewed:
            # Save unreviewed results without thickness assignment
            out_file = f'./outputs/training/catalogs/{flake_name}.npz'
            save_clustering_results(all_results[flake_name], out_file)
            print(f"  ○ {flake_name} -> {out_file}")
    
    print("\nBatch processing complete!")