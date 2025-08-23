import numpy as np
import os
import json
import csv
from pathlib import Path
import glob
from .visualization import training_visualizations, cluster_review_interface, thickness_visualization



def npz2dict(file_location):
    """Converts data stored in an npz zip file into
    a dictionary."""
    dict = {}
    with np.load(file_location, allow_pickle=True) as file_zip:
        for key in file_zip:
            dict[key] = file_zip[key]
    return dict

def consolidate_json_to_csv(input_dir, output_dir):
    """
    Consolidates JSON files from input directory into a single CSV file.
    
    Args:
        input_dir (str): Path to directory containing JSON files
        output_dir (str): Path to directory where CSV file will be saved
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files from input directory
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    # Prepare CSV data
    csv_data = []
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create row for CSV
            row = {
                'filename': json_file.name,
                'flake_name': data.get('flake_name', ''),
                'total_pixels': data.get('total_pixels', 0),
                'valid_pixels': data.get('valid_pixels', 0),
                'invalid_pixels': data.get('invalid_pixels', 0),
                'background_pixels': data.get('background_pixels', 0),
                'background_fraction': data.get('background_fraction', 0),
                'compression_factor': data.get('compression_factor', 0),
                'runtime_seconds': data.get('runtime_seconds', 0)
            }
            
            # Add layer information
            layers = data.get('layers', {})
            for layer_name, layer_data in layers.items():
                row[f'{layer_name}_pixel_count'] = layer_data.get('pixel_count', 0)
                row[f'{layer_name}_percentage'] = layer_data.get('percentage', 0)
            
            csv_data.append(row)
            
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not csv_data:
        print("No valid JSON data to process")
        return
    
    # Get all unique column names from all rows
    all_columns = set()
    for row in csv_data:
        all_columns.update(row.keys())
    
    # Sort columns for consistent output
    base_columns = ['filename', 'flake_name', 'total_pixels', 'valid_pixels', 
                   'invalid_pixels', 'background_pixels', 'background_fraction', 
                   'compression_factor', 'runtime_seconds']
    
    # Get layer columns and sort them
    layer_columns = sorted([col for col in all_columns if col not in base_columns])
    ordered_columns = base_columns + layer_columns
    
    # Write to CSV
    output_file = Path(output_dir) / 'consolidated_data.csv'
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ordered_columns)
        writer.writeheader()
        
        for row in csv_data:
            # Fill missing columns with empty values
            complete_row = {col: row.get(col, '') for col in ordered_columns}
            writer.writerow(complete_row)
    
    print(f"Successfully consolidated {len(csv_data)} JSON files into {output_file}")
    print(f"CSV contains {len(ordered_columns)} columns")


def load_batch_data(config_file="batch_data.json"):
    """
    Load batch processing configuration from JSON file.
    
    Parameters:
    -----------
    config_file : str, optional
        Path to the JSON configuration file. Default is "batch_data.json".
        
    Returns:
    --------
    tuple
        (image_list, base_params) - Returns the image list and base parameters
        from the configuration file.
        
    Raises:
    -------
    FileNotFoundError
        If the configuration file doesn't exist.
    json.JSONDecodeError
        If the JSON file has invalid syntax.
    KeyError
        If required fields are missing from the configuration.
    """
    try:
        # Check if file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
        
        # Load and parse JSON
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract required data
        base_params = config['base_params']
        image_list = config['images']
        
        # Validate and set default values for new parameters
        if 'initialization_method' not in base_params:
            base_params['initialization_method'] = 'meanshift'
            print("No initialization_method specified, defaulting to 'meanshift'")
        elif base_params['initialization_method'] not in ['meanshift', 'kmeans']:
            print(f"Invalid initialization_method '{base_params['initialization_method']}', defaulting to 'meanshift'")
            base_params['initialization_method'] = 'meanshift'
        
        if 'density' not in base_params:
            base_params['density'] = 8
            print("No density specified for mean-shift, defaulting to 8")
        
        if 'batch_size' not in base_params:
            base_params['batch_size'] = 50000
            print("No batch_size specified, defaulting to 50000")
        
        print(f"Loaded configuration from: {config_file}")
        print(f"Number of images: {len(image_list)}")
        print(f"Initialization method: {base_params['initialization_method']}")
        print(f"Base parameters: {base_params}")
        
        return image_list, base_params
        
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON syntax in '{config_file}': {str(e)}", e.doc, e.pos)
    except KeyError as e:
        raise KeyError(f"Missing required field in configuration: {e}")
    except Exception as e:
        raise Exception(f"Error loading batch data: {e}")



def save_clustering_results(results_dict, out_file='./Monolayer Search/Graphene_on_SiO2_catalog.npz', 
                           merge_with_existing=True):
    """

    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results from training() function.
    out_file : str
        Path to the output file.
    merge_with_existing : bool
        Whether to merge with existing file data or overwrite.
        
    Returns:
    --------
    combined_dict : dict
        The final combined dictionary that was saved.
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(out_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created directory: {output_dir}')
    
    combined_dict = {}
    
    if merge_with_existing:
        try:
            combined_dict = npz2dict(out_file)
            print(f"Loaded existing data from {out_file}")
        except FileNotFoundError:
            print(f'Created new file {out_file}')
    
    # Add new results to the combined dictionary
    combined_dict.update(results_dict)
    
    # Extract clustering method info if available
    for key in results_dict.keys():
        if ' initialization_method' in key:
            flake_name = key.replace(' initialization_method', '')
            method = results_dict[key]
            density = results_dict.get(flake_name + ' density', 'N/A')
            print(f"Saving {flake_name}: {method} initialization (density={density})")
    
    # Save data 
    save_dict = {}
    for key, value in combined_dict.items():
        try:
            # Test if the value can be saved to npz
            np.array([value])
            save_dict[key] = value
        except Exception as e:
            print(f"Warning: Could not save '{key}': {e}")
            
            if 'ellipsoids' in key or 'elli_color' in key or 'cluster_colors' in key:
                print(f"Skipping complex object: {key}")
            else:
                save_dict[key] = value
    
    np.savez(out_file, **save_dict)
    print(f"Results saved to {out_file}")
    
    return combined_dict


def review_and_edit_results(results_dict, flake_name):
    """
    Function to review clustering results and assign thickness labels to clusters.
    Now includes functionality to redo assignments if desired.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results.
    flake_name : str
        Name of the flake being processed.
        
    Returns:
    --------
    edited_results : dict
        Results dictionary with thickness assignments.
    """
    
    print(f"\n=== Reviewing Results for {flake_name} ===")
    
    # Extract key information
    weights = results_dict[flake_name + ' weights']
    num_clusters = len(weights)
    nearest_cluster = results_dict[flake_name + ' nearest_cluster']
    
    print(f"Number of clusters found: {num_clusters}")
    print(f"Cluster weights: {weights}")
    
    # Show cluster statistics
    for i in range(num_clusters):
        pixel_count = np.sum(nearest_cluster == i)
        percentage = (pixel_count / len(nearest_cluster)) * 100
        print(f"Cluster {i}: Weight = {weights[i]:.4f}, Pixels = {pixel_count:,} ({percentage:.1f}%)")
    
    # Interactive cluster review interface
    print("\n" + "="*60)
    print("CLUSTER THICKNESS ASSIGNMENT")
    print("="*60)
    print("Please examine the cluster visualization and assign thickness values.")
    print("The visualization shows:")
    print("  - Original and processed images")
    print("  - Cluster assignments with color coding")
    print("  - Individual cluster masks")
    print("  - RGB distribution and statistics")
    
    # Main assignment loop - allows for redoing assignments
    thickness_assignments = None
    assignment_round = 1
    
    while True:
        print(f"\n{'='*40}")
        if assignment_round == 1:
            print("INITIAL THICKNESS ASSIGNMENT")
        else:
            print(f"THICKNESS ASSIGNMENT - ROUND {assignment_round}")
        print(f"{'='*40}", flush=True)
        
        # Display the interactive review interface
        thickness_assignments = cluster_review_interface(results_dict, flake_name)
        
        # If user cancels assignments, return original results
        if not thickness_assignments:
            print("No thickness assignments made. Returning original results.")
            return results_dict
        
        # Show assignment summary
        print(f"\nFinal Assignment Summary (Round {assignment_round}):")
        print("-" * 50)
        for cluster_id, thickness in enumerate(thickness_assignments):
            pixel_count = np.sum(nearest_cluster == cluster_id)
            percentage = (pixel_count / len(nearest_cluster)) * 100
            if thickness == 0:
                thickness_label = "Background/Substrate"
            elif thickness == -1:
                thickness_label = "Exclude from analysis"
            else:
                thickness_label = f"{thickness} layer(s)"
            
            print(f"Cluster {cluster_id}: {thickness_label}")
            print(f"  └─ {pixel_count:,} pixels ({percentage:.1f}%)")
        
        # Temporarily apply thickness assignments to create preview visualization
        print(f"\n{'='*50}")
        print("GENERATING PREVIEW VISUALIZATION")
        print("="*50)
        print("Creating thickness visualization with your assignments...")
        
        # Create a temporary copy of results_dict with thickness assignments
        temp_results = results_dict.copy()
        temp_results[flake_name + ' cluster_thicknesses'] = np.array(thickness_assignments)
        
        # Generate the thickness visualization
        thickness_visualization(temp_results, flake_name)
        print("✓ Preview visualization complete!")
        
        # Ask if user wants to redo assignments
        print(f"\n{'='*60}")
        print("ASSIGNMENT REVIEW")
        print(f"{'='*60}")
        print("Are you satisfied with these thickness assignments?")
        print("Please review the thickness visualization above.")
        print("Options:")
        print("  y/yes  - Accept assignments and continue")
        print("  n/no   - Redo thickness assignments") 
        print("  c/cancel - Cancel and return original results", flush=True)
        
        while True:
            choice = input("Your choice (y/n/c): ").strip().lower()
            
            if choice in ['y', 'yes']:
                print("✓ Assignments accepted. Proceeding with analysis...")
                break
            elif choice in ['n', 'no']:
                print(f"↻ Redoing assignments (Round {assignment_round + 1})...")
                assignment_round += 1
                break
            elif choice in ['c', 'cancel']:
                print("✗ Assignments cancelled. Returning original results.")
                return results_dict
            else:
                print("Please enter 'y' (accept), 'n' (redo), or 'c' (cancel)")
        
        # If user accepted assignments, break out of the main loop
        if choice in ['y', 'yes']:
            break
    
    # Apply thickness assignments to results (only after user accepts them)
    if thickness_assignments:
        results_dict[flake_name + ' cluster_thicknesses'] = np.array(thickness_assignments)
        
        print(f"\n✓ Thickness assignments applied to results.")
        
        # Optional: Filter out excluded clusters
        excluded_clusters = [i for i, t in enumerate(thickness_assignments) if t == -1]
        if excluded_clusters:
            print(f"\nExcluded clusters found: {excluded_clusters}")
            print("Would you like to remove these excluded clusters from the analysis?")
            print("This will permanently remove them from the dataset.", flush=True)
            
            while True:
                remove_choice = input("Remove excluded clusters? (y/n): ").strip().lower()
                if remove_choice in ['y', 'yes']:
                    keep_indices = [i for i, t in enumerate(thickness_assignments) if t != -1]
                    results_dict = filter_clusters(results_dict, flake_name, keep_indices)
                    print(f"✓ Removed {len(excluded_clusters)} excluded clusters.")
                    
                    # Update thickness assignments after filtering
                    new_thickness_assignments = [thickness_assignments[i] for i in keep_indices]
                    results_dict[flake_name + ' cluster_thicknesses'] = np.array(new_thickness_assignments)
                    break
                elif remove_choice in ['n', 'no']:
                    print("✓ Keeping excluded clusters in dataset (marked as thickness -1).")
                    break
                else:
                    print("Please enter 'y' (yes) or 'n' (no)")
        
        print(f"\n✓ Processing complete for {flake_name}!")
    
    return results_dict


def filter_clusters(results_dict, flake_name, keep_indices):
    """
    Filter clustering results to keep only specified cluster indices.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results.
    flake_name : str
        Name of the flake being processed.
    keep_indices : list
        List of cluster indices to keep.
        
    Returns:
    --------
    filtered_results : dict
        Results dictionary with filtered clusters.
    """
    
    # Create mapping from old indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    
    # Filter arrays
    filtered_results = results_dict.copy()
    
    # Filter cluster-specific data
    filtered_results[flake_name + ' weights'] = results_dict[flake_name + ' weights'][keep_indices]
    filtered_results[flake_name + ' blue means'] = results_dict[flake_name + ' blue means'][keep_indices]
    filtered_results[flake_name + ' green means'] = results_dict[flake_name + ' green means'][keep_indices]
    filtered_results[flake_name + ' red means'] = results_dict[flake_name + ' red means'][keep_indices]
    filtered_results[flake_name + ' covariance'] = results_dict[flake_name + ' covariance'][keep_indices]
    
    # Filter colors
    cluster_colors = [results_dict[flake_name + ' cluster_colors'][i] for i in keep_indices]
    elli_color = [results_dict[flake_name + ' elli_color'][i] for i in keep_indices]
    filtered_results[flake_name + ' cluster_colors'] = cluster_colors
    filtered_results[flake_name + ' elli_color'] = elli_color
    
    # Filter ellipsoids
    ellipsoids = {}
    for new_idx, old_idx in enumerate(keep_indices):
        ellipsoids[str(new_idx)] = results_dict[flake_name + ' ellipsoids'][str(old_idx)]
    filtered_results[flake_name + ' ellipsoids'] = ellipsoids
    
    # Update cluster assignments for pixels
    nearest_cluster = results_dict[flake_name + ' nearest_cluster'].copy()
    layer_image = results_dict[flake_name + ' layer_image'].copy()
    
    # Reassign cluster labels
    new_nearest_cluster = np.full_like(nearest_cluster, -1)
    new_layer_image = np.full_like(layer_image, -1)
    
    for old_idx, new_idx in old_to_new.items():
        mask = nearest_cluster == old_idx
        new_nearest_cluster[mask] = new_idx
        
        mask_img = layer_image == old_idx
        new_layer_image[mask_img] = new_idx
    
    # Remove pixels assigned to deleted clusters
    valid_pixels = new_nearest_cluster >= 0
    filtered_results[flake_name + ' nearest_cluster'] = new_nearest_cluster[valid_pixels]
    filtered_results[flake_name + ' B'] = results_dict[flake_name + ' B'][valid_pixels]
    filtered_results[flake_name + ' G'] = results_dict[flake_name + ' G'][valid_pixels]
    filtered_results[flake_name + ' R'] = results_dict[flake_name + ' R'][valid_pixels]
    
    # Set removed clusters to 0 in layer image
    new_layer_image[new_layer_image < 0] = 0
    filtered_results[flake_name + ' layer_image'] = new_layer_image
    
    print(f"Filtered results: {len(keep_indices)} clusters remaining")
    
    # Create new visualizations with filtered results
    training_visualizations(filtered_results, flake_name)
    
    return filtered_results

def create_sample_batch_file():
    """
    Create a sample batch_data.json file with the parameters from main().
    """
    
    sample_data = {
        "base_parameters": {
            "min_clusters": 8,
            "max_clusters": 8,
            "convergence_param": 1e-6,
            "apply_preprocessing": True,
            "comp_rate": 200,
            "batch_size": 50000,
            "initialization_method": "meanshift",
            "density": 8
        },
        "images": [
            {
                "img_file": "./image_data/training/Fig_1a.jpg",
                "flake_name": "KHGR002 3D1",
                "crop": [1100, 1850, 2300, 3300],
                "masking": [[0, 300, 0, 200], [650, -1, 400, -1]]
            }
        ]
    }
    
    try:
        with open("batch_data.json", "w") as f:
            json.dump(sample_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error creating batch_data.json: {e}")
        return False



def save_analysis_data(flake_name, clusters, pixel_counts, total_pixels, 
                      valid_pixel_count, invalid_pixel_count, background_pixel_count,
                      background_fraction, compression_factor, runtime, layer_mapping=None):
    """
    Save analysis data to a text file with proper layer mapping.
    """
    # Create output directory
    output_dir = './outputs/classification_results/data'
    os.makedirs(output_dir, exist_ok=True)
    
    # If no layer_mapping provided, create a simple one
    if layer_mapping is None:
        layer_mapping = {i: i for i in clusters}
    
    # Create data dictionary with Python native types
    data = {
        'flake_name': flake_name,
        'total_pixels': int(total_pixels),
        'valid_pixels': int(valid_pixel_count),
        'invalid_pixels': int(invalid_pixel_count),
        'background_pixels': int(background_pixel_count),
        'background_fraction': float(background_fraction),
        'compression_factor': int(compression_factor),
        'runtime_seconds': float(runtime),
        'layers': {}
    }
    
    # Add layer data with proper naming based on mapping
    for i in clusters:
        count = int(pixel_counts[i])
        percentage = float((count / total_pixels) * 100)
        
        # Use actual layer number or 'residue' from mapping
        layer_label = layer_mapping[i]
        if layer_label == 'residue':
            key_name = 'residue'
            display_name = 'Residue'
        else:
            key_name = f'layer_{layer_label}'
            display_name = f'{layer_label} layer{"s" if layer_label != 1 else ""}'
        
        data['layers'][key_name] = {
            'pixel_count': count,
            'percentage': percentage,
            'display_name': display_name
        }
    
    # Save as JSON 
    json_filename = f'{output_dir}/{flake_name}_data.json'
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save as text file
    txt_filename = f'{output_dir}/{flake_name}_summary.txt'
    with open(txt_filename, 'w') as f:
        f.write(f"Layer Analysis Summary: {flake_name}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Processing Information:\n")
        f.write(f"  Runtime: {runtime:.2f} seconds\n")
        f.write(f"  Compression factor: {compression_factor}\n\n")
        f.write(f"Image Statistics:\n")
        f.write(f"  Total pixels: {total_pixels:,}\n")
        f.write(f"  Valid pixels: {valid_pixel_count:,} ({valid_pixel_count/total_pixels*100:.2f}%)\n")
        f.write(f"  Invalid pixels: {invalid_pixel_count:,} ({invalid_pixel_count/total_pixels*100:.2f}%)\n")
        f.write(f"  Background pixels: {background_pixel_count:,} ({background_pixel_count/total_pixels*100:.2f}%)\n\n")
        f.write(f"Layer Distribution:\n")
        
        # Sort clusters by layer mapping for better readability
        sorted_clusters = sorted(clusters, key=lambda x: (layer_mapping[x] if isinstance(layer_mapping[x], int) else float('inf')))
        
        for i in sorted_clusters:
            count = pixel_counts[i]
            pct = (count / total_pixels) * 100
            layer_label = layer_mapping[i]
            
            if layer_label == 'residue':
                f.write(f"  Residue: {count:,} pixels ({pct:.2f}%)\n")
            else:
                f.write(f"  {layer_label} layer{'s' if layer_label != 1 else ''}: {count:,} pixels ({pct:.2f}%)\n")
    
    print(f"Saved data to {json_filename} and {txt_filename}")


##############################################
# MASTER CATALOGER FUNCTIONS ARE BELOW
##############################################


def create_master_catalog(catalogs_dir, out_file, max_layers=6, include_substrate=True, include_residue=False, residue_value=-1, verbose=True):
    """
    Create a master catalog from multiple catalog files, grouping by actual thickness assignments.
    
    Parameters:
    -----------
    catalogs_dir : str
        Directory containing the .npz catalog files to process
    out_file : str
        Path where the master catalog will be saved
    max_layers : int, optional
        Maximum number of layers to expect (default: 6, looks for 1-6 layers)
    include_substrate : bool, optional
        Whether to include substrate/background (thickness=0) in master catalog (default: False)
    include_residue : bool, optional
        Whether to include residue classification (default: False)
    residue_value : int, optional
        Value used to represent residue in thickness assignments (default: -1)
    verbose : bool, optional
        Whether to print progress messages (default: True)
    
    Returns:
    --------
    dict
        Dictionary containing the master catalog data
    """
    
    # Input validation
    if not isinstance(max_layers, int) or max_layers <= 0:
        raise ValueError("max_layers must be a positive integer")
    
    if not os.path.exists(catalogs_dir):
        raise FileNotFoundError(f"Directory not found: {catalogs_dir}")
    
    # Get all .npz files in the catalogs directory
    catalog_files = glob.glob(os.path.join(catalogs_dir, '*.npz'))
    
    if not catalog_files:
        raise FileNotFoundError(f"No .npz files found in {catalogs_dir}")
    
    if verbose:
        print(f"Found {len(catalog_files)} catalog files:")
        for file in catalog_files:
            print(f"  - {os.path.basename(file)}")
    
    # Initialize dictionaries to categorize the data by thickness
    thickness_categories = []
    if include_substrate:
        thickness_categories.append(0)  # substrate
    for tt in range(1, max_layers + 1):
        thickness_categories.append(tt)
    if include_residue:
        thickness_categories.append('residue')
    
    # Data storage by thickness
    thickness_data = {}
    for category in thickness_categories:
        thickness_data[category] = {
            'weights': [],
            'blue_means': [],
            'green_means': [], 
            'red_means': [],
            'covariances': []
        }
    
    # Process each catalog file
    for catalog_file in catalog_files:
        if verbose:
            print(f"\nProcessing {os.path.basename(catalog_file)}...")
        
        try:
            file_dict = npz2dict(catalog_file)
        except Exception as e:
            if verbose:
                print(f"Error reading {catalog_file}: {e}")
            continue
        
        # Find flake name and data
        flake_name = None
        flake_data = {}
        
        for key in file_dict.keys():
            # Extract flake name from first key
            if flake_name is None:
                space_pos = key.find(' ')
                if space_pos > 0:
                    flake_name = key[:space_pos]
        
        if flake_name is None:
            if verbose:
                print(f"  Could not determine flake name from file keys")
            continue
        
        # Extract required data arrays
        try:
            weights = file_dict[f'{flake_name} weights']
            blue_means = file_dict[f'{flake_name} blue means']
            green_means = file_dict[f'{flake_name} green means']
            red_means = file_dict[f'{flake_name} red means']
            covariances = file_dict[f'{flake_name} covariance']
            cluster_assignments = file_dict[f'{flake_name} nearest_cluster']
            thickness_assignments = file_dict[f'{flake_name} cluster_thicknesses']
            
        except KeyError as e:
            if verbose:
                print(f"  Missing required key: {e}")
            continue
        
        if verbose:
            cluster_to_thickness = {i: thickness_assignments[i] for i in range(len(thickness_assignments))}
            print(f"  Found thickness assignments: {cluster_to_thickness}")
        
        # Process each thickness category
        for category in thickness_categories:
            # Find clusters that correspond to this thickness
            if category == 'residue':
                target_clusters = [i for i, thickness in enumerate(thickness_assignments) if thickness == residue_value]
                category_name = 'residue'
            else:
                target_clusters = [i for i, thickness in enumerate(thickness_assignments) if thickness == category]
                category_name = f'{category} layer{"s" if category != 1 else ""}'
            
            # For each matching cluster, add its data weighted by pixel count
            for cluster_id in target_clusters:
                # Count pixels in this cluster
                pixel_count = np.sum(cluster_assignments == cluster_id)
                
                if pixel_count > 0:
                    if verbose:
                        print(f"  Added cluster {cluster_id} for {category_name}: {pixel_count} pixels")
                    
                    thickness_data[category]['weights'].extend([weights[cluster_id]] * pixel_count)
                    thickness_data[category]['blue_means'].extend([blue_means[cluster_id]] * pixel_count)
                    thickness_data[category]['green_means'].extend([green_means[cluster_id]] * pixel_count)
                    thickness_data[category]['red_means'].extend([red_means[cluster_id]] * pixel_count)
                    thickness_data[category]['covariances'].extend([covariances[cluster_id]] * pixel_count)
        
        valid_categories = sum(1 for cat in thickness_categories if len(thickness_data[cat]['weights']) > 0)
        if verbose:
            print(f"  Successfully processed {valid_categories} thickness categories")
    
    # Create master averages
    master_weights = {}
    master_blue_mean = {}
    master_green_mean = {}
    master_red_mean = {}
    master_cov = {}
    
    categories_with_data = []
    for category in thickness_categories:
        if len(thickness_data[category]['weights']) > 0:
            categories_with_data.append(category)
            
            key_name = 'residue' if category == 'residue' else f'{category}layers'
            
            master_weights[f'weights-{key_name}'] = np.mean(thickness_data[category]['weights'])
            master_blue_mean[f'blue mean-{key_name}'] = np.mean(thickness_data[category]['blue_means'])
            master_green_mean[f'green mean-{key_name}'] = np.mean(thickness_data[category]['green_means'])
            master_red_mean[f'red mean-{key_name}'] = np.mean(thickness_data[category]['red_means'])
            master_cov[f'covariance-{key_name}'] = np.mean(thickness_data[category]['covariances'], axis=0)
            
            if verbose:
                pixel_count = len(thickness_data[category]['weights'])
                thickness_name = 'residue' if category == 'residue' else f'{category} layer{"s" if category != 1 else ""}'
                print(f'\nFinal master data for {thickness_name}: {pixel_count} total pixels averaged')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    # Save the master catalog
    with open(out_file, 'wb') as f:
        np.savez(f, **master_weights, **master_blue_mean, **master_green_mean,
                 **master_red_mean, **master_cov)
    
    if verbose:
        print("\nFinal Results:")
        print("Master weights:", master_weights)
        print("Master blue mean:", master_blue_mean)
        print("Master green mean:", master_green_mean)
        print("Master red mean:", master_red_mean)
        print("Master covariance keys:", list(master_cov.keys()))
        print(f'\n"{out_file}" created successfully.')
        print(f'Combined data from {len(catalog_files)} catalog files.')
        print(f'Categories with data: {categories_with_data}')
    
    # Return summary information
    return {
        'weights': master_weights,
        'blue_mean': master_blue_mean,
        'green_mean': master_green_mean,
        'red_mean': master_red_mean,
        'covariance': master_cov,
        'files_processed': [os.path.basename(f) for f in catalog_files],
        'categories_with_data': categories_with_data
    }




def inspect_catalog_files(catalogs_dir, max_files=3, max_keys_per_file=10):
    """
    Inspect catalog files to understand their structure and key formats.
    """
    
    if not os.path.exists(catalogs_dir):
        raise FileNotFoundError(f"Directory not found: {catalogs_dir}")
    
    catalog_files = glob.glob(os.path.join(catalogs_dir, '*.npz'))
    
    if not catalog_files:
        raise FileNotFoundError(f"No .npz files found in {catalogs_dir}")
    
    inspection_results = {}
    
    for i, catalog_file in enumerate(catalog_files[:max_files]):
        print(f"\n{'='*60}")
        print(f"Inspecting file {i+1}/{min(len(catalog_files), max_files)}: {os.path.basename(catalog_file)}")
        print(f"{'='*60}")
        
        try:
            file_dict = npz2dict(catalog_file)
            
            print(f"Total keys in file: {len(file_dict)}")
            print(f"Sample keys (showing first {max_keys_per_file}):")
            
            keys_list = list(file_dict.keys())[:max_keys_per_file]
            for j, key in enumerate(keys_list):
                print(f"  {j+1:2d}. '{key}'")
                
                # Show data shape/type
                try:
                    data_shape = file_dict[key].shape if hasattr(file_dict[key], 'shape') else 'scalar'
                    data_type = type(file_dict[key]).__name__
                    print(f"      Data: {data_type}, shape: {data_shape}")
                except:
                    print(f"      Data: Could not determine shape/type")
                print()
            
            inspection_results[catalog_file] = {
                'total_keys': len(file_dict),
                'sample_keys': keys_list,
                'file_readable': True
            }
            
        except Exception as e:
            print(f"Error reading file: {e}")
            inspection_results[catalog_file] = {
                'error': str(e),
                'file_readable': False
            }
    
    return inspection_results
