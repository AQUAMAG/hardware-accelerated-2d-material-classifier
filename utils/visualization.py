import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec

def generate_cluster_colors(n_clusters):
    cmap = plt.colormaps['viridis']
    colors = [cmap(i / (n_clusters - 1)) if n_clusters > 1 else cmap(0.5) 
              for i in range(n_clusters)]
    return colors


def get_cluster_colormap(n_clusters):
    """
    Create a discrete colormap with highly contrasting colors for cluster visualization.
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters to create colors for.
        
    Returns:
    --------
    cmap : matplotlib colormap
        Discrete colormap with contrasting colors.
    norm : matplotlib normalization
        Boundary normalization for discrete colors.
    """
    if n_clusters <= 10:
        # Use tab10 for up to 10 clusters 
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif n_clusters <= 20:
        # Use tab20 for up to 20 clusters
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        # If more clusters, combine multiple categorical colormaps
        tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        set3_colors = plt.cm.Set3(np.linspace(0, 1, 12))
        base_colors = np.vstack([tab10_colors, tab20_colors, set3_colors])
    
    # Select the number of colors based on cluster count
    colors = base_colors[:n_clusters]
    
    # Create discrete colormap and normalization
    cmap = ListedColormap(colors)
    bounds = np.arange(n_clusters + 1) - 0.5
    norm = BoundaryNorm(bounds, n_clusters)
    
    return cmap, norm


def training_visualizations(results_dict, flake_name, show_plot=True):
    """
    Create visualizations with color keys and original image display
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results.
    flake_name : str
        Name of the flake being processed.
    show_plot : bool, optional
        Whether to display the plot. Set to False for batch processing.
    """
    # Extract data from results
    B_cpu = results_dict[flake_name + ' B']
    G_cpu = results_dict[flake_name + ' G']
    R_cpu = results_dict[flake_name + ' R']
    nearest_cluster_cpu = results_dict[flake_name + ' nearest_cluster']
    layer_image = results_dict[flake_name + ' layer_image']
    ellipsoids = results_dict[flake_name + ' ellipsoids']
    elli_color = results_dict[flake_name + ' elli_color']
    cluster_colors = results_dict[flake_name + ' cluster_colors']
    weights = results_dict[flake_name + ' weights']
    original_image = results_dict[flake_name + ' original_image']
    processed_image = results_dict[flake_name + ' processed_image']
    crop = results_dict[flake_name + ' crop']
    
    cluster_count = len(weights)
    
    # Get colormap for clusters
    cluster_cmap, cluster_norm = get_cluster_colormap(cluster_count)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Original image
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=14, weight='bold')
    ax1.axis('off')
    
    # Add crop rectangle if cropping was applied
    if crop is not None:
        rect = Rectangle((crop[2], crop[0]), crop[3]-crop[2], crop[1]-crop[0], 
                        linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(crop[2], crop[0]-20, 'Cropped Region', color='red', fontsize=10, weight='bold')
    
    # Processed/cropped image
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(processed_image)
    ax2.set_title('Processed Image', fontsize=14, weight='bold')
    ax2.axis('off')
    
    # Cluster assignments on image 
    ax3 = plt.subplot(2, 4, 3)
    cluster_display = ax3.imshow(layer_image, cmap=cluster_cmap, norm=cluster_norm)
    ax3.set_title('Cluster Assignments', fontsize=14, weight='bold')
    ax3.axis('off')
    
    # Add colorbar 
    cbar = plt.colorbar(cluster_display, ax=ax3, shrink=0.8, ticks=np.arange(cluster_count))
    cbar.set_label('Cluster ID', fontsize=12)
    cbar.ax.set_yticklabels([str(i) for i in range(cluster_count)])
    
    # 3D scatter plot of pixels colored by cluster 
    ax4 = plt.subplot(2, 4, 4, projection='3d')
    scatter = ax4.scatter(B_cpu, G_cpu, R_cpu, c=nearest_cluster_cpu, 
                         s=0.5, marker=',', cmap=cluster_cmap, norm=cluster_norm, alpha=0.6)
    ax4.set_xlabel('Blue', fontsize=12)
    ax4.set_ylabel('Green', fontsize=12)
    ax4.set_zlabel('Red', fontsize=12)
    ax4.set_title('RGB Pixel Clusters', fontsize=14, weight='bold')
    
    # GMM ellipsoids in 3D 
    ax5 = plt.subplot(2, 4, 5, projection='3d')
    for kk in ellipsoids:
        # Use colormap for ellipsoids to match other visualizations
        ellipsoid_color = cluster_cmap(cluster_norm(int(kk)))
        ax5.plot_surface(ellipsoids[kk][:,:,0],
                         ellipsoids[kk][:,:,1],
                         ellipsoids[kk][:,:,2],
                         color=ellipsoid_color,
                         alpha=0.3)
    ax5.set_xlabel('Blue', fontsize=12)
    ax5.set_ylabel('Green', fontsize=12)
    ax5.set_zlabel('Red', fontsize=12)
    ax5.set_title('GMM Confidence Ellipsoids', fontsize=14, weight='bold')
    
    # Combined 3D view (pixels + ellipsoids) 
    ax6 = plt.subplot(2, 4, 6, projection='3d')
    ax6.scatter(B_cpu, G_cpu, R_cpu, c=nearest_cluster_cpu, 
               s=0.3, marker=',', cmap=cluster_cmap, norm=cluster_norm, alpha=0.4)
    for kk in ellipsoids:
        # Use colormap for ellipsoids to match scatter plot colors
        ellipsoid_color = cluster_cmap(cluster_norm(int(kk)))
        ax6.plot_surface(ellipsoids[kk][:,:,0],
                         ellipsoids[kk][:,:,1],
                         ellipsoids[kk][:,:,2],
                         color=ellipsoid_color,
                         alpha=0.2)
    ax6.set_xlabel('Blue', fontsize=12)
    ax6.set_ylabel('Green', fontsize=12)
    ax6.set_zlabel('Red', fontsize=12)
    ax6.set_title('Combined View', fontsize=14, weight='bold')
    
    # Cluster statistics bar plot 
    ax7 = plt.subplot(2, 4, 7)
    # Use colormap for the bar chart
    bar_colors = [cluster_cmap(cluster_norm(i)) for i in range(cluster_count)]
    bars = ax7.bar(range(cluster_count), weights, color=bar_colors, alpha=0.8, edgecolor='black')
    ax7.set_xlabel('Cluster ID', fontsize=12)
    ax7.set_ylabel('Weight', fontsize=12)
    ax7.set_title('Cluster Weights', fontsize=14, weight='bold')
    ax7.set_xticks(range(cluster_count))
    ax7.grid(True, alpha=0.3)
    
    # Rotate x-labels 
    if cluster_count > 10:
        ax7.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars 
    if cluster_count <= 15:
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Color key and cluster information 
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    ax8.set_title('Cluster Color Key', fontsize=14, weight='bold')
    
    # Calculate display parameters
    max_display_clusters = 15  # Maximum clusters to display 
    
    if cluster_count <= max_display_clusters:
        # Normal display 
        y_pos = 0.9
        spacing = 0.8 / cluster_count
        
        for i in range(cluster_count):
            # Color square using the colormap
            cluster_color = cluster_cmap(cluster_norm(i))
            square = plt.Rectangle((0.05, y_pos - 0.03), 0.08, 0.06, 
                                  facecolor=cluster_color, edgecolor='black', transform=ax8.transAxes)
            ax8.add_patch(square)
            
            # Text information
            pixel_count = np.sum(nearest_cluster_cpu == i)
            percentage = (pixel_count / len(nearest_cluster_cpu)) * 100
            
            text = f'Cluster {i}: {weights[i]:.4f} ({percentage:.1f}%)\n{pixel_count:,} pixels'
            ax8.text(0.2, y_pos, text, transform=ax8.transAxes, fontsize=10, 
                    verticalalignment='center')
            
            y_pos -= spacing
    else:
        # Compact display for many clusters
        ax8.text(0.5, 0.8, f'Total: {cluster_count} clusters', 
                transform=ax8.transAxes, ha='center', fontsize=12, weight='bold')
        ax8.text(0.5, 0.6, 'See detailed color key window\nfor all cluster information', 
                transform=ax8.transAxes, ha='center', fontsize=11)
    
    # Add summary statistics
    total_pixels = len(nearest_cluster_cpu)
    ax8.text(0.05, 0.05, f'Total pixels analyzed: {total_pixels:,}\nNumber of clusters: {cluster_count}', 
            transform=ax8.transAxes, fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle(f'Cluster Analysis Results: {flake_name}', fontsize=16, weight='bold', y=0.98)
    
    # Save the figure instead of showing during batch processing
    if show_plot:
        plt.show(block=False)
    else:
        # Save to outputs/cluster_visualization directory
        
        # Create directory if it doesn't exist
        output_dir = './outputs/training/training_cluster_visualization'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save with the new path
        filename = f'{output_dir}/{flake_name}_cluster_visualization.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
        plt.close()  # Close the figure to free memory
        return
    
    # Always create detailed color key for many clusters
    if cluster_count > 8:
        create_color_key(cluster_colors, weights, nearest_cluster_cpu, flake_name, cluster_cmap, cluster_norm)


def create_color_key(cluster_colors, weights, nearest_cluster_cpu, flake_name, cluster_cmap=None, cluster_norm=None):
    """.
    
    Parameters:
    -----------
    cluster_colors : list
        List of colors for each cluster (legacy parameter, now uses discrete colormap).
    weights : array
        Cluster weights.
    nearest_cluster_cpu : array
        Cluster assignments for each pixel.
    flake_name : str
        Name of the flake.
    cluster_cmap : matplotlib colormap, optional
        Discrete colormap for clusters.
    cluster_norm : matplotlib normalization, optional
        Normalization for discrete colors.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    cluster_count = len(weights)
    
    # Get colormap 
    if cluster_cmap is None or cluster_norm is None:
        cluster_cmap, cluster_norm = get_cluster_colormap(cluster_count)
    
    # Title
    ax.text(0.5, 0.95, f'Detailed Cluster Color Key - {flake_name}', 
            ha='center', va='top', transform=ax.transAxes, 
            fontsize=16, weight='bold')
    
    # Create table
    y_start = 0.85
    row_height = 0.08
    
    # Table headers
    headers = ['Color', 'Cluster ID', 'Weight', 'Pixel Count', 'Percentage']
    col_positions = [0.1, 0.25, 0.4, 0.6, 0.8]
    
    for i, header in enumerate(headers):
        ax.text(col_positions[i], y_start, header, transform=ax.transAxes, 
               fontsize=12, weight='bold', ha='center')
    
    # Draw header underline
    ax.plot([0.05, 0.95], [y_start - 0.02, y_start - 0.02], 'k-', 
            transform=ax.transAxes, linewidth=2)
    
    # Table data
    total_pixels = len(nearest_cluster_cpu)
    
    for i in range(cluster_count):
        y_pos = y_start - (i + 1) * row_height
        
        # Color square using discrete colormap
        cluster_color = cluster_cmap(cluster_norm(i))
        square = plt.Rectangle((col_positions[0] - 0.03, y_pos - 0.02), 0.06, 0.04, 
                              facecolor=cluster_color, edgecolor='black', 
                              transform=ax.transAxes)
        ax.add_patch(square)
        
        # Cluster data
        pixel_count = np.sum(nearest_cluster_cpu == i)
        percentage = (pixel_count / total_pixels) * 100
        
        data = [f'{i}', f'{weights[i]:.6f}', f'{pixel_count:,}', f'{percentage:.2f}%']
        
        for j, value in enumerate(data):
            ax.text(col_positions[j+1], y_pos, value, transform=ax.transAxes, 
                   fontsize=11, ha='center', va='center')
        
        # Row separator
        if i < cluster_count - 1:
            ax.plot([0.05, 0.95], [y_pos - row_height/2, y_pos - row_height/2], 
                   'k-', alpha=0.3, transform=ax.transAxes, linewidth=0.5)
    
    # Summary box
    summary_y = y_start - (cluster_count + 2) * row_height
    summary_text = f'Summary:\nTotal Pixels: {total_pixels:,}\nTotal Clusters: {cluster_count}\nImage: {flake_name}'
    
    ax.text(0.5, summary_y, summary_text, transform=ax.transAxes, 
           fontsize=12, ha='center', va='top',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)


def cluster_review_interface(results_dict, flake_name):
    """
    Interactive interface for reviewing clusters and assigning thickness values.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results.
    flake_name : str
        Name of the flake being processed.
        
    Returns:
    --------
    thickness_assignments : list
        List of thickness values assigned to each cluster.
    """
    
    # Extract data from results
    weights = results_dict[flake_name + ' weights']
    nearest_cluster_cpu = results_dict[flake_name + ' nearest_cluster']
    layer_image = results_dict[flake_name + ' layer_image']
    cluster_colors = results_dict[flake_name + ' cluster_colors']
    original_image = results_dict[flake_name + ' original_image']
    processed_image = results_dict[flake_name + ' processed_image']
    crop = results_dict[flake_name + ' crop']
    
    # Get initialization method 
    init_method = results_dict.get(flake_name + ' initialization_method', 'Unknown')
    density = results_dict.get(flake_name + ' density', 'N/A')
    
    cluster_count = len(weights)
    
    # Get colormap
    cluster_cmap, cluster_norm = get_cluster_colormap(cluster_count)
    
    # Create cropped image
    if crop is not None:
        cropped_original = original_image[crop[0]:crop[1], crop[2]:crop[3]]
    else:
        cropped_original = original_image
    
    # Dynamic layout calculation 
    clusters_per_row_group = 2  # 2 large clusters per row group
    cluster_row_groups = (cluster_count + clusters_per_row_group - 1) // clusters_per_row_group
    
    # Calculate figure size dynamically
    fig_height = 16 + (cluster_row_groups * 8)  #
    
    # Create the review figure
    fig = plt.figure(figsize=(24, fig_height))
    
    # Update Title
    title = f'Cluster Review and Thickness Assignment: {flake_name}'
    if init_method != 'Unknown':
        title += f'\n(Clustering Method: {init_method}'
        if init_method == 'meanshift' and density != 'N/A':
            title += f', Initial Density: {density}³ = {density**3} points'
        title += ')'
    
    fig.suptitle(title, fontsize=20, weight='bold', y=0.98)
    
    
    total_rows = 5 + (cluster_row_groups * 2)  
    gs = GridSpec(total_rows, 4, figure=fig, 
                  height_ratios=[1, 2, 2, 2, 2] + [2, 2] * cluster_row_groups)  
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original Full Image', fontsize=14, weight='bold')
    ax1.axis('off')
    
    # Add crop rectangle 
    if crop is not None:
        rect = Rectangle((crop[2], crop[0]), crop[3]-crop[2], crop[1]-crop[0], 
                        linewidth=3, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(crop[2], crop[0]-20, 'Cropped Region', color='red', fontsize=12, weight='bold')
    
    # Cluster statistics 
    ax_stats = fig.add_subplot(gs[0, 1:])  
    # Use colormap for the bar chart
    bar_colors = [cluster_cmap(cluster_norm(i)) for i in range(cluster_count)]
    bars = ax_stats.bar(range(cluster_count), weights, color=bar_colors, alpha=0.8, edgecolor='black')
    ax_stats.set_xlabel('Cluster ID', fontsize=14)
    ax_stats.set_ylabel('Weight', fontsize=14)
    ax_stats.set_title('Cluster Weights & Statistics', fontsize=16, weight='bold')
    ax_stats.set_xticks(range(cluster_count))
    ax_stats.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-labels 
    if cluster_count > 10:
        ax_stats.tick_params(axis='x', rotation=45, labelsize=12)
    else:
        ax_stats.tick_params(axis='x', labelsize=12)
    ax_stats.tick_params(axis='y', labelsize=12)
    
    # Add value labels on bars
    if cluster_count <= 25:
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            pixel_count = np.sum(nearest_cluster_cpu == i)
            percentage = (pixel_count / len(nearest_cluster_cpu)) * 100
            ax_stats.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{weight:.3f}\n{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Cropped Original 
    ax2 = fig.add_subplot(gs[1:3, 0:2])  
    ax2.imshow(cropped_original)
    ax2.set_title('Cropped Original (Unprocessed)', fontsize=18, weight='bold')
    ax2.axis('off')
    
    # Processed image 
    ax3 = fig.add_subplot(gs[1:3, 2:4])  
    ax3.imshow(processed_image)
    ax3.set_title('Processed Image\n(After Filtering & Background Correction)', fontsize=18, weight='bold')
    ax3.axis('off')
    
    # Cluster assignments 
    ax4 = fig.add_subplot(gs[3:5, 0:2])  
    cluster_display = ax4.imshow(layer_image, cmap=cluster_cmap, norm=cluster_norm)
    ax4.set_title('Cluster Assignments\n(Compare with individual masks below)', fontsize=18, weight='bold')
    ax4.axis('off')
    
    # Add colorbar to cluster assignments
    cbar = plt.colorbar(cluster_display, ax=ax4, shrink=0.8, aspect=15, ticks=np.arange(cluster_count))
    cbar.set_label('Cluster ID', fontsize=14)
    cbar.ax.set_yticklabels([str(i) for i in range(cluster_count)])
    cbar.ax.tick_params(labelsize=11)
    
    # Summary/Color key 
    ax_summary = fig.add_subplot(gs[3:5, 2:4])  
    ax_summary.axis('off')
    ax_summary.set_title('Analysis Summary', fontsize=18, weight='bold')
    
    # Create summary text
    summary_text = f"""CLUSTERING SUMMARY:
Method: {init_method.upper()}"""
    
    if init_method == 'meanshift' and density != 'N/A':
        summary_text += f"""
Grid Density: {density}³ = {density**3} points"""
    
    total_pixels = len(nearest_cluster_cpu)
    summary_text += f"""
Total Clusters: {cluster_count}
Total Pixels: {total_pixels:,}

CLUSTER BREAKDOWN:"""
    
    # Add cluster breakdown
    for i in range(min(cluster_count, 8)):  
        pixel_count = np.sum(nearest_cluster_cpu == i)
        percentage = (pixel_count / total_pixels) * 100
        summary_text += f"""
Cluster {i}: {pixel_count:,} pixels ({percentage:.1f}%)"""
    
    if cluster_count > 8:
        summary_text += f"""
... and {cluster_count - 8} more clusters
(See individual masks below)"""
    
    summary_text += f"""

INSTRUCTIONS:
• Compare cluster assignments (left) with individual masks below
• Each mask shows same discrete color as in main assignment
• Assign thickness: 0=substrate, 1=monolayer, 2=bilayer, etc.
• Use -1 to exclude clusters from analysis"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Display ALL cluster masks 
    for i in range(cluster_count):
        row_group = i // clusters_per_row_group  
        position_in_group = i % clusters_per_row_group  
        
        # Calculate the starting row for this cluster 
        start_row = 5 + (row_group * 2)  
        
        # Position within the row group 
        if position_in_group == 0:
            # Left cluster 
            ax = fig.add_subplot(gs[start_row:start_row+2, 0:2])
        else:
            # Right cluster 
            ax = fig.add_subplot(gs[start_row:start_row+2, 2:4])
        
        cluster_mask = (layer_image == i).astype(float)
        
        # Create a custom colormap for this cluster 
        cluster_color = cluster_cmap(cluster_norm(i))
        # Map 0 (background) to white, 1 (cluster pixels) to cluster color
        custom_colors = ['white', cluster_color]
        custom_cmap = ListedColormap(custom_colors)
        
        ax.imshow(cluster_mask, cmap=custom_cmap, vmin=0, vmax=1)
        
        pixel_count = np.sum(nearest_cluster_cpu == i)
        percentage = (pixel_count / len(nearest_cluster_cpu)) * 100
        
        # Add cluster info to title with larger font and colored background
        title_text = f'Cluster {i}\n{pixel_count:,} pixels ({percentage:.1f}%)'
        ax.set_title(title_text, fontsize=18, weight='bold', 
                    bbox=dict(boxstyle="round,pad=0.8", facecolor=cluster_color, alpha=0.3, edgecolor=cluster_color, linewidth=3))
        ax.axis('off')
        
        # Add inner border 
        for spine in ax.spines.values():
            spine.set_edgecolor(cluster_color)
            spine.set_linewidth(4)
            spine.set_visible(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, bottom=0.05)  
    plt.show(block=False)
    
    # Interactive thickness assignment
    print("\n" + "="*60)
    print("THICKNESS ASSIGNMENT")
    print("="*60)
    print(f"Total clusters to assign: {cluster_count}")
    print("Examine the visualization above:")
    print("• Compare large cluster assignments with individual masks below")
    print("Enter thickness for each cluster (0=substrate, 1=monolayer, 2=bilayer, etc., -1=exclude)")
    print("-" * 60, flush=True)
    
    thickness_assignments = []
    
    for i in range(cluster_count):
        pixel_count = np.sum(nearest_cluster_cpu == i)
        percentage = (pixel_count / len(nearest_cluster_cpu)) * 100
        
        while True:
            try:
                prompt = f"Cluster {i} ({pixel_count:,} pixels, {percentage:.1f}%): "
                thickness_input = input(prompt).strip()
                
                if thickness_input == '':
                    # Default to 0 (substrate) for empty input
                    thickness = 0
                    print(f"  └─ Using default: substrate (0)")
                else:
                    thickness = int(thickness_input)
                
                if thickness >= -1:
                    thickness_assignments.append(thickness)
                    
                    # Provide user feedback for assignments
                    if thickness == 0:
                        print(f"  └─ Assigned: Background/Substrate")
                    elif thickness == -1:
                        print(f"  └─ Assigned: Exclude from analysis")
                    else:
                        layer_text = "layer" if thickness == 1 else "layers"
                        print(f"  └─ Assigned: {thickness} {layer_text}")
                    break
                else:
                    print("  └─ Please enter a value ≥ -1")
                    
            except ValueError:
                print("  └─ Please enter a valid integer")
    
    # Show summary and confirmation
    print("\n" + "="*60)
    print("ASSIGNMENT SUMMARY")
    print("="*60, flush=True)
    
    for i, thickness in enumerate(thickness_assignments):
        pixel_count = np.sum(nearest_cluster_cpu == i)
        percentage = (pixel_count / len(nearest_cluster_cpu)) * 100
        
        if thickness == 0:
            thickness_label = "Background/Substrate"
        elif thickness == -1:
            thickness_label = "Exclude from analysis"
        else:
            layer_text = "layer" if thickness == 1 else "layers"
            thickness_label = f"{thickness} {layer_text}"
        
        print(f"Cluster {i}: {thickness_label} ({pixel_count:,} pixels, {percentage:.1f}%)")
    
    
    plt.close(fig)  
    return thickness_assignments


def thickness_visualization(results_dict, flake_name):
    """
    Create a visualization showing the final thickness assignments.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing clustering results with thickness assignments.
    flake_name : str
        Name of the flake being processed.
    """
    if flake_name + ' cluster_thicknesses' not in results_dict:
        print("No thickness assignments found. Run review_and_edit_results first.")
        return
    
    # Extract data
    layer_image = results_dict[flake_name + ' layer_image']
    cluster_thicknesses = results_dict[flake_name + ' cluster_thicknesses']
    original_image = results_dict[flake_name + ' original_image']
    nearest_cluster = results_dict[flake_name + ' nearest_cluster']
    weights = results_dict[flake_name + ' weights']
    
    # Get initialization method if available
    init_method = results_dict.get(flake_name + ' initialization_method', 'Unknown')
    density = results_dict.get(flake_name + ' density', 'N/A')

    cluster_count = len(weights)
    
    # Get colormap for clusters
    cluster_cmap, cluster_norm = get_cluster_colormap(cluster_count)

    # Create thickness map 
    thickness_map = np.zeros_like(layer_image, dtype=float)
    for cluster_id, thickness in enumerate(cluster_thicknesses):
        if thickness >= 0:  # Don't map excluded clusters (-1)
            thickness_map[layer_image == cluster_id] = thickness
    
    # Create colormap for thickness values
    unique_thicknesses = sorted(set([t for t in cluster_thicknesses if t >= 0]))
    max_thickness = max(unique_thicknesses) if unique_thicknesses else 0
    
    
    if max_thickness <= 1:
        thickness_colors = ['#2E8B57', '#FFD700'][:len(unique_thicknesses)]  # Green to Gold
    elif max_thickness <= 5:
        thickness_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_thicknesses)))
    else:
        thickness_colors = plt.cm.plasma(np.linspace(0, 1, len(unique_thicknesses)))
    
    thickness_cmap = ListedColormap(thickness_colors)
    thickness_bounds = np.array(unique_thicknesses + [max(unique_thicknesses) + 1]) - 0.5
    thickness_norm = BoundaryNorm(thickness_bounds, len(unique_thicknesses))
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Update title with method info
    title = f'Thickness Analysis Results: {flake_name}'
    if init_method != 'Unknown':
        title += f' ({init_method} clustering'
        if init_method == 'meanshift' and density != 'N/A':
            title += f', density={density}'
        title += ')'
    
    fig.suptitle(title, fontsize=16, weight='bold')
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    
    # Cluster assignments
    im1 = axes[0, 1].imshow(layer_image, cmap=cluster_cmap, norm=cluster_norm)
    axes[0, 1].set_title('Cluster Assignments', fontsize=14, weight='bold')
    axes[0, 1].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], shrink=0.8, ticks=np.arange(cluster_count))
    cbar1.set_label('Cluster ID')
    cbar1.ax.set_yticklabels([str(i) for i in range(cluster_count)])
    
    # Thickness map
    im2 = axes[0, 2].imshow(thickness_map, cmap=thickness_cmap, norm=thickness_norm)
    axes[0, 2].set_title('Thickness Map', fontsize=14, weight='bold')
    axes[0, 2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[0, 2], shrink=0.8, ticks=unique_thicknesses)
    cbar2.set_label('Number of Layers', fontsize=12)
    
    # Add layer labels to thickness colorbar
    thickness_labels = []
    for t in unique_thicknesses:
        if t == 0:
            thickness_labels.append('Substrate')
        elif t == 1:
            thickness_labels.append('1L')
        else:
            thickness_labels.append(f'{t}L')
    cbar2.ax.set_yticklabels(thickness_labels)
    
    # Thickness histogram
    valid_thicknesses = [t for t in cluster_thicknesses if t >= 0]
    thickness_counts = {}
    for cluster_id, thickness in enumerate(cluster_thicknesses):
        if thickness >= 0:  # Only count non-excluded clusters
            pixel_count = np.sum(layer_image == cluster_id)
            if thickness in thickness_counts:
                thickness_counts[thickness] += pixel_count
            else:
                thickness_counts[thickness] = pixel_count
    
    if thickness_counts:
        thicknesses = list(thickness_counts.keys())
        counts = list(thickness_counts.values())
        
        # Use the same colormap as the thickness map
        bar_colors = [thickness_cmap(thickness_norm(t)) for t in thicknesses]
        bars = axes[1, 0].bar(thicknesses, counts, color=bar_colors, alpha=0.8, edgecolor='black')
        axes[1, 0].set_xlabel('Number of Layers', fontsize=12)
        axes[1, 0].set_ylabel('Pixel Count', fontsize=12)
        axes[1, 0].set_title('Thickness Distribution', fontsize=14, weight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Customize x-axis labels
        axes[1, 0].set_xticks(thicknesses)
        thickness_xlabels = []
        for t in thicknesses:
            if t == 0:
                thickness_xlabels.append('Substrate')
            elif t == 1:
                thickness_xlabels.append('1L')
            else:
                thickness_xlabels.append(f'{t}L')
        axes[1, 0].set_xticklabels(thickness_xlabels)
        
        # Add percentage labels
        total_pixels = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = (count / total_pixels) * 100
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Cluster-thickness mapping table
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Cluster-Thickness Mapping', fontsize=14, weight='bold')
    
    table_data = []
    for cluster_id, thickness in enumerate(cluster_thicknesses):
        pixel_count = np.sum(nearest_cluster == cluster_id)
        percentage = (pixel_count / len(nearest_cluster)) * 100
        
        if thickness == 0:
            thickness_label = "Substrate"
        elif thickness == -1:
            thickness_label = "Excluded"
        else:
            thickness_label = f"{thickness}L"
        
        table_data.append([f"{cluster_id}", thickness_label, f"{pixel_count:,}", f"{percentage:.1f}%"])
    
    # Create table
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Cluster', 'Thickness', 'Pixels', '%'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color table rows by cluster 
    for i, thickness in enumerate(cluster_thicknesses):
        cluster_color = cluster_cmap(cluster_norm(i))
        
        for j in range(4):
            table[(i+1, j)].set_facecolor(cluster_color)
            table[(i+1, j)].set_alpha(0.3)
    
    # Area analysis
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Area Analysis', fontsize=14, weight='bold')
    
    # Calculate areas 
    total_analysis_pixels = np.sum([np.sum(nearest_cluster == i) 
                                   for i, t in enumerate(cluster_thicknesses) if t >= 0])
    
    analysis_text = f"Area Analysis Summary:\n\n"
    analysis_text += f"Total analyzed area: {total_analysis_pixels:,} pixels\n\n"
    
    for thickness in sorted(set(valid_thicknesses)):
        thickness_pixels = sum(np.sum(layer_image == cluster_id) 
                              for cluster_id, t in enumerate(cluster_thicknesses) 
                              if t == thickness)
        percentage = (thickness_pixels / total_analysis_pixels) * 100
        
        if thickness == 0:
            analysis_text += f"Substrate: {thickness_pixels:,} pixels ({percentage:.1f}%)\n"
        else:
            layer_text = "layer" if thickness == 1 else "layers"
            analysis_text += f"{thickness} {layer_text}: {thickness_pixels:,} pixels ({percentage:.1f}%)\n"
    
    excluded_pixels = sum(np.sum(layer_image == cluster_id) 
                         for cluster_id, t in enumerate(cluster_thicknesses) 
                         if t == -1)
    if excluded_pixels > 0:
        analysis_text += f"\nExcluded: {excluded_pixels:,} pixels"
    
    axes[1, 2].text(0.05, 0.95, analysis_text, transform=axes[1, 2].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show(block=False)
    return fig

def inference_visualization(layer_image, original_image, processed_image, img_proc_display,
                                   clusters, pixel_counts, total_pixels, valid_pixel_count,
                                   invalid_pixel_count, background_pixel_count, background_fraction,
                                   mask, flake_name, show_plot):
    """
    Creates a visualization for classification results.
    """
    # Create output directory
    output_dir = './outputs/classification_results/images'
    os.makedirs(output_dir, exist_ok=True)

    max_cols = 3
    base_rows = 2  
    num_clusters = len(clusters)
    cluster_rows = math.ceil(num_clusters / max_cols)
    total_rows = base_rows + cluster_rows

    fig = plt.figure(figsize=(6 * max_cols, 4 * total_rows))

    # Create colormap for clusters 
    if num_clusters <= 10:
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif num_clusters <= 20:
        base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    else:
        tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        base_colors = np.vstack([tab10_colors, tab20_colors])
    
    # Select colors for clusters
    cluster_colors = base_colors[:num_clusters] if num_clusters > 0 else [base_colors[0]]

    # Row 1
    ax1 = plt.subplot(total_rows, max_cols, 1)
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=14, weight='bold')
    ax1.axis('off')

    ax2 = plt.subplot(total_rows, max_cols, 2)
    ax2.imshow(original_image)
    mask_overlay = np.zeros((*mask.shape, 4))  # RGBA
    mask_overlay[mask == 0] = [0, 0, 0, 0.3]
    ax2.imshow(mask_overlay)
    ax2.set_title(f'Original + Mask Overlay ({background_fraction:.1%})', fontsize=14, weight='bold')
    ax2.axis('off')

    ax3 = plt.subplot(total_rows, max_cols, 3)
    ax3.imshow(processed_image)
    ax3.set_title('Background-Corrected Image', fontsize=14, weight='bold')
    ax3.axis('off')

    # Row 2
    ax4 = plt.subplot(total_rows, max_cols, max_cols + 1)
    layer_display = layer_image.copy().astype(float)
    layer_display[layer_image == -1] = np.nan
    
    # Create a custom colormap for layer classification
    if num_clusters > 0:
        layer_cmap = ListedColormap(cluster_colors)
        im = ax4.imshow(layer_display, cmap=layer_cmap, vmin=0, vmax=num_clusters-1, interpolation='nearest')
    else:
        im = ax4.imshow(layer_display, cmap='viridis', interpolation='nearest')
    
    ax4.set_title('Layer Classification', fontsize=14, weight='bold')
    ax4.axis('off')

    # Set up colorbar
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8, pad=0.02)
    cbar.set_label('Layer Number', fontsize=12)
    if len(clusters) > 0:
        cbar.set_ticks(clusters)
        cbar.set_ticklabels([f'Layer {i}' for i in clusters])

    ax5 = plt.subplot(total_rows, max_cols, max_cols + 2)
    ax5.axis('off')
    ax5.set_title('Layer Distribution', fontsize=14, weight='bold')

    y_pos = 0.9
    spacing = 0.8 / max(len(clusters), 1)

    for idx, i in enumerate(clusters):
        # Use colormap for colors
        color = cluster_colors[idx] if idx < len(cluster_colors) else cluster_colors[0]
        
        square = plt.Rectangle((0.1, y_pos - 0.04), 0.08, 0.06,
                               facecolor=color, edgecolor='black',
                               transform=ax5.transAxes, linewidth=2)
        ax5.add_patch(square)

        count = pixel_counts[i]
        percentage = (count / total_pixels) * 100
        text = f'Layer {i}: {count:,} pixels ({percentage:.2f}%)'
        ax5.text(0.25, y_pos - 0.01, text, transform=ax5.transAxes,
                 fontsize=12, verticalalignment='center')
        y_pos -= spacing

    ax6 = plt.subplot(total_rows, max_cols, max_cols + 3)
    ax6.axis('off')
    ax6.set_title('Analysis Summary', fontsize=14, weight='bold')
    valid_pct = (valid_pixel_count / total_pixels) * 100
    invalid_pct = (invalid_pixel_count / total_pixels) * 100
    background_pct = (background_pixel_count / total_pixels) * 100

    summary_text = f"Image: {flake_name}\n"
    summary_text += f"{'='*40}\n\n"
    summary_text += f"Total Pixels: {total_pixels:,}\n\n"
    summary_text += f"Pixel Categories:\n"
    summary_text += f"  Valid pixels:      {valid_pixel_count:>8,} ({valid_pct:>6.2f}%)\n"
    summary_text += f"  Invalid pixels:    {invalid_pixel_count:>8,} ({invalid_pct:>6.2f}%)\n"
    summary_text += f"  Background pixels: {background_pixel_count:>8,} ({background_pct:>6.2f}%)\n\n"
    summary_text += f"Layer Breakdown:\n"
    for i in clusters:
        count = pixel_counts[i]
        pct = (count / total_pixels) * 100
        summary_text += f"  Layer {i}: {count:>8,} ({pct:>6.2f}%)\n"

    ax6.text(0.1, 0.85, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    # Cluster masks
    for idx, i in enumerate(clusters):
        row = base_rows + (idx // max_cols)
        col = idx % max_cols
        ax_idx = row * max_cols + col + 1
        ax = plt.subplot(total_rows, max_cols, ax_idx)
        binary_mask = (layer_image == i).astype(float)
        
        # Create a custom colormap for this layer 
        layer_color = cluster_colors[idx] if idx < len(cluster_colors) else cluster_colors[0]
        
        # Show mask in the layer's color
        colored_mask = np.zeros((*binary_mask.shape, 4))  # RGBA
        colored_mask[binary_mask == 1] = [*layer_color[:3], 1.0]  # Layer color with full alpha
        colored_mask[binary_mask == 0] = [1, 1, 1, 1]  # White background
        
        ax.imshow(colored_mask)
        ax.set_title(f'Layer {i} Mask', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=layer_color, alpha=0.3))
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle(f'Layer Analysis: {flake_name}', fontsize=18, weight='bold', y=1.02)

    filename = f'{output_dir}/{flake_name}_analysis.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {filename}")

    if show_plot:
        plt.show()
    else:
        plt.close()

