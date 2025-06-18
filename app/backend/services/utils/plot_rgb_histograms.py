import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy import stats

def plot_R_G_B_histograms(dataset_map, base_path, subdir_name, index_img):
    """
    Plot density curves of R, G, and B channels for a specific image using cv2.
    
    Parameters:
    -----------
    dataset_map : dict
        Dictionary mapping subdirectories to lists of image files.
    base_path : str
        Base path to the dataset.
    subdir_name : str
        Name of the subdirectory containing the target image.
    index_img : int
        Index of the image in the dataset_map[subdir_name] list.
    
    Returns:
    --------
    None
        Displays the density curves of R, G, and B channels.
    """
    # Get the image path using the dataset map
    list_img = dataset_map[subdir_name]
    img_path = os.path.join(base_path, f"{subdir_name}/{list_img[index_img]}")
    print(f"Image path: {img_path}")
    
    # Load the image using cv2
    img = cv2.imread(img_path)
    
    # Check if the image is loaded correctly
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return
    
    # Convert BGR (cv2 default) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Extract R, G, B channels
    r_channel = img_rgb[:, :, 0]
    g_channel = img_rgb[:, :, 1]
    b_channel = img_rgb[:, :, 2]
    
    # Flatten the channels
    r_flat = r_channel.flatten()
    g_flat = g_channel.flatten()
    b_flat = b_channel.flatten()
    
    # Create figure with 3 subplots (one for each channel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define x range for density estimation (0-255 for 8-bit images)
    x = np.linspace(0, 255, 256)
    
    # Plot density curves for each channel
    # Red channel
    kde_r = stats.gaussian_kde(r_flat)
    axes[0].plot(x, kde_r(x), color='red', linewidth=2)
    axes[0].fill_between(x, kde_r(x), alpha=0.3, color='red')
    axes[0].set_title('Red Channel Density')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Density')
    
    # Green channel
    kde_g = stats.gaussian_kde(g_flat)
    axes[1].plot(x, kde_g(x), color='green', linewidth=2)
    axes[1].fill_between(x, kde_g(x), alpha=0.3, color='green')
    axes[1].set_title('Green Channel Density')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Density')
    
    # Blue channel
    kde_b = stats.gaussian_kde(b_flat)
    axes[2].plot(x, kde_b(x), color='blue', linewidth=2)
    axes[2].fill_between(x, kde_b(x), alpha=0.3, color='blue')
    axes[2].set_title('Blue Channel Density')
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Density')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Print some statistics about each channel
    print(f"Red channel - Min: {r_flat.min()}, Max: {r_flat.max()}, Mean: {r_flat.mean():.2f}")
    print(f"Green channel - Min: {g_flat.min()}, Max: {g_flat.max()}, Mean: {g_flat.mean():.2f}")
    print(f"Blue channel - Min: {b_flat.min()}, Max: {b_flat.max()}, Mean: {b_flat.mean():.2f}")

def plot_average_RGB_distribution(dataset_map, base_path, subdir_name, batch_size=20, sample_size=None, bins=50):
    """
    Plot average density curves of R, G, and B channels for all images in a subdirectory.
    Uses batch processing and histogram-based approach for better performance.
    
    Parameters:
    -----------
    dataset_map : dict
        Dictionary mapping subdirectories to lists of image files.
    base_path : str
        Base path to the dataset.
    subdir_name : str
        Name of the subdirectory containing the images.
    batch_size : int, optional (default=20)
        Number of images to process in each batch.
    sample_size : int, optional (default=None)
        Number of pixels to sample from each image for KDE estimation.
        If None, uses all pixels (slower but more accurate).
    bins : int, optional (default=50)
        Number of bins for the histogram approach.
    
    Returns:
    --------
    None
        Displays the average density curves of R, G, and B channels.
    """
    # Check if the subdirectory exists in the dataset map
    if subdir_name not in dataset_map:
        print(f"Error: Subdirectory '{subdir_name}' not found in dataset map")
        return
    
    # Get the list of images in the subdirectory
    list_img = dataset_map[subdir_name]
    num_images = len(list_img)
    
    if num_images == 0:
        print(f"Error: No images found in subdirectory '{subdir_name}'")
        return
    
    print(f"Processing {num_images} images from subdirectory '{subdir_name}'...")
    
    # Initialize arrays for histograms
    r_hist_total = np.zeros(bins)
    g_hist_total = np.zeros(bins)
    b_hist_total = np.zeros(bins)
    
    # Statistics accumulators
    r_mins, r_maxs, r_means = [], [], []
    g_mins, g_maxs, g_means = [], [], []
    b_mins, b_maxs, b_means = [], [], []
    
    # Process images in batches
    for batch_start in range(0, num_images, batch_size):
        batch_end = min(batch_start + batch_size, num_images)
        batch_images = list_img[batch_start:batch_end]
        batch_size_actual = len(batch_images)
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(num_images-1)//batch_size + 1} (images {batch_start+1}-{batch_end})")
        
        # Process each image in the batch
        for img_name in batch_images:
            # Construct the full image path
            img_path = os.path.join(base_path, f"{subdir_name}/{img_name}")
            
            # Load the image using cv2
            img = cv2.imread(img_path)
            
            # Skip if image couldn't be loaded
            if img is None:
                print(f"Warning: Could not load image at {img_path}. Skipping...")
                continue
            
            # Convert BGR (cv2 default) to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Extract R, G, B channels
            r_channel = img_rgb[:, :, 0]
            g_channel = img_rgb[:, :, 1]
            b_channel = img_rgb[:, :, 2]
            
            # Sample pixels if sample_size is specified
            if sample_size is not None:
                h, w = r_channel.shape
                total_pixels = h * w
                if total_pixels > sample_size:
                    # Create a random sample of pixel indices
                    indices = np.random.choice(total_pixels, sample_size, replace=False)
                    r_flat = r_channel.flatten()[indices]
                    g_flat = g_channel.flatten()[indices]
                    b_flat = b_channel.flatten()[indices]
                else:
                    r_flat = r_channel.flatten()
                    g_flat = g_channel.flatten()
                    b_flat = b_channel.flatten()
            else:
                r_flat = r_channel.flatten()
                g_flat = g_channel.flatten()
                b_flat = b_channel.flatten()
            
            # Collect statistics
            r_mins.append(r_flat.min())
            r_maxs.append(r_flat.max())
            r_means.append(r_flat.mean())
            
            g_mins.append(g_flat.min())
            g_maxs.append(g_flat.max())
            g_means.append(g_flat.mean())
            
            b_mins.append(b_flat.min())
            b_maxs.append(b_flat.max())
            b_means.append(b_flat.mean())
            
            # Calculate histograms for each channel
            r_hist, _ = np.histogram(r_flat, bins=bins, range=(0, 255), density=True)
            g_hist, _ = np.histogram(g_flat, bins=bins, range=(0, 255), density=True)
            b_hist, _ = np.histogram(b_flat, bins=bins, range=(0, 255), density=True)
            
            # Accumulate histograms
            r_hist_total += r_hist
            g_hist_total += g_hist
            b_hist_total += b_hist
    
    # Calculate average histograms
    r_hist_avg = r_hist_total / num_images
    g_hist_avg = g_hist_total / num_images
    b_hist_avg = b_hist_total / num_images
    
    # Create x-axis for plotting (bin centers)
    bin_edges = np.linspace(0, 255, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Create figure with 3 subplots (one for each channel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot average density curves for each channel
    # Red channel
    axes[0].plot(bin_centers, r_hist_avg, color='red', linewidth=2)
    axes[0].fill_between(bin_centers, r_hist_avg, alpha=0.3, color='red')
    axes[0].set_title(f'Average Red Channel Density\n({subdir_name})')
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Density')
    
    # Green channel
    axes[1].plot(bin_centers, g_hist_avg, color='green', linewidth=2)
    axes[1].fill_between(bin_centers, g_hist_avg, alpha=0.3, color='green')
    axes[1].set_title(f'Average Green Channel Density\n({subdir_name})')
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Density')
    
    # Blue channel
    axes[2].plot(bin_centers, b_hist_avg, color='blue', linewidth=2)
    axes[2].fill_between(bin_centers, b_hist_avg, alpha=0.3, color='blue')
    axes[2].set_title(f'Average Blue Channel Density\n({subdir_name})')
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Density')
    
    # Adjust layout and display
    plt.suptitle(f'Average RGB Distribution for {num_images} Images in {subdir_name}')
    plt.tight_layout()
    plt.show()
    
    # Print average statistics for each channel
    print(f"\nAverage Statistics for {num_images} images in '{subdir_name}':")
    print(f"Red channel - Min: {np.mean(r_mins):.2f}, Max: {np.mean(r_maxs):.2f}, Mean: {np.mean(r_means):.2f}")
    print(f"Green channel - Min: {np.mean(g_mins):.2f}, Max: {np.mean(g_maxs):.2f}, Mean: {np.mean(g_means):.2f}")
    print(f"Blue channel - Min: {np.mean(b_mins):.2f}, Max: {np.mean(b_maxs):.2f}, Mean: {np.mean(b_means):.2f}")

    axes[2].set_ylabel('Density')
    
    # Adjust layout and display
    plt.suptitle(f'Average RGB Distribution for {num_images} Images in {subdir_name}')
    plt.tight_layout()
    plt.show()
    
    # Print average statistics for each channel
    print(f"\nAverage Statistics for {num_images} images in '{subdir_name}':")
    print(f"Red channel - Min: {np.mean(r_mins):.2f}, Max: {np.mean(r_maxs):.2f}, Mean: {np.mean(r_means):.2f}")
    print(f"Green channel - Min: {np.mean(g_mins):.2f}, Max: {np.mean(g_maxs):.2f}, Mean: {np.mean(g_means):.2f}")
    print(f"Blue channel - Min: {np.mean(b_mins):.2f}, Max: {np.mean(b_maxs):.2f}, Mean: {np.mean(b_means):.2f}")

# Example usage:
# Assuming you have already defined:
# - dataset_map = map_subdirs_to_files(data_path)
# - data_path = "/path/to/your/dataset"
# 
# Then you can call:
# plot_R_G_B_histograms(dataset_map, data_path, "ig", 0)
# plot_average_RGB_distribution(dataset_map, data_path, "ig")
# Example usage:
# Assuming you have already defined:
# - dataset_map = map_subdirs_to_files(data_path)
# - data_path = "/path/to/your/dataset"