import os
from os import listdir
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from tqdm import tqdm
from utils.dataloader1 import dataloader
from EPcode.SARNNSim2 import SARNNM 
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    # ==============================================
    ## Initialize random seed
    set_seed()

    # Setup GPU/CPU device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA device index: {torch.cuda.current_device()}")
        gpu_properties = torch.cuda.get_device_properties(device)
        print(f"GPU Name: {gpu_properties.name}")
        print(f"GPU Memory: {gpu_properties.total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    # Set paths
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "space_station.obj")
    if os.path.exists(obj_filename):
        print(f"3D model found: {obj_filename}")
    else:
        print(f"3D model not found: {obj_filename}")
        print("Please place your space station 3D model at this location")
        exit(1)

    # Output directory for SAR images
    output_dir = './gaofen3_sar_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # ==============================================
    # Gaofen-3 SAR parameters (Complete specifications)
    
    # Basic parameters
    wavelength = 0.055517  # C-band wavelength in meters
    frequency = 5.4e9      # C-band frequency in Hz
    
    # Resolution and pixel spacing
    slant_range_resolution = 0.63  # meters (range direction)
    slant_range_pixel_spacing = 0.56  # meters (range direction)
    azimuth_resolution = 2.5  # meters (NEW INFO: azimuth resolution is 2-3 meters)
    azimuth_pixel_spacing = 2.0  # meters (estimated based on typical SAR systems)
    
    # Signal parameters
    bandwidth = 240e6     # 240 MHz
    sampling_rate = 267e6 # 267 MHz
    prf = 1745.000488     # Pulse Repetition Frequency in Hz
    
    # Imaging geometry
    distance = 10000     # 755 km typical orbit height
    
    # Image size calculation
    # For a roughly square target area with different resolutions in range/azimuth
    # we need different numbers of pixels in each dimension
    coverage_area = 200   # meters (area to cover around target)
    range_pixels = int(coverage_area / slant_range_pixel_spacing)
    azimuth_pixels = int(coverage_area / azimuth_pixel_spacing)
    
    # Make sure dimensions are even numbers
    range_pixels = range_pixels + (range_pixels % 2)
    azimuth_pixels = azimuth_pixels + (azimuth_pixels % 2)
    
    print(f"Image dimensions: {range_pixels}×{azimuth_pixels} pixels")
    print(f"Physical coverage: {range_pixels*slant_range_pixel_spacing:.1f}m × {azimuth_pixels*azimuth_pixel_spacing:.1f}m")
    
    # Use the larger dimension for square output (SARNNM may require square images)
    imagesize = max(range_pixels, azimuth_pixels)
    scalefactor = 0.1 # Scale factor for the 3D model
    
    # Create complete simulation settings dictionary
    simsetting = {
        'distance': distance,
        'SlantPixSpacing': slant_range_pixel_spacing,
        'AzimuthPixSpacing': azimuth_pixel_spacing,
        'SlantResolution': slant_range_resolution,
        'AzimuthResolution': azimuth_resolution,
        'imagesize': imagesize,
        'scalefactor': scalefactor,
        'wavelength': wavelength,
        'frequency': frequency,
        'bandwidth': bandwidth,
        'sampling_rate': sampling_rate,
        'prf': prf
    }

    # ==============================================
    # Create model for rendering
    EPNum = 0  # No training - just for initialization
    print(f"Creating SARNNM model for Gaofen-3 SAR simulation...")
    model = SARNNM(
        obj_filename=obj_filename, 
        simsetting=simsetting, 
        EPNum=EPNum, 
        savepath=output_dir, 
        device=device
    )
    
    # ==============================================
    # Define viewing angles for space station imaging
    # For a space station in orbit, we want to simulate viewing from different angles
    
    # Define elevation angles (angle from horizontal plane)
    # For satellite SAR, typical elevation angles are between 20-60 degrees
    elevations = list(range(0, 90, 15))  # From directly overhead to horizontal
    
    # Define azimuth angles (angle around vertical axis)
    # Full 360° coverage to view the space station from all sides
    azimuths = list(range(0, 360, 15 ))  # Every 15 degrees
    
    # Generate images for each angle combination
    print("Generating Gaofen-3 SAR images of space station...")
    progress_bar = tqdm(total=len(elevations)*len(azimuths))
    
    # Create a summary figure to show sample results
    fig_rows = len(elevations)
    fig_cols = 4  # Show 4 sample azimuths
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(16, 4*len(elevations)))
    sample_azimuths = [0, 90, 180, 270]  # Sample angles for visualization
    
    for elev_idx, elev in enumerate(elevations):
        for azim_idx, azim in enumerate(azimuths):
            # Create input sample with angle information
            sample = {
                'elev': torch.tensor([elev], device=device, dtype=torch.float32),
                'azim': torch.tensor([azim], device=device, dtype=torch.float32),
                'samplename': [f'gaofen3_elev{elev}_azim{azim}.png'],
                # Dummy tensor for real image (not used for rendering)
                'realimg': torch.zeros(1, 1, imagesize, imagesize, device=device)
            }
            
            # Generate the SAR image (no training/optimization)
            with torch.no_grad():
               
   # Get vertices
                verts = model.src_mesh.verts_padded()
   
   # Calculate center of model
                center = verts.mean(dim=1, keepdim=True)
   
   # Center the model by subtracting center
                centered_verts = verts - center
   
   # Update the mesh vertices
                model.src_mesh = model.src_mesh.update_padded(centered_verts)
                sarimgs, silimgs = model.sarimgnnsim(sample, model.src_mesh)
                
                # Apply netPix for realistic SAR simulation if available
                if hasattr(model, 'netPix'):
                    sarimgs = model.netPix(sarimgs)
                
                # Save the SAR image
                sar_filename = f'gaofen3_elev{elev}_azim{azim}_sar.png'
                sar_path = os.path.join(output_dir, sar_filename)
                vutils.save_image(sarimgs[0], fp=sar_path, normalize=True)
                
                # Save the silhouette image
                sil_filename = f'gaofen3_elev{elev}_azim{azim}_silhouette.png'
                sil_path = os.path.join(output_dir, sil_filename)
                vutils.save_image(silimgs[0], fp=sil_path, normalize=True)
                
                # Add to summary figure if this is one of our sampled azimuths
                if azim in sample_azimuths:
                    ax_idx = sample_azimuths.index(azim)
                    if fig_rows > 1:
                        ax = axes[elev_idx, ax_idx]
                    else:
                        ax = axes[ax_idx]
                    
                    # Display the SAR image
                    ax.imshow(sarimgs[0, 0].cpu().numpy(), cmap='gray')
                    ax.set_title(f'Elev: {elev}°, Azim: {azim}°')
                    ax.axis('off')
            
            # Save 3D mesh visualization for select angles
            if azim % 90 == 0:
                mesh_name = f'spacestation_elev{elev}_azim{azim}'
                # Use meshvisual if it exists, otherwise use the standard visualization
                if hasattr(model, 'meshvisual'):
                    model.meshvisual(epoch=0, meshname=mesh_name)
            
            progress_bar.update(1)
    
    # Save the summary figure
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gaofen3_sar_summary.png'), dpi=300)
    plt.close()
    
    # Save the 3D mesh
    model.meshsave(epoch=0, meshname="spacestation")
    
    # Generate metadata file with simulation parameters
    with open(os.path.join(output_dir, 'gaofen3_simulation_parameters.txt'), 'w') as f:
        f.write("Gaofen-3 SAR Simulation Parameters\n")
        f.write("==================================\n\n")
        f.write(f"Wavelength: {wavelength} m (C-band)\n")
        f.write(f"Frequency: {frequency/1e9} GHz\n")
        f.write(f"Slant range resolution: {slant_range_resolution} m\n")
        f.write(f"Slant range pixel spacing: {slant_range_pixel_spacing} m\n")
        f.write(f"Azimuth resolution: {azimuth_resolution} m\n")
        f.write(f"Azimuth pixel spacing: {azimuth_pixel_spacing} m\n")
        f.write(f"Bandwidth: {bandwidth/1e6} MHz\n")
        f.write(f"Sampling rate: {sampling_rate/1e6} MHz\n")
        f.write(f"PRF: {prf} Hz\n")
        f.write(f"Imaging distance: {distance/1000} km\n")
        f.write(f"Image dimensions: {imagesize}×{imagesize} pixels\n")
        f.write(f"Ground coverage: ~{imagesize*slant_range_pixel_spacing}m × ~{imagesize*azimuth_pixel_spacing}m\n")
    
    print(f"\nAll Gaofen-3 SAR images generated and saved to {output_dir}")
    print(f"A summary of the simulation parameters is available in {output_dir}/gaofen3_simulation_parameters.txt")