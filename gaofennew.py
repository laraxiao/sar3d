# use scale factor of 400 is good
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
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def auto_center_and_scale_model(mesh, device, target_scale=400):
    """Center the model and scale it to fit properly in the image
       Using a larger target_scale (0.8) to make the model larger in frame"""
    with torch.no_grad():
        verts = mesh.verts_padded()
        min_coords, _ = verts.min(dim=1)
        max_coords, _ = verts.max(dim=1)
        size = (max_coords - min_coords).max().item()
        center = (min_coords + max_coords) / 2
        scale_factor = target_scale / size
        centered_verts = verts - center.unsqueeze(1)
        scaled_verts = centered_verts * scale_factor
        updated_mesh = mesh.update_padded(scaled_verts)
        return updated_mesh, scale_factor

def save_tensor_as_csv(tensor, filepath):
    """Save a PyTorch tensor as a CSV file"""
    np_array = tensor.cpu().detach().numpy()
    np.savetxt(filepath, np_array, delimiter=',')
    return filepath

def save_high_quality_image(tensor, filepath, normalize=True, dpi=300):
    """Save tensor as high-quality image using matplotlib"""
    # Convert tensor to numpy array
    if tensor.dim() == 4:  # batch, channel, H, W
        array = tensor[0, 0].cpu().detach().numpy()
    elif tensor.dim() == 3:  # channel, H, W
        array = tensor[0].cpu().detach().numpy()
    else:  # H, W
        array = tensor.cpu().detach().numpy()
    
    # Create high-quality figure
    plt.figure(figsize=(10, 10))
    if normalize:
        plt.imshow(array, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(array, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    return filepath

if __name__ == '__main__':
    # ==============================================
    # Initialize random seed
    set_seed()

    # Setup GPU/CPU device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")

    # Set paths
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "teapot.obj")
    if os.path.exists(obj_filename):
        print(f"3D model found: {obj_filename}")
    else:
        print(f"3D model not found: {obj_filename}")
        print("Please place your space station 3D model at this location")
        exit(1)

    # Output directory for SAR images
    output_dir = './teapot_enhanced_03.27'
    hq_dir = os.path.join(output_dir, 'high_quality')
    csv_dir_sar = os.path.join(output_dir, 'csv_sar_data')
    csv_dir_sil = os.path.join(output_dir, 'csv_sil_data')
    raw_dir = os.path.join(output_dir, 'raw_output')  # For unprocessed output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hq_dir, exist_ok=True)
    os.makedirs(csv_dir_sar, exist_ok=True)
    os.makedirs(csv_dir_sil, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    # ==============================================
    # Gaofen-3 SAR parameters with HIGH RESOLUTION settings
    wavelength = 0.055517
    frequency = 5.4e9
    slant_range_resolution = 0.63
    slant_range_pixel_spacing = 0.56
    azimuth_resolution = 2.5
    azimuth_pixel_spacing = 2.0
    bandwidth = 240e6
    sampling_rate = 267e6
    prf = 1745.000488
    distance = 5000  # For visualization (not actual satellite distance)
    
    # High-resolution image size
    imagesize = 1024  # Much higher resolution for better quality
    scalefactor = 1.0
    
    # Create simulation settings
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
        'prf': prf,
        # Add these values to force consistent behavior
        'debug_mode': True,  # Enable extra checking
        'imgsamfactor': 1,   # Ensure consistent sampling
        'faces_per_pixel': 5  # More samples for better quality
    }

    # ==============================================
    print(f"Creating SARNNM model with HIGH QUALITY settings...")
    '''
    Override the SARNNM class to force consistent camera parameters
    try:
        # Try to monkey-patch the SARNNM renderer function to ensure consistent cameras
        original_sarnnm_init = SARNNM.__init__
        def patched_sarnnm_init(self, *args, **kwargs):
            original_sarnnm_init(self, *args, **kwargs)
            print("Patched SARNNM renderer for consistent cameras")
            self.force_camera_consistency = True
        SARNNM.__init__ = patched_sarnnm_init
        print("Successfully patched SARNNM for consistent cameras")
    except:
        print("Warning: Could not patch SARNNM class for camera consistency")
    
    # Create model for rendering'
    '''
    model = SARNNM(
        obj_filename=obj_filename, 
        simsetting=simsetting, 
        EPNum=0, 
        savepath=output_dir, 
        device=device
    )
    
    # Auto-center and scale model - using larger scale factor
    model.src_mesh, effective_scale = auto_center_and_scale_model(model.src_mesh, device, target_scale=400)
    print(f"Model auto-centered and scaled by factor: {effective_scale:.6f}")
    
    # ==============================================
    # Define viewing angles for space station imaging
    elevations = [25, 35, 45, 55]  # Better coverage of operational angles
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]  # 45° increments (good balance)
    
    # Prepare CSV metadata file
    metadata_csv_path = os.path.join(csv_dir_sil, 'gaofen3_sar_metadata.csv')
    
    # Generate images for each angle combination
    print("Generating HIGH QUALITY Gaofen-3 SAR images...")
    progress_bar = tqdm(total=len(elevations)*len(azimuths))
    
    # Create dataframe to store all metadata
    all_metadata = []
    
    for elev_idx, elev in enumerate(elevations):
        for azim_idx, azim in enumerate(azimuths):
            # Create input sample with angle information
            sample = {
                'elev': torch.tensor([elev], device=device, dtype=torch.float32),
                'azim': torch.tensor([azim], device=device, dtype=torch.float32),
                'samplename': [f'gaofen3_elev{elev}_azim{azim}.png'],
                'realimg': torch.zeros(1, 1, imagesize, imagesize, device=device)
            }
            
            # Generate the SAR image
            try:
                with torch.no_grad():
                    # Print debug info for this rendering
                    print(f"\nRendering elev={elev}, azim={azim} at {imagesize}x{imagesize}")
                    
                    # Generate SAR and silhouette images with size verification
                    sarimgs, silimgs = model.sarimgnnsim(sample, model.src_mesh)
                    
                    # Debug size info
                    print(f"SAR image size: {sarimgs.shape}, Silhouette size: {silimgs.shape}")
                    
                    # Verify both images have the same dimensions
                    if sarimgs.shape != silimgs.shape:
                        print(f"WARNING: Dimension mismatch! SAR={sarimgs.shape}, Sil={silimgs.shape}")
                        # Resize to match if needed
                        if silimgs.shape[2] != sarimgs.shape[2] or silimgs.shape[3] != sarimgs.shape[3]:
                            silimgs = torch.nn.functional.interpolate(
                                silimgs, size=(sarimgs.shape[2], sarimgs.shape[3]), 
                                mode='bilinear', align_corners=False
                            )
                    
                    # Optional netPix processing for SAR
                    if hasattr(model, 'netPix'):
                        sarimgs_processed = model.netPix(sarimgs)
                    else:
                        sarimgs_processed = sarimgs
                    
                    # Base filenames
                    base_filename = f'gaofen3-{azim}-{elev}-simulated'
                    
                    # Save standard output
                    sar_png_path = os.path.join(output_dir, f'{base_filename}_sar.png')
                    sil_png_path = os.path.join(output_dir, f'{base_filename}_silhouette.png')
                    vutils.save_image(sarimgs_processed[0], fp=sar_png_path, normalize=True)
                    vutils.save_image(silimgs[0], fp=sil_png_path, normalize=True)
                    
                    # Save raw unprocessed output
                    raw_sar_path = os.path.join(raw_dir, f'{base_filename}_raw_sar.png')
                    vutils.save_image(sarimgs[0], fp=raw_sar_path, normalize=False)
                    
                    # Save high-quality output
                    hq_sar_path = os.path.join(hq_dir, f'{base_filename}_sar_hq.png')
                    hq_sil_path = os.path.join(hq_dir, f'{base_filename}_silhouette_hq.png')
                    save_high_quality_image(sarimgs_processed[0], hq_sar_path, normalize=True, dpi=300)
                    save_high_quality_image(silimgs[0], hq_sil_path, normalize=True, dpi=300)
                    
                    # Save CSV data
                    sar_csv_path = os.path.join(csv_dir_sar, f'{base_filename}_sar.csv')
                    sil_csv_path = os.path.join(csv_dir_sil, f'{base_filename}_silhouette.csv')
                    save_tensor_as_csv(sarimgs_processed[0, 0], sar_csv_path)
                    save_tensor_as_csv(silimgs[0, 0], sil_csv_path)
                    
                    # Add metadata
                    all_metadata.append({
                        'filename': base_filename,
                        'elevation': elev,
                        'azimuth': azim,
                        'sar_image_shape': f"{sarimgs.shape[2]}x{sarimgs.shape[3]}",
                        'silhouette_image_shape': f"{silimgs.shape[2]}x{silimgs.shape[3]}",
                        'sar_csv_path': sar_csv_path,
                        'silhouette_csv_path': sil_csv_path,
                        'hq_sar_path': hq_sar_path,
                        'hq_silhouette_path': hq_sil_path,
                        'wavelength': wavelength,
                        'frequency': frequency,
                        'range_resolution': slant_range_resolution,
                        'range_pixel_spacing': slant_range_pixel_spacing,
                        'azimuth_resolution': azimuth_resolution,
                        'azimuth_pixel_spacing': azimuth_pixel_spacing,
                        'imagesize': imagesize,
                        'effective_scale': effective_scale
                    })
                    
                    # Save a side-by-side comparison for quick visual check
                    plt.figure(figsize=(20, 8))
                    
                    plt.subplot(1, 2, 1)
                    plt.imshow(sarimgs_processed[0, 0].cpu().numpy(), cmap='gray')
                    plt.title(f"SAR Image: Elev={elev}°, Azim={azim}°")
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(silimgs[0, 0].cpu().numpy(), cmap='gray')
                    plt.title(f"Silhouette: Elev={elev}°, Azim={azim}°")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'{base_filename}_comparison.png'), dpi=150)
                    plt.close()
                    
            except Exception as e:
                print(f"ERROR rendering at elev={elev}, azim={azim}:\n{str(e)}")
                import traceback
                traceback.print_exc()
                
                # Add error entry to metadata
                all_metadata.append({
                    'filename': f'gaofen3_elev{elev}_azim{azim}_error',
                    'elevation': elev,
                    'azimuth': azim,
                    'sar_csv_path': 'ERROR',
                    'silhouette_csv_path': 'ERROR',
                    'error': str(e)
                })
            
            progress_bar.update(1)
    
    # Save metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_csv_path, index=False)
    
    print(f"\nHigh-quality Gaofen-3 SAR images generated")
    print(f"- Standard images: {output_dir}")
    print(f"- High-quality images: {hq_dir}")
    print(f"- Raw unprocessed images: {raw_dir}")
    print(f"- CSV data: {csv_dir_sar}")
