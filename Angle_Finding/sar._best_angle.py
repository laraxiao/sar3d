import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardFlatShader,
    TexturesVertex
)
import torchvision.utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt

class SARBestAngleFinder:
    def __init__(self, obj_path, output_dir="./sar_best_view", device="cuda"):
        """
        Initialize the SAR Best Angle Finder.
        
        Args:
            obj_path: Path to the 3D model OBJ file
            output_dir: Directory to save results
            device: Device to use for computation ("cuda" or "cpu")
        """
        # Check if CUDA is available
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device = "cpu"
        
        self.device = torch.device(device)
        self.obj_path = obj_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Load the 3D model
        print(f"Loading 3D model from {obj_path}...")
        self.load_mesh()
        
        # Default SAR simulation settings
        self.distance = 2.7  # Camera distance
        self.image_size = 256  # Render resolution
    
    def load_mesh(self):
        """Load the 3D model mesh from file."""
        # Load mesh from OBJ file
        verts, faces, aux = load_obj(self.obj_path)
        verts = verts.to(self.device)
        faces = faces.verts_idx.to(self.device)
        
        # Create a texture based on vertex positions for visualization
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb)
        
        # Create a mesh
        self.mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
        
        # For compatibility with your existing code
        self.verts = verts
        self.faces = faces
    
    def get_differentiable_renderer(self, image_size=None):
        """Create a differentiable renderer for angle optimization."""
        if image_size is None:
            image_size = self.image_size
            
        device = self.device
        
        # Rasterization settings for differentiable rendering
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0, 
            faces_per_pixel=1,
        )
        
        # Create a differentiable renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=device,
                cameras=None,
                lights=PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            )
        )
        
        return renderer
    
    def simulate_sar(self, rendered_image):
        """
        Simulate SAR characteristics from a standard rendered image.
        
        In a full implementation, this would use your SimPix network.
        Here we use a simplified approach for demonstration.
        """
        # Extract RGB channels (first 3 channels)
        if rendered_image.shape[-1] > 3:
            rgb_image = rendered_image[..., :3]
        else:
            rgb_image = rendered_image
        
        # Convert to grayscale (simplified SAR simulation)
        # In your actual code, replace this with your SimPix network
        gray_image = 0.299 * rgb_image[..., 0] + 0.587 * rgb_image[..., 1] + 0.114 * rgb_image[..., 2]
        
        # Add some noise and contrast to simulate SAR characteristics
        noise = torch.randn_like(gray_image) * 0.05
        sar_image = torch.clamp(gray_image * 1.2 + noise, 0, 1)
        
        # Return as [B, 1, H, W] format for further processing
        return sar_image.unsqueeze(1)
    
    def compute_sar_visibility_score(self, sar_image):
        """
        Compute visibility score optimized for SAR imagery.
        
        Args:
            sar_image: Processed SAR image tensor [B, C, H, W]
            
        Returns:
            score: Single scalar visibility score (higher is better)
        """
        # Convert to grayscale if necessary
        if sar_image.shape[1] > 1:
            sar_image = sar_image[:, 0:1]  # Use first channel
            
        # 1. Target intensity - we want bright returns from the object
        intensity_score = sar_image.mean()
        
        # 2. Local contrast - measures feature definition
        # Compute local variance using convolution
        mean_filter = torch.ones(5, 5, device=sar_image.device) / 25
        mean_filter = mean_filter.view(1, 1, 5, 5)
        
        # Calculate local mean
        local_mean = F.conv2d(sar_image, mean_filter, padding=2)
        # Calculate local variance
        local_variance = F.conv2d(sar_image**2, mean_filter, padding=2) - local_mean**2
        contrast_score = local_variance.mean()
        
        # 3. Edge prominence - SAR images benefit from clear structural edges
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=sar_image.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=sar_image.device).float()
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        edges_x = F.conv2d(sar_image, sobel_x, padding=1)
        edges_y = F.conv2d(sar_image, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_score = edge_magnitude.mean()
        
        # 4. Information content - measured via entropy
        # Quantize image into bins to calculate histogram
        bins = 32
        sar_image_flat = sar_image.view(sar_image.shape[0], -1)
        min_val = torch.min(sar_image_flat, dim=1, keepdim=True)[0]
        max_val = torch.max(sar_image_flat, dim=1, keepdim=True)[0]
        sar_image_norm = (sar_image_flat - min_val) / (max_val - min_val + 1e-8)
        
        # Quantize to compute histogram
        sar_image_quantized = torch.floor(sar_image_norm * bins).long()
        
        # Compute histogram and entropy
        entropy_score = 0.0
        for b in range(sar_image.shape[0]):
            hist = torch.zeros(bins, device=sar_image.device)
            for i in range(bins):
                hist[i] = (sar_image_quantized[b] == i).float().mean()
            
            # Add small constant to avoid log(0)
            hist = hist + 1e-10
            hist = hist / hist.sum()
            
            # Calculate entropy
            entropy = -torch.sum(hist * torch.log2(hist))
            entropy_score += entropy

        entropy_score = entropy_score / sar_image.shape[0]
        
        # Combine all scores with appropriate weights
        # These weights should be tuned for specific SAR applications
        w_intensity = 0.2
        w_contrast = 0.3
        w_edge = 0.3
        w_entropy = 0.2
        
        total_score = (w_intensity * intensity_score + 
                       w_contrast * contrast_score + 
                       w_edge * edge_score + 
                       w_entropy * entropy_score)
        
        return total_score
    
    def find_best_angle_gradient_descent(self, iterations=100, lr=0.5, num_starts=5):
        """
        Find the best viewing angle for SAR imagery using gradient descent.
        
        Args:
            iterations: Total number of optimization steps across all starts
            lr: Learning rate for optimization
            num_starts: Number of random starting points to try
            
        Returns:
            best_angle: Dictionary with elevation and azimuth values
            best_image: Best simulated SAR image
        """
        device = self.device
        
        # 1. Initialize parameters with multiple starting points
        # Using multiple starting points helps avoid local minima
        elevations = []
        azimuths = []
        
        for i in range(num_starts):
            # Distribute starting points across the viewing sphere
            elev = torch.tensor(20.0 + i * 15.0, device=device, requires_grad=True)
            azim = torch.tensor(i * 72.0, device=device, requires_grad=True)  # 360/5 = 72
            
            elevations.append(elev)
            azimuths.append(azim)
        
        # 2. Setup optimizers
        optimizers = [torch.optim.Adam([elevations[i], azimuths[i]], lr=lr) 
                      for i in range(num_starts)]
        
        # 3. Setup renderer
        renderer = self.get_differentiable_renderer()
        
        # 4. Track best results
        best_score = -float('inf')
        best_elevation = None
        best_azimuth = None
        best_image = None
        
        # 5. Distribute iterations among starting points
        iters_per_start = iterations // num_starts
        
        # 6. Run optimization for each starting point
        for start_idx in range(num_starts):
            elevation = elevations[start_idx]
            azimuth = azimuths[start_idx]
            optimizer = optimizers[start_idx]
            
            print(f"Starting point {start_idx+1}/{num_starts}: " 
                  f"Elevation={elevation.item():.1f}, Azimuth={azimuth.item():.1f}")
            
            # Optimization loop
            for iteration in tqdm(range(iters_per_start)):
                # a. Setup camera for current angles
                R, T = look_at_view_transform(
                    dist=self.distance, 
                    elev=elevation.unsqueeze(0), 
                    azim=azimuth.unsqueeze(0), 
                    device=device
                )
                
                cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
                
                # b. Render standard image (differentiable)
                rendered_image = renderer(self.mesh, cameras=cameras)
                
                # c. Convert to SAR using simulation
                sar_image = self.simulate_sar(rendered_image)
                
                # d. Compute SAR-specific visibility score
                score = self.compute_sar_visibility_score(sar_image)
                
                # e. Gradient ascent step (maximize score)
                loss = -score  # Negate for gradient descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # f. Apply constraints to angles
                with torch.no_grad():
                    elevation.clamp_(5.0, 85.0)  # Avoid extreme angles
                    azimuth.data = (azimuth % 360)  # Wrap to [0, 360]
                
                # g. Track best result
                current_score = score.item()
                if current_score > best_score:
                    best_score = current_score
                    best_elevation = elevation.item()
                    best_azimuth = azimuth.item()
                    best_image = sar_image.detach().clone()
        
        # 7. Report final results
        print(f"\nBest SAR viewing angle: Elevation={best_elevation:.2f}, Azimuth={best_azimuth:.2f}")
        print(f"SAR visibility score: {best_score:.4f}")
        
        # 8. Save results
        self.save_results(best_elevation, best_azimuth, best_score, best_image)
        
        return {'elevation': best_elevation, 'azimuth': best_azimuth}, best_image
    
    def save_results(self, elevation, azimuth, score, sar_image):
        """Save the results of the best angle optimization."""
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Save best SAR image
        best_img_path = os.path.join(self.output_dir, "best_sar_view.png")
        vutils.save_image(sar_image, fp=best_img_path)
        
        # Also save a standard rendered image from this angle
        with torch.no_grad():
            R, T = look_at_view_transform(
                dist=self.distance, 
                elev=torch.tensor([elevation], device=self.device), 
                azim=torch.tensor([azimuth], device=self.device),
                device=self.device
            )
            cameras = FoVPerspectiveCameras(R=R, T=T, device=self.device)
            
            renderer = self.get_differentiable_renderer()
            standard_render = renderer(self.mesh, cameras=cameras)
            
            standard_img_path = os.path.join(self.output_dir, "best_standard_view.png")
            vutils.save_image(standard_render[0, ..., :3], fp=standard_img_path)
        
        # Create a visualization of multiple angles for comparison
        self.visualize_angle_comparison(elevation, azimuth)
        
        # Save angle information
        with open(os.path.join(self.output_dir, "best_angle_info.txt"), 'w') as f:
            f.write(f"Elevation: {elevation:.2f}\n")
            f.write(f"Azimuth: {azimuth:.2f}\n")
            f.write(f"Distance: {self.distance:.2f}\n")
            f.write(f"SAR Visibility Score: {score:.4f}\n")
        
        print(f"Results saved to {self.output_dir}")
    
    def visualize_angle_comparison(self, best_elev, best_azim):
        """Generate a comparison grid of different viewing angles."""
        device = self.device
        renderer = self.get_differentiable_renderer(image_size=200)
        
        # Create a grid of angles to visualize
        elevations = [20, best_elev, 60]
        azimuths = [0, best_azim, 180]
        
        fig, axes = plt.subplots(len(elevations), len(azimuths), figsize=(12, 12))
        
        for i, elev in enumerate(elevations):
            for j, azim in enumerate(azimuths):
                # Highlight the best angle
                is_best = (elev == best_elev and azim == best_azim)
                
                # Render from this angle
                with torch.no_grad():
                    R, T = look_at_view_transform(
                        dist=self.distance, 
                        elev=torch.tensor([elev], device=device), 
                        azim=torch.tensor([azim], device=device),
                        device=device
                    )
                    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
                    
                    # Render both standard and SAR
                    standard_render = renderer(self.mesh, cameras=cameras)
                    sar_image = self.simulate_sar(standard_render)
                    
                    # Convert to numpy for matplotlib
                    sar_np = sar_image[0, 0].cpu().numpy()
                    
                    # Plot
                    axes[i, j].imshow(sar_np, cmap='gray')
                    axes[i, j].set_title(f"Elev: {elev:.0f}, Azim: {azim:.0f}")
                    axes[i, j].axis('off')
                    
                    # Highlight best angle
                    if is_best:
                        axes[i, j].set_title(f"BEST: Elev: {elev:.0f}, Azim: {azim:.0f}", color='red')
                        for spine in axes[i, j].spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "angle_comparison.png"))
        plt.close()

    def find_best_angle_sampling(self, num_samples=100):
        """
        Find the best viewing angle using uniform sampling (alternative to gradient descent).
        
        Args:
            num_samples: Number of angles to sample
            
        Returns:
            best_angle: Dictionary with elevation and azimuth values
            best_image: Best simulated SAR image
        """
        device = self.device
        renderer = self.get_differentiable_renderer()
        
        # Sample uniformly
        n_elev = int(np.sqrt(num_samples))
        n_azim = int(np.sqrt(num_samples))
        
        elevations = torch.linspace(5, 85, n_elev, device=device)
        azimuths = torch.linspace(0, 360, n_azim, device=device)
        
        # Track best results
        best_score = -float('inf')
        best_elevation = None
        best_azimuth = None
        best_image = None
        
        print(f"Sampling {n_elev * n_azim} angles...")
        
        # Evaluate all combinations
        for elev in tqdm(elevations):
            for azim in azimuths:
                # Render from this angle
                with torch.no_grad():
                    R, T = look_at_view_transform(
                        dist=self.distance, 
                        elev=elev.unsqueeze(0), 
                        azim=azim.unsqueeze(0), 
                        device=device
                    )
                    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
                    
                    # Render and simulate SAR
                    rendered_image = renderer(self.mesh, cameras=cameras)
                    sar_image = self.simulate_sar(rendered_image)
                    
                    # Compute score
                    score = self.compute_sar_visibility_score(sar_image)
                    
                    # Track best
                    current_score = score.item()
                    if current_score > best_score:
                        best_score = current_score
                        best_elevation = elev.item()
                        best_azimuth = azim.item()
                        best_image = sar_image.detach().clone()
        
        print(f"\nBest SAR viewing angle: Elevation={best_elevation:.2f}, Azimuth={best_azimuth:.2f}")
        print(f"SAR visibility score: {best_score:.4f}")
        
        # Save results
        self.save_results(best_elevation, best_azimuth, best_score, best_image)
        
        return {'elevation': best_elevation, 'azimuth': best_azimuth}, best_image

def main():
    parser = argparse.ArgumentParser(description="Find best SAR viewing angle for a 3D model")
    parser.add_argument('--obj_path', required=True, help='Path to the OBJ file')
    parser.add_argument('--output_dir', default='./sar_best_view', help='Directory to save results')
    parser.add_argument('--method', default='gradient', choices=['gradient', 'sampling'], 
                       help='Method for finding best angle (gradient descent or sampling)')
    parser.add_argument('--iterations', type=int, default=100, 
                       help='Number of iterations (for gradient descent)')
    parser.add_argument('--samples', type=int, default=100, 
                       help='Number of samples (for sampling method)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], 
                       help='Device to use for computation')
    
    args = parser.parse_args()
    
    # Initialize the angle finder
    finder = SARBestAngleFinder(
        obj_path=args.obj_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Find best angle using selected method
    if args.method == 'gradient':
        finder.find_best_angle_gradient_descent(iterations=args.iterations)
    else:
        finder.find_best_angle_sampling(num_samples=args.samples)

if __name__ == "__main__":
    main()
