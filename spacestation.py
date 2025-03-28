import os
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    look_at_view_transform,
    MeshRasterizer, MeshRenderer,
    DirectionalLights, RasterizationSettings,
    SoftPhongShader, SoftSilhouetteShader,
    BlendParams,
)
import matplotlib.pyplot as plt
from custom_shaders import SARSoftPhongShader

def generate_space_station_renders(
    num_elev: int = 6,
    num_azim: int = 6,
    data_dir: str = "./SpaceStation",
    azimuth_range: float = 180,
    elevation_range: float = 30,
    device: str = "cuda",
    image_size: int = 124,
    distance: float = 90.0,
    output_dir: str = "./SpaceStation",
):
    # Load the mesh
    obj_filename = os.path.join(data_dir, "space_station.obj")
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # IMPORTANT: Keep this normalization only if your training code does similar normalization
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    mesh.scale_verts_((1.0 / float(scale)))

    # Create output directories
    os.makedirs(os.path.join(output_dir, "sar_images"), exist_ok=True)
    
    # Generate angles
    elev = torch.linspace(-elevation_range, elevation_range, num_elev, device=device)
    azim = torch.linspace(-azimuth_range/2, azimuth_range/2, num_azim, device=device)

    # Rasterization settings matching training code
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        max_faces_per_bin=100000
    )
    
    # Lighting setup
    lights = DirectionalLights(device=device, direction=[[0.0, 0.0, 1.0]])
    
    # Process each viewpoint
    for i in range(num_elev):
        for j in range(num_azim):
            # Transform angles to match training convention
            current_elev = 90.0 - elev[i].item()  # Converts from zenith-based to horizon-based
            current_azim = azim[j].item() + 180.0  # Shifts reference point
            
            # Camera setup
            R, T = look_at_view_transform(dist=distance, elev=current_elev, azim=current_azim)
            cameras = FoVOrthographicCameras(device=device, R=R, T=T)
            
            # Create SAR renderer
            sar_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                shader=SARSoftPhongShader(device=device, lights=lights)
            )
            
            # Render SAR image
            sar = sar_renderer(mesh, cameras=cameras, lights=lights).squeeze(0)
            sar_grayscale = sar[..., 0].cpu().numpy()
            
            # Save SAR image as CSV
            elev_val = int(elev[i].item())
            azim_val = int(azim[j].item() - 180)
            sar_filename = f"space_station-elev-{elev_val}-azim-{azim_val}.csv"
            sar_filepath = os.path.join(output_dir, "sar_images", sar_filename)
            np.savetxt(sar_filepath, sar_grayscale, delimiter=",")

    return "Renders completed"