#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

from pbr import CubemapLight, get_brdf_lut, pbr_shading
import torch
import torch.nn.functional as F
from scene import Scene
import time
import numpy as np
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gs_ir import recon_occlusion, IrradianceVolumes

def render_fps(dataset : ModelParams, checkpoint_path: str, pipeline : PipelineParams, 
    light: CubemapLight,
    pbr: bool = False,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    indirect: bool = False,
    
    renders_per_view : int = 100
) -> None:
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)
        cubemap = CubemapLight(base_res=256).cuda()
        
        # occlusion volumes
        filepath = os.path.join(os.path.dirname(checkpoint_path), "occlusion_volumes.pth")
        print(f"begin to load occlusion volumes from {filepath}")
        if os.path.exists(filepath):
            occlusion_volumes = torch.load(filepath)
        else:
            occlusion_volumes = None

        if occlusion_volumes is not None:
            occlusion_ids = occlusion_volumes["occlusion_ids"]
            occlusion_coefficients = occlusion_volumes["occlusion_coefficients"]
            occlusion_degree = occlusion_volumes["degree"]
            bound = occlusion_volumes["bound"]
        else:
            bound = 0.5

        aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
        irradiance_volumes = IrradianceVolumes(aabb=aabb).cuda()
        
        checkpoint = torch.load(checkpoint_path)
        model_params = checkpoint["gaussians"]
        cubemap_params = checkpoint["cubemap"]
        irradiance_volumes_params = checkpoint["irradiance_volumes"]
        
        gaussians.restore(model_params)
        cubemap.load_state_dict(cubemap_params)
        cubemap.eval()
        irradiance_volumes.load_state_dict(irradiance_volumes_params)
        irradiance_volumes.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_times = []
        views = scene.getTestCameras()
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            for i in range(renders_per_view):
                t1 = time.time()

                rendering_result = render(
                    viewpoint_camera=view,
                    pc=scene.gaussians,
                    pipe=pipeline,
                    bg_color=background,
                    inference=True,
                    pad_normal=True,
                    derive_normal=True,
                )
                if pbr:
                    # normal from point cloud
                    canonical_rays = scene.get_canonical_rays()
                    H, W = view.image_height, view.image_width
                    c2w = torch.inverse(view.world_view_transform.T)  # [4, 4]
                    view_dirs = -(
                        (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
                        .sum(dim=-1)
                        .reshape(H, W, 3)
                    )  # [H, W, 3]
                    normal_map = rendering_result["normal_map"]
                    depth_map = rendering_result["depth_map"]
                    if indirect and occlusion_volumes is not None:
                        points = (
                            (-view_dirs.reshape(-1, 3) * depth_map.reshape(-1, 1) + c2w[:3, 3])
                            .clamp(min=-bound, max=bound)
                            .contiguous()
                        )  # [HW, 3]
                        occlusion = recon_occlusion(
                            H=H,
                            W=W,
                            points=points,
                            normals=normal_map.permute(1, 2, 0).reshape(-1, 3).contiguous(),
                            bound=bound,
                            occlusion_coefficients=occlusion_coefficients,
                            occlusion_ids=occlusion_ids,
                            aabb=aabb,
                            degree=occlusion_degree,
                        ).reshape(H, W, 1)
                        irradiance = irradiance_volumes.query_irradiance(
                            points=points.reshape(-1, 3).contiguous(),
                            normals=normal_map.permute(1, 2, 0).reshape(-1, 3).contiguous(),
                        ).reshape(H, W, -1)
                    else:
                        occlusion = torch.ones_like(depth_map).permute(1, 2, 0)  # [H, W, 1]
                        irradiance = torch.zeros_like(depth_map).permute(1, 2, 0)  # [H, W, 1]

                    brdf_lut = get_brdf_lut().cuda()
                    normal_mask = rendering_result["normal_mask"]
                    albedo_map = rendering_result["albedo_map"]  # [3, H, W]
                    roughness_map = rendering_result["roughness_map"]  # [1, H, W]
                    metallic_map = rendering_result["metallic_map"]  # [1, H, W] 
                    pbr_result = pbr_shading(
                        light=light,
                        normals=normal_map.permute(1, 2, 0),  # [H, W, 3]
                        view_dirs=view_dirs,
                        mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
                        albedo=albedo_map.permute(1, 2, 0),  # [H, W, 3]
                        roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
                        metallic=metallic_map.permute(1, 2, 0) if metallic else None,  # [H, W, 1]
                        tone=tone,
                        gamma=gamma,
                        occlusion=occlusion,
                        irradiance=irradiance,
                        brdf_lut=brdf_lut,
                    )
                    render_rgb = (
                        pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1)
                    )  # [3, H, W]
                    background_ = torch.zeros_like(render_rgb) + background[:, None, None]
                    render_rgb = torch.where(
                        normal_mask,
                        render_rgb,
                        background_,
                    )
                
                render_time = time.time() - t1
                render_times.append(render_time)
        with open(dataset.model_path + "/fps.txt", 'w') as fp:
            fps = 1.0/np.array(render_times).mean()
            fp.write('fps:{}\n'.format(fps))
            fp.write('count:{}\n'.format(len(gaussians.get_xyz)))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--pbr", action="store_true", help="Enable pbr rendering for NVS evaluation and export BRDF map.")
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    parser.add_argument("--indirect", action="store_true", help="Enable indirect diffuse modeling.")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Measuring FPS for " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_fps(model.extract(args), args.iteration, pipeline.extract(args))