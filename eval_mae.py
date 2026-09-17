from pathlib import Path
import traceback
import os, sys
import re
from natsort import natsorted
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from argparse import ArgumentParser

def angular_error_map(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
  # Per-pixel norms → [H, W]
  norm_pred = torch.linalg.norm(pred, dim=0)
  norm_gt   = torch.linalg.norm(gt, dim=0)

  # Valid pixels
  valid = (norm_pred > eps) & (norm_gt > eps)

  # Allocate normalized tensors
  normalized_pred = torch.zeros_like(pred)
  normalized_gt   = torch.zeros_like(gt)

  # Normalize ONLY valid pixels
  normalized_pred[:, valid] = pred[:, valid] / norm_pred[valid]
  normalized_gt[:, valid]   = gt[:, valid] / norm_gt[valid]

  # Per-pixel dot product → [H, W]
  dot = torch.sum(normalized_pred * normalized_gt, dim=0)

  # Cosine of angle
  # Angle in degrees
  dot = torch.clamp(dot, -1.0, 1.0)
  ang_deg = torch.acos(dot) * (180.0 / torch.pi)

  ang_deg[~valid] = 0.0
  
  return ang_deg


def compute_mae(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8):
  """
  pred, gt: torch tensors of shape [1, 3, H, W]
            pred assumed in [0,255], gt in [0,65535]

  returns: torch tensor of shape [H, W] with angular error in degrees
            invalid pixels set to NaN
  """

  if pred.ndim != 4 or gt.ndim != 4:
    raise ValueError(f"Expected 4D tensors [B,3,H,W], got pred {pred.shape}, gt {gt.shape}")

  if pred.shape[0] != 1 or gt.shape[0] != 1:
    raise ValueError("This function expects batch size = 1")

  if pred.shape[1] != 3 or gt.shape[1] != 3:
    raise ValueError("Expected channel dimension = 3")

  # Remove batch dimension → [3, H, W]
  pred = pred[0]
  gt = gt[0]

  # Convert to float + normalize
  pred = pred.to(torch.float32)
  gt   = gt.to(torch.float32)

  angular_error = angular_error_map(pred, gt, eps)

  return angular_error[~torch.isnan(angular_error)].mean().item()

def extract_iteration(key: str):
  """
  Extract a numeric iteration from keys like:
    'ref_gs_30000' -> 30000
    'ours_31000'   -> 31000
  If none found, return -1 so it loses in max().
  """
  m = re.findall(r"(\d+)", key)
  if not m:
    return -1
  return int(m[-1])

def pick_best_key(results):
  """
  Pick the key with the highest numeric iteration.
  Fallback: first key if parsing fails.
  """
  if not results:
    return None

  best = max(results, key=lambda k: extract_iteration(k))
  return best

def evaluate(model_path, source_path):
  try:          
    if not model_path.is_dir():
      print(f"[!] {model_path} is not a directory")
      return

    test_dir = Path(model_path) / "test"
    if not test_dir.exists():
      print(f"[!] No test directory found in {model_path}, skipping.")
      return
    
    best_method = pick_best_key(os.listdir(test_dir))
    if best_method is None:
      print(f"[!] No valid method directories found in {test_dir}, skipping.")
      return

    method_dir = test_dir / best_method

    normal_renders_dir = method_dir / "normals"
    normal_gts_dir = Path(source_path) / "test"
    normal_renders, normal_gts, alphas = readNormalsImages(normal_renders_dir, normal_gts_dir)

    mean_angular_error = 0.0
    for idx in tqdm(range(len(normal_renders)), desc="Generating tiles"):
      normal_gt = normal_gts[idx]
      normal_gt = (normal_gt-0.5)*2
      normal_gt = normal_gt[0].to(torch.float32)

      normal_render = normal_renders[idx]
      normal_render = (normal_render-0.5)*2
      normal_render = normal_render[0].to(torch.float32)

      angular_error = angular_error_map(normal_render, normal_gt)

      if len(alphas) > idx:
        alpha = alphas[idx][0,0,:,:]
        angular_error[alpha < 1.0] = 0

      mean_angular_error += angular_error.mean().item()

    mean_angular_error /= len(normal_renders)
    print(f"    Normal Angular Error (MAE): {mean_angular_error:.4f} degrees")
    with open(model_path / "mae.txt", 'w') as f:
      f.write(f"{mean_angular_error:.4f}\n")

  except Exception as e:
    print(f"[!] Unable to compute metrics for model {model_path}")
    print(e)
    traceback.print_exc()

def readNormalsImages(renders_dir, gt_dir):
  render_normals = []
  for fname in natsorted(os.listdir(renders_dir)):
    normal = Image.open(renders_dir / fname)
    render_normals.append(tf.to_tensor(normal).unsqueeze(0)[:, :3, :, :].cuda())

  gt_normals = []
  alphas = []
  for fname in natsorted(os.listdir(gt_dir)):
    if fname.endswith("_normal.png"):
      normal = Image.open(gt_dir / fname)
      gt_normals.append(tf.to_tensor(normal).unsqueeze(0)[:, :3, :, :].cuda())
    elif fname.endswith("_alpha.png"):
      alpha = Image.open(gt_dir / fname)
      alpha_tensor = tf.to_tensor(alpha).unsqueeze(0)[:, :1, :, :].cuda()
      # should shape like [1,3,H,W]
      alpha_tensor = alpha_tensor.repeat(1,3,1,1)
      alphas.append(alpha_tensor)
    else:
      image = Image.open(gt_dir / fname)
      has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
      if has_alpha:
        alpha = image.getchannel("A")
        alpha_tensor = tf.to_tensor(alpha).unsqueeze(0)[:, :1, :, :].cuda()
        alpha_tensor = alpha_tensor.repeat(1,3,1,1)
        alphas.append(alpha_tensor)

  return render_normals, gt_normals, alphas


if __name__ == "__main__":
  # Set up command line argument parser
  parser = ArgumentParser(description="Testing script parameters")
  parser.add_argument("--model_path", type=str)
  parser.add_argument("--source_path", default="../../datasets/shiny_blender", type=str)
  
  cmdlne_string = sys.argv[1:]
  args = parser.parse_args(cmdlne_string)

  evaluate(Path(args.model_path), Path(args.source_path))