import os, gc, argparse
from pathlib import Path
from typing import Callable, Dict, Tuple
from . import vision_transformer
import torch.nn as nn
import torch
import timm
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from timm.layers import SwiGLUPacked
import os, torch
from huggingface_hub import PyTorchModelHubMixin
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

def reset_cuda():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(" CUDA memory fully reset")

def list_images(input_root: str):
    input_root = str(input_root)
    for dirpath, _, filenames in os.walk(input_root):
        imgs = [f for f in filenames if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            continue
        rel_dir = os.path.relpath(dirpath, input_root)
        abs_dir = os.path.abspath(dirpath)
        for fname in imgs:
            abs_path = os.path.join(abs_dir, fname)
            stem = os.path.splitext(fname)[0]
            yield abs_path, rel_dir, stem

def ensure_out_dir(base: str, rel_dir: str) -> str:
    out_dir = os.path.join(base, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def open_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# -------------------- Model builder utilities --------------------
# Each builder returns: (encode_fn: PIL->Tensor, teardown_fn: Callable)

def _standard_transform(resize_to=(224, 224), interp=transforms.InterpolationMode.BILINEAR,
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.Resize(resize_to, interpolation=interp, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
