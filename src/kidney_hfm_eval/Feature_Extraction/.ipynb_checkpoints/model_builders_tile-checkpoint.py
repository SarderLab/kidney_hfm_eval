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
from .utils import _standard_transform

def teardown_model(model):
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def build_UNI(weights_path: str, device: torch.device):
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        init_values=1e-5,
        dynamic_img_size=True
    )
    sd = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tfm = _standard_transform((224, 224), transforms.InterpolationMode.BILINEAR)

    # @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        with torch.inference_mode():
            x = tfm(img).unsqueeze(0).to(device)
            y =  model(x).cpu()
        return y

    def teardown():
        teardown_model(model)
    return encode, teardown

def build_UNI2_h(weights_path: str, device: torch.device):
    model = timm.create_model("vit_giant_patch14_224", init_values=1e-5,
                              pretrained=False, num_classes=0,
                              img_size=224, patch_size=14,
                              num_heads=24, embed_dim=1536, depth= 24,
                              dynamic_img_size=True, mlp_ratio=2.66667*2,
                              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                              reg_tokens=8, no_embed_class=True)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tfm = _standard_transform((224, 224), transforms.InterpolationMode.BILINEAR)

    # @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        with torch.inference_mode():
            x = tfm(img).unsqueeze(0).to(device)
            y =  model(x).cpu()
        return y

    def teardown():
        teardown_model(model)
    return encode, teardown

def build_Virchow(weights_path: str, device: torch.device):
    model = timm.create_model("vit_huge_patch14_224", init_values=1e-5,
                              pretrained=False, num_classes=0,
                              img_size=224, patch_size=14,
                              dynamic_img_size=True, mlp_ratio=5.3375,
                              mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tfm = _standard_transform((224, 224), transforms.InterpolationMode.BICUBIC)

    def encode(img: Image.Image) -> torch.Tensor:
        x = tfm(img).unsqueeze(0).to(device, non_blocking=True)  # [1,3,224,224]
        with torch.inference_mode():
            feats = model.forward_features(x)
            # feats = y if y.dim() == 3 else model(x)                      # hope for [B,N,D]
        cls = feats[:, 0]                        # [B,D]
        patch_mean = feats[:, 1:, :].mean(1)     # [B,D]
        emb = torch.cat([cls, patch_mean], dim=-1)  # [B,2560]
        # print(emb.shape)
        return emb.cpu()

    def teardown():
        teardown_model(model)
    return encode, teardown

def build_Virchow2(weights_path: str, device: torch.device):
    model = timm.create_model("vit_huge_patch14_224", init_values=1e-5,
                              pretrained=False, num_classes=0,
                              img_size=224, patch_size=14,
                              dynamic_img_size=True, mlp_ratio=5.3375,
                              mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
                              reg_tokens=4)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tfm = _standard_transform((224, 224), transforms.InterpolationMode.BICUBIC)

    # @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        x = tfm(img).unsqueeze(0).to(device, non_blocking=True)
        with torch.inference_mode():
            feats = model.forward_features(x)
        cls = feats[:, 0]                        # [B,D]
        patch_mean = feats[:, 5:, :].mean(1)     # [B,D]
        emb = torch.cat([cls, patch_mean], dim=-1)  # [B,2560]
        # print(emb.shape)
        return emb.cpu()

    def teardown():
        teardown_model(model)
    return encode, teardown

def build_Hibou(local_dir: str, device: torch.device):
    # Works for both Hibou-B and Hibou-L if you pass the correct directory
    model = AutoModel.from_pretrained(local_dir, local_files_only=True, trust_remote_code=True)
    # Use the canonical processor id to get the right preprocessing
    processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
    model.to(device).eval()

    # @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]   # CLS
        return emb.cpu()

    def teardown():
        teardown_model(model)
    return encode, teardown

def build_Hoptimus(weights_path: str, device: torch.device):
    model = timm.create_model("vit_giant_patch14_reg4_dinov2", init_values=1e-5,
                              pretrained=False, num_classes=0, img_size=224,
                              patch_size=14, dynamic_img_size=False)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.707223, 0.578729, 0.703617),
            std=(0.211883, 0.230117, 0.177517)
        ),
    ])

    def encode(img: Image.Image) -> torch.Tensor:
        with torch.inference_mode():
            x = tfm(img).unsqueeze(0).to(device)
            y = model(x).detach().cpu()
        return y

    def teardown():
        teardown_model(model)
    return encode, teardown

def build_ProvGigapath(weights_path: str, device: torch.device):
    model = timm.create_model("vit_giant_patch14_dinov2", init_values=1e-5, 
                        pretrained=False, num_classes=0,            
                        img_size=224, patch_size=16,
                        num_heads = 24, dynamic_img_size=True, mlp_ratio=5.33334)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        with torch.inference_mode():
            x = tfm(img).unsqueeze(0).to(device)
            y = model(x).cpu()
        return y

    def teardown():
        teardown_model(model)
    return encode, teardown


class SP22M(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vision_transformer.vit_small(num_classes=0)

    def forward(self, x):
        return self.encoder(x)

def build_SP22M(weights_path: str, device: torch.device):
    model = SP22M()  # now checkpoint keys match
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tfm = _standard_transform((224, 224), transforms.InterpolationMode.BILINEAR)

    # @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        with torch.inference_mode():
            x = tfm(img).unsqueeze(0).to(device)
            y = model(x).cpu()
        return y

    def teardown():
        teardown_model(model)
    return encode, teardown

class SP85M(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        # ViT small → feature output (no head)
        self.encoder = vision_transformer.vit_base(num_classes=0)

    def forward(self, x):
        return self.encoder(x)

def build_SP85M(weights_path: str, device: torch.device):
    model = SP85M()  # now checkpoint keys match
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    tfm = _standard_transform((224, 224), transforms.InterpolationMode.BILINEAR)

    # @torch.inference_mode()
    def encode(img: Image.Image) -> torch.Tensor:
        with torch.inference_mode():
            x = tfm(img).unsqueeze(0).to(device)
            y = model(x).cpu()
        return y

    def teardown():
        teardown_model(model)
    return encode, teardown
