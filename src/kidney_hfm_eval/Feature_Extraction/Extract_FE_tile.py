import gc, argparse, os
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
import torch
from huggingface_hub import PyTorchModelHubMixin
from .model_builders_tile import build_ProvGigapath, build_UNI, build_UNI2_h, build_Virchow, build_Virchow2, build_Hibou, build_Hoptimus, build_SP22M, build_SP85M                             
from .utils import reset_cuda, list_images, ensure_out_dir, open_rgb

# 1) Turn off TF32 (cuBLAS & cuDNN)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 2) Deterministic kernels where possible
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# 3) Deterministic GEMMs in cuBLAS (PyTorch 1.12+ / 2.x)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # or ":4096:8" (bigger workspace)

# 4) Optional: keep FP32 math path stable
try:
    torch.set_float32_matmul_precision("high")   # avoid "medium" on some builds
except Exception:
    pass

# 5) Keep threads stable (helps CPU-side transforms)
torch.set_num_threads(1)

# -------------------- Registry & runner --------------------

def model_builders_tile(device: torch.device) -> Dict[str, Callable[[], Tuple[Callable, Callable]]]:
    """
    Returns a dict of name -> zero-arg callable that, when called,
    constructs (encode_fn, teardown_fn). This defers model construction
    until the moment we actually run that model.
    """
    BASE_MODEL_WEIGHT = "/orange/pinaki.sarder/harishwarreddy.k/Validation_of_Hist_FMs_new/Model_weights"

    return {
        # Prov-Gigapath
        "Prov-Gigapath": lambda: build_ProvGigapath(f"{BASE_MODEL_WEIGHT}/prov-gigapath.bin", device),

        # UNI family
        "UNI":        lambda: build_UNI(f"{BASE_MODEL_WEIGHT}/UNI.bin", device),
        "UNI2-h":     lambda: build_UNI2_h(f"{BASE_MODEL_WEIGHT}/UNI2.bin", device),

        # Virchow family
        "Virchow":    lambda: build_Virchow(f"{BASE_MODEL_WEIGHT}/Virchow.bin", device),
        "Virchow2":   lambda: build_Virchow2(f"{BASE_MODEL_WEIGHT}/Virchow2.bin", device),

        # Hibou (B/L via local dirs)
        "Hibou-B":    lambda: build_Hibou(f"{BASE_MODEL_WEIGHT}/Hibou-B", device),
        "Hibou-L":    lambda: build_Hibou(f"{BASE_MODEL_WEIGHT}/Hibou-L", device),

        # H-optimus family
        "H-optimus-0": lambda: build_Hoptimus(f"{BASE_MODEL_WEIGHT}/H-optimus-0.bin", device),
        "H-optimus-1": lambda: build_Hoptimus(f"{BASE_MODEL_WEIGHT}/H-optimus-1.bin", device),


        # SP family
        "SP22M":      lambda: build_SP22M(f"{BASE_MODEL_WEIGHT}/SP22M.bin", device),
        "SP85M":      lambda: build_SP85M(f"{BASE_MODEL_WEIGHT}/SP85M.bin", device),
    }

def run_extraction_model_by_model(
    input_root: str,
    model_builder_registry: Dict[str, Callable[[], Tuple[Callable[[Image.Image], torch.Tensor], Callable]]],
    output_roots: Dict[str, str],
    selected_models: Tuple[str, ...] = (),
    skip_existing: bool = True,
):
    """
    Build ONE model at a time, run it over ALL images, save features, teardown, then move on.
    This minimizes VRAM usage.

    input_root: dataset root to scan for images
    model_builder_registry: name -> zero-arg callable returning (encode_fn, teardown_fn)
    output_roots: name -> base out dir for that model
    selected_models: optional tuple of model names to run; if empty, run all in registry
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine which models to run
    all_model_names = list(model_builder_registry.keys())
    if selected_models:
        model_names = [m for m in selected_models if m in model_builder_registry]
    else:
        model_names = all_model_names

    # Pre-enumerate all images once (list of (abs_path, rel_dir, stem))
    image_list = list(list_images(input_root))
    if not image_list:
        print(f"[warn] No images found under: {input_root}")
        return

    # Ensure output roots exist
    for name in model_names:
        base = output_roots.get(name, None)
        if base is None:
            print(f"[warn] No output root configured for model '{name}', skipping.")
            continue
        Path(base).mkdir(parents=True, exist_ok=True)

    for name in model_names:
        base_out = output_roots.get(name, None)
        if base_out is None:
            continue

        print(f"\n=== Running {name} → {base_out}")
        # Build the model NOW (lazy)
        try:
            encode_fn, teardown_fn = model_builder_registry[name]()
        except Exception as e:
            print(f"[warn] Failed to build model '{name}': {e}")
            continue

        processed = 0
        failed = 0

        for abs_path, rel_dir, stem in image_list:
            out_dir = ensure_out_dir(base_out, rel_dir)
            out_path = os.path.join(out_dir, stem + ".pt")
            if skip_existing and os.path.exists(out_path):
                continue

            try:
                img = open_rgb(abs_path)
            except Exception as e:
                failed += 1
                print(f"[warn] Failed to open {abs_path}: {e}")
                continue

            try:
                with torch.inference_mode():
                    emb = encode_fn(img)
                torch.save(emb, out_path)
                processed += 1
                if processed % 500 == 0:
                    print(f"… {name}: saved {processed} embeddings")
            except Exception as e:
                failed += 1
                print(f"[warn] {name} failed on {abs_path}: {e}")

        # Teardown & free VRAM before the next model
        try:
            teardown_fn()
        except Exception:
            pass
        reset_cuda()
        print(f" {name} finished (saved: {processed}, failed: {failed})")

    print("\n Feature extraction complete for selected models.")

# -------------------- Example usage --------------------
if __name__ == "__main__":
    reset_cuda()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Sequential FM feature extraction (VRAM-friendly).")
    parser.add_argument("--input_root", type=str, default=None)
    parser.add_argument("--output_base", type=str, default=None)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models to run (space-separated). If omitted, runs all.")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip saving if output .pt exists.")
    args = parser.parse_args()

    INPUT_ROOT = args.input_root
    OUTPUT_BASE = args.output_base

    # Build lazy registry (nothing loaded yet)
    builders = model_builders(DEVICE)

    # Where to save per model
    output_roots = {
        "UNI":           f"{OUTPUT_BASE}/UNI",
        "UNI2-h":        f"{OUTPUT_BASE}/UNI2-h",
        "Virchow":       f"{OUTPUT_BASE}/Virchow",
        "Virchow2":      f"{OUTPUT_BASE}/Virchow2",
        "Hibou-B":       f"{OUTPUT_BASE}/Hibou-B",
        "Hibou-L":       f"{OUTPUT_BASE}/Hibou-L",
        "H-optimus-0":   f"{OUTPUT_BASE}/H-optimus-0",
        "H-optimus-1":   f"{OUTPUT_BASE}/H-optimus-1",
        "Prov-Gigapath": f"{OUTPUT_BASE}/Prov-Gigapath",
        "SP22M":         f"{OUTPUT_BASE}/SP22M",
        "SP85M":         f"{OUTPUT_BASE}/SP85M",
    }

    selected = tuple(args.models) if args.models else tuple()
    run_extraction_model_by_model(
        input_root=INPUT_ROOT,
        model_builder_registry=builders,
        output_roots=output_roots,
        selected_models=selected,
        skip_existing=args.skip_existing,
    )