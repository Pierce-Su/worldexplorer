#!/usr/bin/env python3
"""
Generate 180-degree counterpart images for every sample in the dataset using
Google Gemini API (Nano Banana Pro / gemini-3-pro-image-preview).

Each original image is sent to the model with a prompt asking for the exact opposite
field of view (same standing position, 180° rotation). Outputs are saved so they
pair easily with originals (same folder structure + manifest JSON).

Requires: GOOGLE_API_KEY or GEMINI_API_KEY in the environment, and:
  pip install google-genai

Usage:
  python generate_180_views.py --dataset_dir data/curated_set
  python generate_180_views.py --dataset_dir data/curated_set --output_dir data/curated_set/views_180
  python generate_180_views.py --dataset_dir data/curated_set --indices 0 1 2-5
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Default prompt for 180° view generation
PROMPT_180 = (
    "Treat the attached image as the 0-degree reference point of a 3D environment. "
    "Generate a new image from the exact same standing position but rotated 180 degrees "
    "to show the opposite field of view. Reference the lighting, weather, and "
    "architectural/natural style of the original. If the sun or a light source is visible "
    "in the original, ensure the shadows in this new view reflect that the light is now "
    "behind the camera."
)

# Gemini model: Nano Banana Pro (Gemini 3 Pro Image Preview)
GEMINI_IMAGE_MODELS = {
    "pro": "gemini-3-pro-image-preview",
    "flash": "gemini-2.5-flash-image",
}


def _parse_indices(specs: List[str]) -> Set[int]:
    """Parse --indices into a set of integers. Supports ranges like 0-24 or 0:24."""
    out: Set[int] = set()
    range_re = re.compile(r"^(-?\d+)[-:](-?\d+)$")
    for spec in specs:
        s = str(spec).strip()
        m = range_re.match(s)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= hi:
                out.update(range(lo, hi + 1))
            else:
                out.update(range(hi, lo + 1))
        else:
            out.add(int(s))
    return out


def get_variants_for_sample(sample: dict, dataset_dir: Path) -> List[Tuple[str, Path]]:
    """
    Return (variant_name, image_path) for each variant that has an existing image.
    Matches run_batch_pipeline logic: photorealistic/stylized with filename.
    """
    out: List[Tuple[str, Path]] = []
    for variant in ("photorealistic", "stylized"):
        if variant not in sample:
            continue
        path_val = f"{variant}/{sample[variant]['filename']}"
        p = dataset_dir / path_val if not os.path.isabs(path_val) else Path(path_val)
        if p.exists():
            out.append((variant, p))
    return out


def generate_180_image(
    image_path: Path,
    prompt: str,
    model: str,
    api_key: Optional[str],
) -> Optional[bytes]:
    """
    Call Gemini API with the given image and prompt; return generated image bytes or None.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("ERROR: google-genai is required. Run: pip install google-genai", file=sys.stderr)
        return None

    client = genai.Client(api_key=api_key)

    # Load image (PIL or bytes); API accepts PIL Image or inline_data
    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
    except Exception:
        with open(image_path, "rb") as f:
            image = f.read()
        # genai can accept bytes with mime type
        image = (image, "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg")

    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
    )

    response = client.models.generate_content(
        model=model,
        contents=[prompt, image],
        config=config,
    )

    # response.parts is the standard iterator; fallback to candidates[0].content.parts
    parts = getattr(response, "parts", None)
    if parts is None and response.candidates:
        parts = response.candidates[0].content.parts
    if not parts:
        return None

    for part in parts:
        if getattr(part, "inline_data", None) is not None:
            # Prefer raw bytes (inline_data.data); else as_image() and encode to PNG
            if hasattr(part.inline_data, "data") and part.inline_data.data:
                return part.inline_data.data
            img = getattr(part, "as_image", None) and part.as_image()
            if img is not None:
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 180° counterpart images for dataset samples using Gemini Nano Banana Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to curated_set (metadata.json + images)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for 180° images. Default: <dataset_dir>/views_180")
    parser.add_argument("--indices", type=str, nargs="+", default=None,
                        help="Only process these sample indices (e.g. 0 2 5 or 0-24)")
    parser.add_argument("--model", type=str, default="pro",
                        help=f"Gemini model for image generation pro (default), flash")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override the default 180° prompt")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Keep processing remaining samples if one fails")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only list jobs, do not call API")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.json"
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "views_180"
    prompt = args.prompt or PROMPT_180

    if args.model not in GEMINI_IMAGE_MODELS:
        print(f"ERROR: Invalid model: {args.model}. Must be one of: {list(GEMINI_IMAGE_MODELS.keys())}", file=sys.stderr)
        sys.exit(1)
    else:
        args.model = GEMINI_IMAGE_MODELS[args.model]

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    samples = metadata.get("samples", [])
    if not samples:
        print("ERROR: No samples in metadata.json", file=sys.stderr)
        sys.exit(1)

    if args.indices is not None:
        idx_set = _parse_indices(args.indices)
        samples = [s for s in samples if s.get("index") in idx_set]
        print(f"Filtered to {len(samples)} samples (indices: {sorted(idx_set)})")
    if not samples:
        print("ERROR: No samples to process after filtering", file=sys.stderr)
        sys.exit(1)

    jobs: List[Tuple[dict, str, Path]] = []
    for sample in samples:
        for variant_name, variant_image_path in get_variants_for_sample(sample, dataset_dir):
            jobs.append((sample, variant_name, variant_image_path))
    if not jobs:
        print("ERROR: No jobs. Each sample must have at least one variant image (photorealistic/stylized).", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: Set GOOGLE_API_KEY or GEMINI_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset: {dataset_dir}")
    print(f"Output:  {output_dir}")
    print(f"Jobs:    {len(jobs)}")
    if args.dry_run:
        for i, (sample, variant_name, image_path) in enumerate(jobs, 1):
            print(f"  {i}. index_{sample['index']:04d} [{variant_name}] {image_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for job_num, (sample, variant_name, image_path) in enumerate(jobs, 1):
        index = sample.get("index", 0)
        # Paired naming: views_180/<variant>/index_XXXX_180.png
        out_subdir = output_dir / variant_name
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / f"index_{index:04d}_180.png"

        print(f"[{job_num}/{len(jobs)}] index_{index:04d} [{variant_name}] -> {out_path}")
        try:
            data = generate_180_image(image_path, prompt, args.model, api_key)
            if not data:
                print(f"  ERROR: No image in response")
                manifest.append({
                    "index": index,
                    "variant": variant_name,
                    "original_path": str(image_path),
                    "path_180": str(out_path),
                    "status": "failed",
                    "error": "no image in response",
                })
                if not args.continue_on_error:
                    sys.exit(1)
                continue
            with open(out_path, "wb") as f:
                f.write(data)
            manifest.append({
                "index": index,
                "variant": variant_name,
                "original_path": str(image_path),
                "path_180": str(out_path),
                "status": "success",
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            manifest.append({
                "index": index,
                "variant": variant_name,
                "original_path": str(image_path),
                "path_180": str(out_path),
                "status": "failed",
                "error": str(e),
            })
            if not args.continue_on_error:
                raise

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"prompt": prompt, "model": args.model, "pairs": manifest}, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    success = sum(1 for p in manifest if p.get("status") == "success")
    print(f"Done: {success}/{len(manifest)} succeeded.")


if __name__ == "__main__":
    main()
