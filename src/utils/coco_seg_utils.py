"""
coco_seg_utils.py — Parse COCO instance annotations → binary masks for GradCAM evaluation.

Usage:
    from src.utils.coco_seg_utils import load_coco_masks

    samples = load_coco_masks(
        ann_file='data/coco2017/annotations/instances_val2017.json',
        img_dir='data/coco2017/val2017',
        n=200,
        min_area=1024,
        seed=42,
    )
    # samples: list of {'image_path': str, 'mask': np.ndarray (H, W) bool}

Requires: pycocotools (preferred) or PIL (fallback polygon rasterization).
Install: pip install pycocotools
"""

from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np


def _poly_to_mask(polygons: list, height: int, width: int) -> np.ndarray:
    """Rasterize polygon list to binary mask using PIL (no pycocotools needed)."""
    from PIL import Image, ImageDraw
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 6:
            coords = list(zip(poly[0::2], poly[1::2]))
            draw.polygon(coords, fill=1)
    return np.array(mask, dtype=bool)


def _rle_to_mask(rle: dict, height: int, width: int) -> np.ndarray:
    """Decode COCO RLE to binary mask (simple implementation)."""
    try:
        from pycocotools import mask as mask_util
        return mask_util.decode(rle).astype(bool)
    except ImportError:
        pass

    # Manual RLE decode (counts are run-length encoded column-major)
    counts = rle['counts']
    if isinstance(counts, str):
        # Encoded RLE — needs pycocotools
        raise ImportError(
            "Encoded RLE requires pycocotools. Install: pip install pycocotools")
    flat = np.zeros(height * width, dtype=bool)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            flat[pos:pos + count] = True
        pos += count
    # COCO stores column-major (Fortran order)
    return flat.reshape((height, width), order='F')


def load_coco_masks(
    ann_file: str,
    img_dir: str,
    n: int = 200,
    min_area: float = 1024.0,
    seed: int = 42,
) -> list[dict]:
    """
    Load up to `n` COCO val images that have at least one instance annotation
    with area > min_area. Returns merged binary mask (union of all qualifying
    instances in the image).

    Parameters
    ----------
    ann_file : path to instances_val2017.json
    img_dir  : path to val2017/ image directory
    n        : number of images to sample
    min_area : minimum annotation area (px²) to include
    seed     : random seed for reproducible sampling

    Returns
    -------
    list of dicts:
        {
            'image_path': str,
            'mask':       np.ndarray (H, W, dtype=bool)  — union of large instances
            'image_id':   int,
        }
    """
    ann_path = Path(ann_file)
    img_path = Path(img_dir)
    assert ann_path.exists(), f"Annotation file not found: {ann_path}"
    assert img_path.exists(), f"Image directory not found: {img_path}"

    print(f"[coco_seg_utils] Loading {ann_path.name} …", flush=True)
    with open(ann_path, 'r') as f:
        data = json.load(f)

    # Build image id → info mapping
    id_to_info = {img['id']: img for img in data['images']}

    # Group annotations by image, filter by area
    img_to_anns: dict[int, list] = {}
    for ann in data['annotations']:
        if ann.get('area', 0) < min_area:
            continue
        iid = ann['image_id']
        if iid not in img_to_anns:
            img_to_anns[iid] = []
        img_to_anns[iid].append(ann)

    # Keep only images where file exists
    valid_ids = []
    for iid, anns in img_to_anns.items():
        info = id_to_info.get(iid)
        if info is None:
            continue
        fp = img_path / info['file_name']
        if fp.exists():
            valid_ids.append(iid)

    print(f"[coco_seg_utils] {len(valid_ids):,} images with area>{min_area:.0f} px²")

    # Sample n images
    rng = random.Random(seed)
    rng.shuffle(valid_ids)
    selected = valid_ids[:n]

    results = []
    for iid in selected:
        info    = id_to_info[iid]
        anns    = img_to_anns[iid]
        H, W    = info['height'], info['width']
        fp      = str(img_path / info['file_name'])

        merged_mask = np.zeros((H, W), dtype=bool)
        for ann in anns:
            seg = ann.get('segmentation', [])
            try:
                if isinstance(seg, list):
                    # Polygon format
                    m = _poly_to_mask(seg, H, W)
                elif isinstance(seg, dict):
                    # RLE format
                    m = _rle_to_mask(seg, H, W)
                else:
                    continue
                merged_mask |= m
            except Exception:
                continue

        if merged_mask.any():
            results.append({
                'image_path': fp,
                'mask':       merged_mask,
                'image_id':   iid,
            })

    print(f"[coco_seg_utils] Loaded {len(results)} samples")
    return results
